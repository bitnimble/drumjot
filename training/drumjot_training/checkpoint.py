"""Save / load a trained run: model weights + everything inference needs.

A run produces two files in `out_dir`:
  - `model.pt`   the `MultiLaneHeads` state_dict
  - `meta.json`  the lane vocab, encoder name/layer/fps, head shape, and the
                 tuned per-lane peak thresholds

`meta.json` is the handoff contract for inference and for the eventual
transcriber integration: it pins which encoder + layer produced the features
and how to peak-pick each lane. `run_metadata` is pure (host-testable);
`save`/`load` lazily import torch.
"""
from __future__ import annotations

import json
from pathlib import Path

from drumjot_training.config import Config
from drumjot_training.embeddings import FEAT_DIM


def run_metadata(cfg: Config, thresholds: dict[str, float], in_dim: int = FEAT_DIM) -> dict:
    """JSON-serializable description of a trained model (the bits not in the
    weights). Thresholds are coerced to plain floats."""
    return {
        "lanes": list(cfg.lanes),
        "encoder": cfg.encoder,
        "encoder_layer": cfg.encoder_layer,
        # Per-lane MERT layer map {lane: layer} when the heads span >1 layer; None
        # for the single-layer model (every head reads `encoder_layer`). Inference
        # routes each head to its layer when this is set. (See model.MultiLaneHeads.)
        "lane_layers": cfg.lane_layer_map() if cfg.is_multilayer() else None,
        "encoder_fps": cfg.encoder_fps,
        "sigma_frames": cfg.sigma_frames,
        "peak_threshold": cfg.peak_threshold,
        "peak_min_distance_s": cfg.peak_min_distance_s,
        "onset_tolerance_s": cfg.onset_tolerance_s,
        "head_hidden": cfg.head_hidden,
        "head_layers": cfg.head_layers,
        "in_dim": in_dim,
        "high_band": cfg.high_band,  # whether the heads expect the high-band block
        "aux_activity": True,  # heads carry the auxiliary ring-activity output
        "cymbal_softmax": cfg.cymbal_softmax,  # rd/cr decoded as a joint 3-way softmax

        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }


def save(out_dir: str | Path, model, cfg: Config, thresholds: dict[str, float],
         in_dim: int = FEAT_DIM) -> Path:
    """Write `model.pt` + `meta.json` into `out_dir`; return the dir."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(json.dumps(run_metadata(cfg, thresholds, in_dim), indent=2))
    return out


def load(out_dir: str | Path, device: str = "cpu"):
    """Rebuild `MultiLaneHeads` from a saved run; returns `(model, meta)`."""
    import torch

    from drumjot_training.model import MultiLaneHeads

    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    model = MultiLaneHeads(
        in_dim=meta["in_dim"],
        hidden=meta["head_hidden"],
        num_layers=meta["head_layers"],
        lane_names=tuple(meta["lanes"]),
        lane_layers=meta.get("lane_layers"),  # per-lane-layer routing (None on old/single-layer)
    )
    sd = torch.load(out / "model.pt", map_location=device)
    # Tolerate three benign kinds of state_dict drift; raise on anything else:
    #  - missing `.act.*`: older checkpoints predate the auxiliary activity head;
    #    inference uses only the onset path, so those are fine.
    #  - missing `.calib.*`: older checkpoints predate the per-clip calibration head;
    #    a zero-init calib is identity, so those load as plain (uncalibrated) heads.
    #  - UNEXPECTED whole heads for lanes outside `meta["lanes"]`: a checkpoint
    #    may carry MORE heads than we load -- trained with the full lane vocab
    #    but reported on a subset (the `--lanes` runs), or loaded after a vocab
    #    reduction (the hp->hc merge). Those extra heads are simply ignored.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    keep = set(meta["lanes"])

    def _extra_head(k: str) -> bool:
        p = k.split(".")
        return len(p) >= 2 and p[0] == "heads" and p[1] not in keep

    bad = ([k for k in missing if ".act." not in k and ".calib." not in k]
           + [k for k in unexpected if not _extra_head(k)])
    if bad:
        raise RuntimeError(f"checkpoint mismatch in {out}: {bad}")
    model.eval()
    # `map_location` only places the loaded state_dict; `load_state_dict` copies
    # into the CPU-constructed params, so the module itself is still on CPU. Move
    # it to the requested device or every head forward runs on CPU (~2s/window).
    return model.to(device), meta
