"""Convert a loss-A/B experiment checkpoint into the standard checkpoint dir.

The A/B harness saves a single `.pt` holding `{state_dict, lanes, hidden,
num_layers, thresholds}` -- not the `model.pt` + `meta.json` dir that
`inference.load_model` / `eval_paradb.py` read. This rewrites one into the
standard format so it drops into the eval + param-predictor pipeline.

The encoder fields aren't stored in the A/B `.pt`; they come from `Config`
defaults (the A/B runs trained with them). `in_dim` is recovered from the head
GRU weights, which decides whether the high-band block is expected.

Usage:
  PYTHONPATH=dsp:training python3 training/scripts/convert_ab_checkpoint.py \
      --in <ab_checkpoint.pt> --out <std_checkpoint_dir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import checkpoint  # noqa: E402
from drumjot_training.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Convert an A/B .pt to a standard checkpoint dir")
    ap.add_argument("--in", dest="src", required=True, help="A/B checkpoint .pt")
    ap.add_argument("--out", required=True, help="output checkpoint dir (model.pt + meta.json)")
    args = ap.parse_args()

    import torch

    obj = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    lanes = tuple(obj["lanes"])
    hidden = int(obj["hidden"])
    layers = int(obj["num_layers"])
    in_dim = int(sd[f"heads.{lanes[0]}.gru.weight_ih_l0"].shape[1])
    cfg = Config(lanes=lanes, head_hidden=hidden, head_layers=layers, high_band=in_dim > 1024)
    meta = checkpoint.run_metadata(cfg, obj.get("thresholds", {}), in_dim=in_dim)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out / "model.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/model.pt + meta.json", flush=True)
    print(f"  lanes={list(lanes)} in_dim={in_dim} high_band={in_dim > 1024} "
          f"hidden={hidden} layers={layers}", flush=True)
    print(f"  encoder={meta['encoder']} layer={meta['encoder_layer']} fps={meta['encoder_fps']}", flush=True)
    print(f"  thresholds={meta['thresholds']}", flush=True)


if __name__ == "__main__":
    main()
