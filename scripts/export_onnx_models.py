#!/usr/bin/env python3
"""Export the canonical ONNX model set for shipping (fp16 by default).

Every ML model the transcriber runs on onnxruntime, exported once here and
uploaded to HF so the app downloads ready `.onnx` instead of exporting at first
run. fp16 is the shipping format (GPU EPs only; validated corr >= 0.99998 vs
fp32 on CUDA). Pass --fp32 for full precision, --only to subset.

  transcriber/.venv/bin/python3 scripts/export_onnx_models.py [OUT_DIR] [--fp32] [--only sep,onset,adtof,beat,lyrics]

Runs on CPU (torch.onnx.export). Sequential with gc so peak RAM ~ one model.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "training", REPO / "dsp"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(REPO / "transcriber"))

SEP_CACHE = Path("/codebox-workspace/drumjot/models-cache")
ONSET_CKPT = Path("/codebox-workspace/datasets/ab3_prev")
LYRICS_EN = "facebook/wav2vec2-large-robust-ft-libri-960h"
LYRICS_MMS = "MahmoudAshraf/mms-300m-1130-forced-aligner"


def _report(path: Path):
    print(f"  ok  {path.name}  {path.stat().st_size >> 20} MiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="/codebox-workspace/drumjot/onnx-export")
    ap.add_argument("--fp32", action="store_true", help="export fp32 instead of fp16")
    ap.add_argument("--only", default="", help="comma list of: sep,onset,adtof,beat,lyrics")
    args = ap.parse_args()

    fp16 = not args.fp32
    tag = "fp16" if fp16 else "fp32"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    def want(name: str) -> bool:
        return not only or name in only

    print(f"exporting {tag} set -> {out}", flush=True)

    if want("sep"):
        print("[separation]", flush=True)
        from app.pipeline.provision import yaml_for_ckpt
        from app.pipeline.separation.export import export_body
        from app.pipeline.separation.loader import load_model

        for ckpt in ("drumsep_5stems_mdx23c_jarredou.ckpt", "model_bs_roformer_sw.ckpt"):
            loaded = load_model(SEP_CACHE / ckpt, SEP_CACHE / yaml_for_ckpt(ckpt), device="cpu")
            path = export_body(loaded, out / f"{Path(ckpt).stem}.{tag}.onnx", fp16=fp16)
            _report(path)
            del loaded
            gc.collect()

    if want("onset"):
        print("[learned onsets]", flush=True)
        import json

        from app.pipeline.onset_onnx.export import export_heads, export_mert

        meta = json.loads((ONSET_CKPT / "meta.json").read_text())
        layer = meta["encoder_layer"]
        _report(export_mert(out / f"mert_L{layer}.{tag}.onnx", layer, name=meta["encoder"], fp16=fp16))
        gc.collect()
        path, _ = export_heads(ONSET_CKPT, out / f"onset_heads.{tag}.onnx", fp16=fp16)
        _report(path)
        gc.collect()

    if want("adtof"):
        print("[adtof]", flush=True)
        from app.pipeline.adtof_onnx import export_adtof

        _report(export_adtof(out / f"adtof_frame_rnn.{tag}.onnx", fp16=fp16))
        gc.collect()

    if want("beat"):
        print("[beats]", flush=True)
        from app.pipeline.beat_onnx import export_beatthis

        _report(export_beatthis(out / f"beat_this.{tag}.onnx", fp16=fp16))
        gc.collect()

    if want("lyrics"):
        print("[lyrics]", flush=True)
        from app.pipeline.lyrics_onnx import _sanitize, export_ctc_model

        for model_path in (LYRICS_EN, LYRICS_MMS):
            path = export_ctc_model(model_path, out / f"ctc_align__{_sanitize(model_path)}.{tag}.onnx", fp16=fp16)
            _report(path)
            gc.collect()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
