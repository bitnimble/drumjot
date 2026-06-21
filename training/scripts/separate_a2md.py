"""Separation-aware A2MD: run our drum separator over A2MD's real (YouTube) song
MIXES so the training audio carries the SAME separation artifacts as ParaDB (the
deployment domain) -- the missing real-domain diversity for the param predictor.

A2MD tracks are full mixes (not drums-only like E-GMD/ENST), so each goes through
BOTH stages, exactly like ParaDB in eval_paradb: BS-Roformer (mix -> drum stem) ->
MDX23C (drum stem -> 5 per-instrument stems). Mono at SEP_SR to match how ParaDB
is separated. The aligned drum MIDI is copied as the label (parsed later by the
a2md loader). Subset by alignment quality with --max-dist (A2MD's dist0pNN bucket).

Per-song (full songs already keep the GPU busy; no batching) and RESUMABLE: a
rerun skips tracks whose stems + MIDI already exist, so you can ctrl-C (e.g. to
use the GPU) and pick up where it left off.

Output (perstem-consistent; a future a2md.perstem_index consumes it):
  annotation/<id>.mid                copied aligned MIDI (full-song; drum channel parsed later)
  audio/sep_drum/<id>.flac           BS-Roformer combined drum stem
  audio/perstem/<pitch>/<id>.flac    MDX23C 5-class (pitch in k/s/h/c/t)

Run on a GPU box with the transcriber `app` separator + MODELS_DIR; PYTHONPATH
must include dsp + training.

  MODELS_DIR=<cache> PYTHONPATH=dsp:training python3 training/scripts/separate_a2md.py \
      --zip /codebox-workspace/datasets/a2md/a2md_public.zip \
      --out /codebox-workspace/datasets/a2md_sep [--max-dist 0.10] [--scratch-dir /tmp/a2md]
"""
import argparse
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault("DRUMJOT_SEP_LOG_LEVEL", "WARNING")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "transcriber"))  # transcriber app

SEP_SR = 44100  # match eval_paradb (mono, full band)
PITCHES = ("k", "s", "h", "c", "t")


def _dist(name: str) -> float:
    m = re.search(r"dist0p(\d\d)", name)
    return int(m.group(1)) / 100.0 if m else 9.9


def tracks_from_zip(zf: zipfile.ZipFile, max_dist: float):
    """Pair `align_mid` + `ytd_audio` by their shared `<idx>_<MSDID>` key, keep
    tracks whose alignment-distance bucket is <= `max_dist`. Returns sorted
    (id, audio_member, midi_member, dist)."""
    mids, auds = {}, {}
    for n in zf.namelist():
        if n.endswith(".mid") and "/align_mid/" in n:
            mids[Path(n).stem.replace("align_mid_", "")] = (n, _dist(n))
        elif n.endswith(".mp3") and "/ytd_audio/" in n:
            auds[Path(n).stem.replace("ytd_audio_", "")] = n
    out = []
    for key, (midn, dist) in mids.items():
        if key in auds and dist <= max_dist:
            out.append((key, auds[key], midn, dist))
    return sorted(out)


def out_paths(out: Path, tid: str):
    return {"mid": out / "annotation" / f"{tid}.mid",
            "drum": out / "audio" / "sep_drum" / f"{tid}.flac",
            "per": {p: out / "audio" / "perstem" / p / f"{tid}.flac" for p in PITCHES}}


def is_done(paths) -> bool:
    return (paths["mid"].exists() and paths["drum"].exists()
            and all(p.exists() for p in paths["per"].values()))


def _write_flac(y, sr, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, format="FLAC")


def _fmt_eta(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True, help="a2md_public.zip")
    ap.add_argument("--out", type=Path, required=True, help="output a2md_sep dir")
    ap.add_argument("--max-dist", type=float, default=0.10, help="keep alignment buckets <= this (0.10 = dist0p00+0p10)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tracks (0 = all in the buckets)")
    ap.add_argument("--scratch-dir", type=Path, default=None, help="LOCAL fast dir for separator intermediates")
    args = ap.parse_args()

    import librosa

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir is not None:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    zf = zipfile.ZipFile(args.zip)
    tracks = tracks_from_zip(zf, args.max_dist)
    if args.limit:
        tracks = tracks[: args.limit]
    pending = [t for t in tracks if not is_done(out_paths(out, t[0]))]
    log(f"A2MD: {len(tracks)} tracks (dist<={args.max_dist}); {len(tracks) - len(pending)} done, {len(pending)} pending")
    if not pending:
        log("nothing to do")
        return

    from app.pipeline.separate import Separator
    sep = Separator()
    sep.load()
    try:  # same batch-perf tweak as separate_egmd: skip per-call gc+empty_cache
        from audio_separator.separator.common_separator import CommonSeparator
        CommonSeparator.clear_gpu_cache = lambda self: None  # type: ignore[method-assign]
    except Exception as e:  # noqa: BLE001
        log(f"  (could not disable per-call cache clear: {e!r})")

    t0 = time.perf_counter()
    for i, (tid, audn, midn, dist) in enumerate(pending, 1):
        try:
            with tempfile.TemporaryDirectory(dir=args.scratch_dir) as td:
                tdp = Path(td)
                mp3 = tdp / "in.mp3"
                mp3.write_bytes(zf.read(audn))
                y, _ = librosa.load(str(mp3), sr=SEP_SR, mono=True)
                peak = float(np.max(np.abs(y)) or 1.0)
                mix = tdp / "mix.wav"
                sf.write(str(mix), (y / peak * 0.98).astype(np.float32), SEP_SR)

                drum = sep.run_stems_all(mix, tdp / "s1", build_no_drums=False).drum_stem
                per = sep.run_stems_per(drum, tdp / "s2", build_residual=False).per_instrument
                d_arr, d_sr = sf.read(str(drum), always_2d=True)
                _write_flac(d_arr, d_sr, out_paths(out, tid)["drum"])  # persist combined drum stem
                for p in PITCHES:
                    src = per.get(p)
                    if src and Path(src).exists():
                        arr, ssr = sf.read(str(src), always_2d=True)
                        _write_flac(arr, ssr, out_paths(out, tid)["per"][p])
                    else:  # separator skips a near-silent stem -> write silence (accurate)
                        _write_flac(np.zeros((int(len(y)), 1), dtype="float32"), SEP_SR, out_paths(out, tid)["per"][p])
            # MIDI copied LAST = completion marker (is_done true only after this)
            ap_mid = out_paths(out, tid)["mid"]
            ap_mid.parent.mkdir(parents=True, exist_ok=True)
            ap_mid.write_bytes(zf.read(midn))
        except Exception as e:  # noqa: BLE001  keep the unattended run alive; track retried on resume
            log(f"  !! {tid} FAILED: {e!r} (retried on resume)")
            continue
        eta = _fmt_eta((time.perf_counter() - t0) / i * (len(pending) - i))
        log(f"[{i}/{len(pending)}] {tid} (dist {dist:.2f}) done  eta {eta}")

    log(f"DONE -> {out}")


if __name__ == "__main__":
    main()
