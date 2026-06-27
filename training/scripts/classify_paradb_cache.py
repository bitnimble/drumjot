"""Split ParaDB songs by whether their MERT features are fully cached, then write
N per-worker maps-lists for a parallel eval. Worker 0 (the only encoder worker)
gets ALL uncached songs + a share of cached; workers 1..N-1 get cached songs only.
This keeps concurrent MERT loads to a single GPU worker -- cache-only workers never
load the encoder, so N of them share one GPU. Driven by eval_paradb_parallel.sh.

  classify_paradb_cache.py <maps-dir> <checkpoint> <stems-cache> <N> <out-dir> [lanes]
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import embeddings, inference  # noqa: E402
from drumjot_training.paradb import PERSTEM_TO_LANES  # noqa: E402


def main():
    maps_dir = Path(sys.argv[1])
    checkpoint = sys.argv[2]
    stems_cache = Path(sys.argv[3])
    n = int(sys.argv[4])
    out_dir = Path(sys.argv[5])
    lanes = set(sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[6] else None
    out_dir.mkdir(parents=True, exist_ok=True)

    # stems carrying any requested lane (h -> hc/ho, c -> rd/cr, ...)
    stem_pitches = [p for p, lns in PERSTEM_TO_LANES.items() if lanes is None or (set(lns) & lanes)]
    _, meta = inference.load_model(checkpoint, "cpu")  # meta only; encoder stays lazy (no MERT load)
    enc = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])

    cached, uncached = [], []
    for zp in sorted(maps_dir.glob("*.zip")):
        stems = [s for p in stem_pitches if (s := stems_cache / f"{zp.stem}.{p}.flac").exists()]
        ok = bool(stems) and all(embeddings.windows_cached(s, enc, meta) for s in stems)
        (cached if ok else uncached).append(str(zp))
    print(f"{len(cached) + len(uncached)} songs: {len(cached)} fully cached, "
          f"{len(uncached)} need encoding", flush=True)

    # worker 0 = encoder (all uncached + every N-th cached); 1..N-1 = cached only
    lists: list[list[str]] = [[] for _ in range(n)]
    lists[0].extend(uncached)
    for i, zp in enumerate(cached):
        lists[i % n].append(zp)
    for i, lst in enumerate(lists):
        (out_dir / f"maps_{i}.txt").write_text("\n".join(lst) + ("\n" if lst else ""))
        print(f"  worker {i}: {len(lst)} songs{' (encoder)' if i == 0 else ''}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(
        {"n": n, "cached": len(cached), "uncached": len(uncached)}))


if __name__ == "__main__":
    main()
