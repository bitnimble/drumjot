"""Extract ONLY the mix audio + annotations the trainer uses, from the STAR
parts. `star.index` pairs `annotation/*.txt` with `audio/mix/*.flac`, so the
per-instrument stem buckets (original_mix / non_drum / original_drum /
re_synthesized_drum) are dead weight: this keeps ~39 GB instead of ~181 GB.

Reads the `STAR_Drums_full.zip.part-a?` chunks as one virtual seekable file
and extracts only the needed members' bytes (no 181 GB reassembly), preserving
the `data/<split>/.../{annotation,audio/mix}` layout so `star.index(out_dir)`
works unchanged.

Usage: extract_star_mix.py <parts_dir> <out_dir>
"""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_star_subset import MultiPartFile  # noqa: E402


def wanted(name: str) -> bool:
    """Members the trainer needs: mix flac + annotations (+ tiny CSV/mapping
    metadata). Stem buckets like audio/original_mix/ are excluded."""
    if name.endswith("/"):
        return False
    if "/audio/mix/" in name and name.endswith(".flac"):
        return True
    if "/annotation/" in name and name.endswith(".txt"):
        return True
    return name.endswith(".csv") or name.endswith("class_mappings.py")


def main() -> None:
    parts_dir, out_dir = sys.argv[1], sys.argv[2]
    parts = sorted(
        os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if ".zip.part-" in f
    )
    print("parts:", [os.path.basename(p) for p in parts], flush=True)
    z = zipfile.ZipFile(MultiPartFile(parts))
    members = sorted(n for n in z.namelist() if wanted(n))
    n_flac = sum(1 for n in members if n.endswith(".flac"))
    n_txt = sum(1 for n in members if n.endswith(".txt"))
    print(
        f"selected {len(members)} members "
        f"({n_flac} mix flac, {n_txt} annotations) -> {out_dir}",
        flush=True,
    )
    for i, m in enumerate(members):
        z.extract(m, out_dir)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(members)}", flush=True)
    print("DONE. dataset root:", out_dir, flush=True)


if __name__ == "__main__":
    main()
