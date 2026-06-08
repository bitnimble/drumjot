#!/usr/bin/env bash
# Download + reassemble + unpack STAR Drums *full* (Zenodo 15690078, BSD-3).
#
# Usage: fetch_star.sh [DEST_DIR]
#   DEST_DIR defaults to $DRUMJOT_STAR_PARENT or /codebox-workspace/datasets.
#
# ~181 GB across 6 split parts (part-aa..af) that concatenate into one zip.
# Idempotent: `curl -C -` resumes each part. Verifies the reassembled zip
# against the published sha256, then extracts via python zipfile (unzip is
# not always present). Parts are removed after a verified reassembly.
set -euo pipefail

REC="https://zenodo.org/api/records/15690078/files"
DEST="${1:-${DRUMJOT_STAR_PARENT:-/codebox-workspace/datasets}}"
WORK="$DEST/star_full"
mkdir -p "$WORK"

# Download the 6 parts in parallel: Zenodo/GCS throttles per-connection, so
# concurrent transfers aggregate more bandwidth. Each curl still resumes
# (-C -) and retries; per-part progress goes to its own log.
echo "downloading 6 parts in parallel ..."
pids=()
for p in aa ab ac ad ae af; do
    f="$WORK/STAR_Drums_full.zip.part-$p"
    curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors -C - \
        -sS -o "$f" "$REC/STAR_Drums_full.zip.part-$p/content" \
        > "$WORK/dl-$p.log" 2>&1 &
    pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
[ "$fail" -eq 0 ] || { echo "ERROR: a part failed; see $WORK/dl-*.log"; exit 1; }
echo "all parts downloaded"

curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors \
    -o "$WORK/STAR_Drums_full.zip.sha256" "$REC/STAR_Drums_full.zip.sha256/content"

ZIP="$WORK/STAR_Drums_full.zip"
echo "reassembling parts -> $ZIP"
cat "$WORK"/STAR_Drums_full.zip.part-a? > "$ZIP"

echo "verifying sha256 ..."
python3 - "$ZIP" "$WORK/STAR_Drums_full.zip.sha256" <<'PY'
import hashlib
import sys

zip_path, sha_path = sys.argv[1], sys.argv[2]
want = open(sha_path).read().split()[0].strip()
h = hashlib.sha256()
with open(zip_path, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
got = h.hexdigest()
print("expected", want)
print("got     ", got)
assert got == want, "sha256 mismatch"
print("sha256 OK")
PY

echo "removing parts (verified zip is the canonical artifact)"
rm -f "$WORK"/STAR_Drums_full.zip.part-*

echo "extracting -> $WORK"
python3 - "$ZIP" "$WORK" <<'PY'
import sys
import zipfile

zp, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zp) as z:
    z.extractall(dest)
    print("extracted", len(z.namelist()), "entries")
PY

echo
echo "done. STAR full extracted under $WORK"
echo "  set DRUMJOT_STAR=<extracted root> (or add to data_paths.toml)"
