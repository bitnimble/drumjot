#!/usr/bin/env bash
# Download + unpack E-GMD v1.0.0 (Magenta, fixed public URL).
#
# Usage: fetch_egmd.sh [DEST_DIR]
#   DEST_DIR defaults to $DRUMJOT_EGMD_PARENT or /codebox-workspace/datasets.
#
# The zip is ~96 GB; extracted (mostly WAV) is ~120-140 GB. `unzip`/`bsdtar`
# are not always present (e.g. the sandbox lacks them), so extraction uses
# python3's zipfile. Idempotent: `curl -C -` resumes a partial download and
# the download is skipped if the zip is already complete; extractall is safe
# to re-run.
set -euo pipefail

URL="https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"
DEST="${1:-${DRUMJOT_EGMD_PARENT:-/codebox-workspace/datasets}}"
ZIP="$DEST/e-gmd-v1.0.0.zip"

mkdir -p "$DEST"
echo "downloading E-GMD (~96 GB) -> $ZIP"
curl -L --fail -C - -o "$ZIP" "$URL"

echo "extracting via python zipfile -> $DEST"
python3 - "$ZIP" "$DEST" <<'PY'
import sys
import zipfile

zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest)
    print("extracted", len(z.namelist()), "entries")
PY

ROOT="$DEST/e-gmd-v1.0.0"
echo
echo "done. E-GMD root: $ROOT"
echo "point training at it:"
echo "  export DRUMJOT_EGMD=$ROOT"
echo "  # or add to training/data_paths.toml:  egmd = \"$ROOT\""
