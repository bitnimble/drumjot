#!/usr/bin/env bash
# Periodic real-song eval for VISIBILITY DURING TRAINING -- NOT model selection.
#
# The held-out ParaDB-102 + MDB sets stay clean (they never feed keep-best / early-stop /
# threshold tuning); this just logs their F1 every time a new checkpoint_every save appears,
# so you can watch real-domain quality climb instead of inferring it from the noisy per-stem
# val. Run it ALONGSIDE training (run_bestmodel.sh backgrounds it); it exits when that
# launcher is gone. Output accumulates in $OUT.periodic-eval.log.
#
# Caveat: each eval briefly shares the GPU with training (~a few min every ~10 epochs).
#   usage: eval_periodic.sh OUT_DIR [EVAL_N]
set -u
OUT=${1:?usage: eval_periodic.sh OUT_DIR [EVAL_N]}
EVAL_N=${2:-6}
DS=/codebox-workspace/datasets
REPO=/home/bitnimble/code/drumjot
cd "$REPO" || exit 1
PLOG="$OUT.periodic-eval.log"
MC=/codebox-workspace/drumjot/models-cache
seen=""

# Eval the current checkpoint iff its model.pt is newer than the last one we logged.
eval_if_new() {
  local ckpt mt ep
  ckpt="$OUT/model.pt"
  [ -f "$ckpt" ] || return
  mt=$(stat -c %Y "$ckpt" 2>/dev/null || echo 0)
  [ "$mt" = "$seen" ] && return
  seen=$mt
  ep=$(grep -oE "checkpoint saved @ epoch [0-9]+" "$OUT.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  {
    echo "=== periodic eval @ epoch ${ep:-?} ($(date -u +%H:%M)) [VISIBILITY ONLY] ==="
    scripts/sandbox-run env CKPT="$OUT" WORK="$DS/paradb_eval_periodic" \
      MODELS_DIR="$MC" DRUMJOT_MERT_CACHE="$DS/mert_cache" \
      bash "$REPO/training/scripts/eval_paradb_parallel.sh" "$EVAL_N" 2>&1 |
      rg -N "oracle|^  (k|s|t|hc|ho|rd|cr) |macro|WRONG|leak" | tail -16
    echo "-- MDB --"
    scripts/sandbox-run env MODELS_DIR="$MC" DRUMJOT_MERT_CACHE="$DS/mert_cache" \
      PYTHONPATH="$REPO/training:$REPO/dsp" \
      python3 "$REPO/training/scripts/eval_mdb.py" --checkpoint "$OUT" --lanes hc,ho,rd,cr \
      --sep-root "$DS/mdb-sep/perstem" 2>&1 | tail -10
  } >> "$PLOG" 2>&1
}

# Track the launcher so we exit when training (+ its final eval) is done.
while pgrep -f "tmp/run_bestmodel.sh" >/dev/null 2>&1; do
  eval_if_new
  sleep 120
done
# Final pass: catch a checkpoint that landed during the last sleep. Matters when the launcher
# exited abnormally (no final clean eval to fall back on); a no-op if we already logged it.
eval_if_new
