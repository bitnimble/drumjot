#!/usr/bin/env bash
# Parallel ParaDB adaptive eval (research harness): classify songs by MERT-cache
# state, pin all uncached onto ONE encoder worker (the rest are --require-cached =
# never load the encoder), run N workers over disjoint song-lists in parallel, then
# merge the per-worker dumps into one report. Cache-hit workers load only the tiny
# heads (windows batched per forward), so N share one GPU; only worker 0 loads MERT.
#
# Run it where eval_paradb.py runs (a GPU + the separator + drumjot_training, e.g.
# the sandbox: `scripts/sandbox-run bash training/scripts/eval_paradb_parallel.sh`).
# Paths default to the research box; override via env.
#   eval_paradb_parallel.sh [N] [maps-dir]
set -u
N=${1:-8}
MAPS_DIR=${2:-${MAPS_DIR:-/codebox-workspace/datasets/paradb_tier1_102/zips}}
CKPT=${CKPT:-/codebox-workspace/datasets/ab3_prev}
PRED=${PRED:-/codebox-workspace/checkpoints/ovn3080/mixed_c3000_h256_s1/param_predictor_a2md.joblib}
STEMS=${STEMS:-/codebox-workspace/datasets/paradb_tier1_200/stems_cache}
LANES=${LANES:-hc,ho,rd,cr}
WORK=${WORK:-/codebox-workspace/datasets/paradb_parallel}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYPATH=${PYTHONPATH:-$HERE/..:$HERE/../../dsp}
export PYTHONPATH=$PYPATH
rm -rf "$WORK"; mkdir -p "$WORK"

echo "== classify (cached vs needs-encoding) =="
python3 "$HERE/classify_paradb_cache.py" "$MAPS_DIR" "$CKPT" "$STEMS" "$N" "$WORK" "$LANES" || exit 1

echo "== launch $N workers =="
pids=()
for i in $(seq 0 $((N - 1))); do
  rc=""; [ "$i" -ne 0 ] && rc="--require-cached"
  OMP_NUM_THREADS=1 python3 "$HERE/eval_paradb.py" \
    --maps-dir "$MAPS_DIR" --checkpoint "$CKPT" --param-predictor "$PRED" \
    --oracle-report --lanes "$LANES" --stems-cache "$STEMS" \
    --maps-list "$WORK/maps_$i.txt" --dump "$WORK/shard_$i.pkl" $rc \
    > "$WORK/shard_$i.out" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "== workers done (fail=$fail) =="

echo "== merge =="
python3 "$HERE/merge_paradb_shards.py" "$WORK"/shard_*.pkl --predictor
