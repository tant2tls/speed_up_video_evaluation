#!/usr/bin/env bash
# Baseline launcher: fork one Python process per GPU. The production pattern.
#
# This is a faithful, cleaned-up version of the hand-rolled launcher this project
# actually shipped with (`ray_learning/movebench/run.sh`): a bash `for` loop, one
# process per GPU, each hard-pinned with CUDA_VISIBLE_DEVICES, then `wait`.
#
# Two things are deliberately DIFFERENT from the original, and both are bugs the
# original has:
#
#   1. The original set WORLD_SIZE=6 but looped `for gpu in 0 1`, so 4 of 6 data
#      shards were silently never evaluated. Here the shard count is *derived*
#      from the GPU list, so the two cannot disagree. (README "Tip 6".)
#   2. The original set no OMP_NUM_THREADS, so each worker defaulted to
#      torch's `nproc/2` = 128 threads; G of them oversubscribed the 198-CPU
#      cgroup by up to 5x and got silently CFS-throttled. Here threads are pinned
#      to (cgroup quota / G) by default. Pass THREADS=0 to reproduce the
#      unpinned behaviour and measure the thread-storm. (README "Tip 1", "Tip 4".)
#
# Usage:
#   ./run_fork.sh --dataset data/eval81 --gpus 0,1,2,3
#   THREADS=0 ./run_fork.sh --dataset data/eval81 --gpus 0,1,2,3   # unpinned baseline
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python}"

DATASET=""
GPUS=""
OUT_DIR=""
TAG="fork"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --out)     OUT_DIR="$2"; shift 2 ;;
    --tag)     TAG="$2"; shift 2 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$DATASET" ]] || { echo "usage: $0 --dataset DIR [--gpus 0,1,..] [--out DIR] [-- extra worker args]" >&2; exit 2; }

# Resolve to absolute BEFORE the `cd $REPO` below, so a relative --dataset given
# from another directory still points where the caller meant.
DATASET="$(cd "$(dirname "$DATASET")" && pwd)/$(basename "$DATASET")"
[[ -e "$DATASET" ]] || { echo "no such dataset path: $DATASET" >&2; exit 2; }
if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

# Default to every visible GPU -- the `--gpus all` habit this repo argues against.
# We keep it as the default *because* it is the default everywhere else; the
# sweep is what shows it is often wrong.
if [[ -z "$GPUS" ]]; then
  GPUS="$(nvidia-smi -L | awk -F'[ :]' '{print $2}' | paste -sd, -)"
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
WORLD_SIZE="${#GPU_ARR[@]}"
[[ $WORLD_SIZE -gt 0 ]] || { echo "no GPUs selected" >&2; exit 2; }

# ---------------------------------------------------------------------------
# The thread budget. Read the cgroup quota, NEVER nproc.
# ---------------------------------------------------------------------------
QUOTA="$("$PYTHON" -c "import sys; sys.path.insert(0,'$REPO'); from common.resources import cpu_quota; print(cpu_quota())")"
if [[ -z "${THREADS:-}" ]]; then
  THREADS=$(( QUOTA / WORLD_SIZE ))
  [[ $THREADS -lt 1 ]] && THREADS=1
fi

OUT_DIR="${OUT_DIR:-$REPO/results/movebench_${TAG}_g${WORLD_SIZE}}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " launcher     : bash-fork (one process per GPU)"
echo " dataset      : $DATASET"
echo " GPUs         : $GPUS  (world_size=$WORLD_SIZE)"
echo " CPU quota    : $QUOTA   (nproc reports $(nproc) -- the host, not us)"
if [[ "$THREADS" == "0" ]]; then
  echo " threads/proc : UNPINNED (torch default; expect a thread-storm at G>1)"
else
  echo " threads/proc : $THREADS   ($QUOTA / $WORLD_SIZE)"
  echo " total threads: $(( THREADS * WORLD_SIZE )) vs quota $QUOTA"
fi
echo " out          : $OUT_DIR"
echo "=============================================================="

cd "$REPO"
START=$EPOCHREALTIME

# Record the load average during the run: this is how the thread-storm is caught.
# `sort` is fed from a FILE, not a pipe from the still-running sampler -- reading a
# live pipe and closing it early kills the writer with SIGPIPE (rc=141), which
# once took down the merge step after every worker had already succeeded.
( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 1; done ) \
  > "$OUT_DIR/loadavg.txt" 2>/dev/null &
SAMPLER=$!

WORKER_PIDS=()
for rank in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$rank]}"
  log="$LOG_DIR/rank${rank}_gpu${gpu}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$HERE/worker.py" \
      --dataset "$DATASET" \
      --rank "$rank" \
      --world-size "$WORLD_SIZE" \
      --threads "$THREADS" \
      --out "$OUT_DIR/shard_${rank}.json" \
      "${EXTRA[@]}" > "$log" 2>&1
  ) &
  WORKER_PIDS+=("$!")
done

# Collect each worker's REAL exit code. A bare `wait` (what the original launcher
# used) returns the status of the last job only, so a dead worker reports success
# and its shard is silently missing from the merged result. That failure mode is
# structurally impossible in the Ray version, which raises RayActorError.
FAIL=0
for i in "${!WORKER_PIDS[@]}"; do
  if ! wait "${WORKER_PIDS[$i]}"; then
    echo " !! rank $i (pid ${WORKER_PIDS[$i]}) exited non-zero" >&2
    FAIL=1
  fi
done

kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

END=$EPOCHREALTIME
# bash-native float subtract (no `bc` in this container).
WALL="$(awk -v a="$END" -v b="$START" 'BEGIN{printf "%.3f", a-b}')"

echo "--------------------------------------------------------------"
printf " wall clock   : %.2fs\n" "$WALL"
# `head -1` closing the pipe early SIGPIPEs `sort`, and under `pipefail` that
# non-zero status would abort the script AFTER every worker had succeeded.
# `|| true` keeps a cosmetic stat from failing the run.
PEAK="$(awk '{print $2}' "$OUT_DIR/loadavg.txt" 2>/dev/null | sort -g -r | head -1 || true)"
echo " peak loadavg : ${PEAK:-n/a}  (quota $QUOTA)"

# Merge shards and emit the run summary.
"$PYTHON" "$HERE/merge.py" \
  --shards "$OUT_DIR" \
  --out "$OUT_DIR/summary.json" \
  --launcher "bash-fork" \
  --world-size "$WORLD_SIZE" \
  --threads "$THREADS" \
  --wall "$WALL" \
  ${PEAK:+--peak-load "$PEAK"}

if [[ $FAIL -ne 0 ]]; then
  echo " !! at least one worker failed -- see $LOG_DIR" >&2
  exit 1
fi
