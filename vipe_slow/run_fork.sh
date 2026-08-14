#!/usr/bin/env bash
# ViPE bash-fork launcher: one process per GPU, static shard, thread-pinned.
#
# This is the pattern the production launcher used (`node1.sh` in the original
# project): a bash loop over GPUs, each worker pinned with CUDA_VISIBLE_DEVICES.
# The one thing added here is the thread pin, because the original set no
# OMP_NUM_THREADS and that turned out to be the dominant cold-start cost at 8-way
# concurrency: an unpinned `python` defaults to torch's `nproc/2` = 128 threads,
# so 8 workers request up to 1024 threads against a 198-CPU cgroup and every one
# of them gets CFS-throttled during model load.
#
# Pass THREADS=0 to reproduce the unpinned original and measure the storm.
#
# Usage:
#   ./run_fork.sh --videos test_video --gpus 0,1,2,3
#   THREADS=0 ./run_fork.sh --videos test_video --gpus 0,1,2,3,4,5,6,7
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python}"

VIDEOS=""
GPUS=""
OUT_DIR=""
TAG="fork"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos) VIDEOS="$2"; shift 2 ;;
    --gpus)   GPUS="$2"; shift 2 ;;
    --out)    OUT_DIR="$2"; shift 2 ;;
    --tag)    TAG="$2"; shift 2 ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$VIDEOS" ]] || { echo "usage: $0 --videos DIR [--gpus 0,1,..] [--out DIR]" >&2; exit 2; }

# Resolve to an absolute path BEFORE the `cd` below, or a relative --videos
# silently resolves against the wrong directory and the worker sees no files.
VIDEOS="$(cd "$(dirname "$VIDEOS")" && pwd)/$(basename "$VIDEOS")"
[[ -e "$VIDEOS" ]] || { echo "no such path: $VIDEOS" >&2; exit 2; }
if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

if [[ -z "$GPUS" ]]; then
  GPUS="$(nvidia-smi -L | awk -F'[ :]' '{print $2}' | paste -sd, -)"
fi
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
WORLD_SIZE="${#GPU_ARR[@]}"

QUOTA="$("$PYTHON" -c "import sys; sys.path.insert(0,'$REPO'); from common.resources import cpu_quota; print(cpu_quota())")"
if [[ -z "${THREADS:-}" ]]; then
  THREADS=$(( QUOTA / WORLD_SIZE ))
  [[ $THREADS -lt 1 ]] && THREADS=1
fi

OUT_DIR="${OUT_DIR:-$REPO/results/vipe_${TAG}_g${WORLD_SIZE}}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " launcher     : bash-fork (one ViPE process per GPU)"
echo " videos       : $VIDEOS"
echo " GPUs         : $GPUS  (world_size=$WORLD_SIZE)"
echo " CPU quota    : $QUOTA   (nproc reports $(nproc))"
if [[ "$THREADS" == "0" ]]; then
  echo " threads/proc : UNPINNED (torch default ~128; expect a cold-start storm)"
else
  echo " threads/proc : $THREADS   ($QUOTA / $WORLD_SIZE)"
fi
echo " out          : $OUT_DIR"
echo "=============================================================="

cd "$HERE"
START=$EPOCHREALTIME

( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 1; done ) \
  > "$OUT_DIR/loadavg.txt" 2>/dev/null &
SAMPLER=$!

WORKER_PIDS=()
for rank in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$rank]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$HERE/run_infer.py" \
      --videos "$VIDEOS" \
      --rank "$rank" \
      --world-size "$WORLD_SIZE" \
      --threads "$THREADS" \
      --out "$OUT_DIR/shard_${rank}.json" \
      --output-path "$OUT_DIR/artifacts" \
      "${EXTRA[@]}" > "$LOG_DIR/rank${rank}_gpu${gpu}.log" 2>&1
  ) &
  WORKER_PIDS+=("$!")
done

FAIL=0
for i in "${!WORKER_PIDS[@]}"; do
  if ! wait "${WORKER_PIDS[$i]}"; then
    echo " !! rank $i exited non-zero" >&2
    FAIL=1
  fi
done

kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

END=$EPOCHREALTIME
WALL="$(awk -v a="$END" -v b="$START" 'BEGIN{printf "%.3f", a-b}')"
# `head -1` closing the pipe early SIGPIPEs `sort`, and under `pipefail` that
# non-zero status would abort the script AFTER every worker had succeeded.
# `|| true` keeps a cosmetic stat from failing the run.
PEAK="$(awk '{print $2}' "$OUT_DIR/loadavg.txt" 2>/dev/null | sort -g -r | head -1 || true)"

echo "--------------------------------------------------------------"
printf " wall clock   : %.2fs\n" "$WALL"
echo " peak loadavg : ${PEAK:-n/a}  (quota $QUOTA)"

"$PYTHON" "$HERE/summarize.py" \
  --shards "$OUT_DIR" \
  --out "$OUT_DIR/vipe_summary.json" \
  --launcher "bash-fork" \
  --world-size "$WORLD_SIZE" \
  --threads "$THREADS" \
  --wall "$WALL" \
  ${PEAK:+--peak-load "$PEAK"}

[[ $FAIL -eq 0 ]] || exit 1
