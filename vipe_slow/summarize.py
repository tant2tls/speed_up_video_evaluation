#!/usr/bin/env python
"""Roll up ViPE shards into one run summary.

Unlike the eval suite there are no scores to reduce -- ViPE writes pose/depth
artifacts per video. What is worth aggregating is the timing, and specifically
the numbers the GPU-count argument needs:

  cpu_seconds_per_video      W_cpu in the model  (measured, per-worker CPU time)
  gpu_busy_seconds_per_video W_gpu in the model  (measured, integrated NVML util)
  straggler_s                the slowest shard -- what wall clock actually equals
  shard_imbalance            slowest/fastest; the cost of a static stride

Wall clock is set by the slowest worker, not the mean, so `straggler_s` and
`shard_imbalance` are reported rather than averaged away. A static stride over
videos of unequal cost shows up here as an imbalance > 1.0; a dynamic queue
should push it toward 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--launcher", default="unknown")
    ap.add_argument("--world-size", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--wall", type=float, default=0.0)
    ap.add_argument("--peak-load", type=float, default=None)
    args = ap.parse_args()

    files = sorted(args.shards.glob("shard_*.json"))
    if not files:
        raise SystemExit(f"no shard_*.json under {args.shards}")
    shards = [json.loads(f.read_text()) for f in files]

    walls = [s.get("compute_s", 0.0) for s in shards]
    cpu = [s.get("timing", {}).get("cpu_total_s") or 0.0 for s in shards]
    gpu = [s.get("timing", {}).get("gpu_busy_total_s") or 0.0 for s in shards]
    n = sum(s.get("timing", {}).get("n", 0) for s in shards)
    steady = [
        s.get("timing_steady_state", {}).get("wall_mean_s")
        for s in shards
        if s.get("timing_steady_state", {}).get("wall_mean_s")
    ]
    util = [
        s.get("timing", {}).get("gpu_occupancy")
        for s in shards
        if s.get("timing", {}).get("gpu_occupancy") is not None
    ]

    summary = {
        "launcher": args.launcher,
        "world_size": args.world_size or len(shards),
        "threads_per_worker": args.threads,
        "n_shards": len(shards),
        "n_videos": n,
        "wall_s": round(args.wall, 3) if args.wall else None,
        "peak_loadavg": args.peak_load,
        "shard_compute_s": [round(w, 3) for w in walls],
        "straggler_s": round(max(walls), 3) if walls else None,
        "shard_imbalance": (
            round(max(walls) / min(walls), 3) if walls and min(walls) > 0 else None
        ),
        "model_load_s": [s.get("model_load_s") for s in shards],
        "cpu_total_s": round(sum(cpu), 3),
        "gpu_busy_total_s": round(sum(gpu), 3),
        "cpu_seconds_per_video": round(sum(cpu) / n, 3) if n else None,
        "gpu_busy_seconds_per_video": round(sum(gpu) / n, 3) if n else None,
        "gpu_occupancy_mean": round(sum(util) / len(util), 3) if util else None,
        "throughput_videos_per_s": round(n / args.wall, 4) if args.wall else None,
        "steady_state_wall_mean_s": round(sum(steady) / len(steady), 3) if steady else None,
        "videos_per_shard": [len(s.get("videos", [])) for s in shards],
        "thread_checks": [s.get("thread_check") for s in shards],
        # No scores: ViPE emits pose/depth artifacts, not metrics.
        "final_scores": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f" merged       : {len(files)} shards, {n} videos")
    print(f" videos/shard : {summary['videos_per_shard']}")
    if summary["shard_imbalance"]:
        print(
            f" imbalance    : {summary['shard_imbalance']}x "
            f"(slowest {summary['straggler_s']}s / fastest {min(walls):.1f}s)"
        )
    print(
        f" per video    : cpu={summary['cpu_seconds_per_video']}s  "
        f"gpu_busy={summary['gpu_busy_seconds_per_video']}s  "
        f"occupancy={summary['gpu_occupancy_mean']}"
    )
    print(f" summary      : {args.out}")

    bad = [t for t in summary["thread_checks"] if t and t.get("mismatch")]
    if bad:
        print(f" !! {len(bad)} shard(s) had a THREAD MISMATCH -- timings suspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
