#!/usr/bin/env python
"""One worker: score a shard of the dataset. Used by BOTH launchers.

This is deliberately the only file that computes anything. `run_fork.sh` spawns
G copies of it with `--rank/--world-size` (the production pattern); `run_ray.py`
imports `score_shard()` and calls it inside actors. So a wall-clock difference
between the two launchers cannot come from the arithmetic -- it can only come
from scheduling, thread placement, or process startup. That is the experiment.

Two sharding modes, because the difference between them is a finding:

  static   video_names[rank::world_size] -- the production suite's approach.
           Perfectly balanced *if and only if* every video costs the same.
  queue    pull the next index from a shared counter. Needs a coordinator, which
           is exactly the thing Ray gives you for free (see run_ray.py).

`--threads` is mandatory-by-default rather than inherited, because an unset
thread count is an unmeasured variable (README "Tip 1").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set thread env BEFORE torch is imported anywhere: torch fixes its intra-op
# pool size at import time, so exporting OMP_NUM_THREADS afterwards is a silent
# no-op. This ordering bug has cost this project a full session before.
def _pin_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads for this worker. Default: cgroup quota / world_size. "
        "Pass 0 to leave torch's default alone (the unpinned baseline).",
    )
    ap.add_argument("--out", type=Path, required=True, help="output json path")
    ap.add_argument("--metrics", default="clip,epe,lpips,ssim,psnr")
    ap.add_argument("--device-policy", default="original", choices=["original", "gpu"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap dataset size")
    ap.add_argument("--pinned-memory", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from common.resources import cpu_quota, threads_per_worker

    if args.threads is None:
        args.threads = threads_per_worker(args.world_size)
    if args.threads > 0:
        _pin_threads(args.threads)

    # torch import happens only now, after the env is set.
    import torch

    from common.timing import SampleTimer, check_thread_mismatch, thread_env_report
    from movebench.metrics import ScoreConfig, list_pairs, load_models, score_one_video

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cfg = ScoreConfig(
        metrics=tuple(m.strip() for m in args.metrics.split(",") if m.strip()),
        device_policy=args.device_policy,
        max_frames=args.max_frames,
        pinned_memory=args.pinned_memory,
    )

    pairs = list_pairs(args.dataset)
    if args.limit:
        pairs = pairs[: args.limit]
    shard = pairs[args.rank :: args.world_size]

    mismatch = check_thread_mismatch(args.threads) if args.threads > 0 else {}
    if mismatch.get("mismatch"):
        print(
            f"[rank {args.rank}] !! THREAD MISMATCH: requested {mismatch['requested']}, "
            f"torch reports {mismatch['actual_torch_threads']}. "
            "Every timing below is suspect.",
            flush=True,
        )

    print(
        f"[rank {args.rank}/{args.world_size}] {len(shard)}/{len(pairs)} videos | "
        f"threads={args.threads} (quota={cpu_quota()}) | "
        f"torch={torch.get_num_threads()} | "
        f"cuda_devices={torch.cuda.device_count()} | "
        f"policy={cfg.device_policy}",
        flush=True,
    )

    t_load0 = time.perf_counter()
    models = load_models(cfg)
    load_s = time.perf_counter() - t_load0
    print(f"[rank {args.rank}] models loaded in {load_s:.1f}s", flush=True)

    timer = SampleTimer(use_cuda=torch.cuda.is_available())
    results = []
    t0 = time.perf_counter()
    for i, (name, raw, gen) in enumerate(shard):
        with timer.sample(name):
            results.append(score_one_video(name, raw, gen, models, cfg))
        s = timer.samples[-1]
        print(
            f"[rank {args.rank}] {i + 1}/{len(shard)} {name} "
            f"wall={s.wall_s:.2f}s cpu={s.cpu_s:.1f}s "
            f"gpu_busy={s.gpu_busy_s if s.gpu_busy_s is None else round(s.gpu_busy_s, 2)}s "
            f"gpu_util={s.gpu_util_mean}% cores_busy={s.cpu_par:.1f}",
            flush=True,
        )
    total = time.perf_counter() - t0
    timer.close()

    payload = {
        "rank": args.rank,
        "world_size": args.world_size,
        "threads_requested": args.threads,
        "model_load_s": round(load_s, 3),
        "compute_s": round(total, 3),
        "env": thread_env_report(),
        "thread_check": mismatch,
        "timing": timer.summary(),
        "timing_steady_state": timer.steady_state_summary(skip=1),
        "per_sample": [s.as_dict() for s in timer.samples],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"[rank {args.rank}] DONE {len(shard)} videos in {total:.1f}s "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
