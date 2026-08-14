#!/usr/bin/env python
"""Merge per-shard results into one score set, and summarize the run.

Two things happen here, and only one of them is bookkeeping.

1. **The reduction is a hand-written tree reduce**, not `sum(all_scores)/N`.
   Merging per-rank partial results *is* a collective operation -- the same
   `reduce` that gradient averaging uses -- and the production suite did it by
   opening N JSON files and averaging them by hand. Implementing it as a proper
   tree reduce (log2(N) rounds, pairwise associative merge of
   (sum, count) accumulators) makes two properties explicit that the hand merge
   left implicit:
     - it is **order-independent up to float associativity**: we merge
       (sum, count) pairs, never means-of-means, so an unequal shard split
       cannot skew the result the way averaging per-rank means does;
     - it is the shape that generalizes to a real `ray.util.collective.reduce`
       if the merge ever needs to happen on-device instead of via files.

   The mean-of-means bug is worth naming because the original had it: with
   shards of 5 and 3 videos, averaging the two rank means weights each video
   1/10 and 1/6 instead of 1/8. It only vanishes when every shard is the same
   size -- true for a static stride over equal-cost videos, false the moment a
   dynamic queue is used.

2. **FVD is reduced separately** because it is not an average. It is a Frechet
   distance between two Gaussians fitted to I3D embeddings over the *whole* set,
   so it cannot be computed per video and then averaged -- the embeddings must be
   concatenated first. Any eval harness that shards FVD by averaging per-shard
   FVDs is computing a different (and wrong) statistic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# the from-scratch tree reduce
# ---------------------------------------------------------------------------
Accum = dict[str, tuple[float, int]]  # metric -> (sum, count)


def to_accum(scores: dict[str, list[float]]) -> Accum:
    return {k: (float(np.sum(v)), len(v)) for k, v in scores.items() if len(v)}


def merge_pair(a: Accum, b: Accum) -> Accum:
    """The associative binary operator. This is the whole collective."""
    out: Accum = dict(a)
    for k, (s, c) in b.items():
        ps, pc = out.get(k, (0.0, 0))
        out[k] = (ps + s, pc + c)
    return out


def tree_reduce(accums: list[Accum]) -> tuple[Accum, list[str]]:
    """Pairwise-merge in log2(N) rounds. Returns (result, trace).

    The trace exists so the reduction tree is visible in the report rather than
    asserted -- it prints which ranks combined in which round.
    """
    if not accums:
        return {}, []
    level = list(accums)
    trace = []
    labels = [f"r{i}" for i in range(len(level))]
    rnd = 0
    while len(level) > 1:
        rnd += 1
        nxt, nxt_labels = [], []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(merge_pair(level[i], level[i + 1]))
                nxt_labels.append(f"({labels[i]}+{labels[i + 1]})")
            else:
                nxt.append(level[i])
                nxt_labels.append(labels[i])
        trace.append(f"round {rnd}: {len(level)} -> {len(nxt)}  {' '.join(nxt_labels)}")
        level, labels = nxt, nxt_labels
    return level[0], trace


def finalize(acc: Accum) -> dict[str, float]:
    return {k: s / c for k, (s, c) in acc.items() if c}


# ---------------------------------------------------------------------------
# FVD: a set-level statistic, reduced over concatenated embeddings
# ---------------------------------------------------------------------------
def frechet_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two embedding sets."""
    from scipy import linalg

    m1, m2 = x1.mean(0), x2.mean(0)
    s1 = np.cov(x1, rowvar=False)
    s2 = np.cov(x2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def reduce_fvd(shards: list[dict]) -> float | None:
    raw, gen = [], []
    for sh in shards:
        for r in sh.get("results", []):
            lg = r.get("fvd_logits")
            if lg:
                raw.append(np.asarray(lg["raw"], dtype=np.float64).reshape(-1))
                gen.append(np.asarray(lg["gen"], dtype=np.float64).reshape(-1))
    # A Frechet distance needs enough samples to estimate a covariance; below
    # ~dim samples it is dominated by estimation noise, so we refuse rather than
    # emit a confidently wrong number.
    if len(raw) < 8:
        return None
    return frechet_distance(np.stack(raw), np.stack(gen))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", required=True, type=Path, help="dir of shard_*.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--launcher", default="unknown")
    ap.add_argument("--world-size", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--wall", type=float, default=0.0)
    ap.add_argument("--peak-load", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    files = sorted(args.shards.glob("shard_*.json"))
    if not files:
        raise SystemExit(f"no shard_*.json under {args.shards}")
    shards = [json.loads(f.read_text()) for f in files]

    # Per-video scores, keyed by name -- the parity ground truth.
    per_video: dict[str, dict[str, float]] = {}
    accums = []
    for sh in shards:
        collected: dict[str, list[float]] = {}
        for r in sh.get("results", []):
            per_video[r["name"]] = r["scores"]
            for k, v in r["scores"].items():
                collected.setdefault(k, []).append(v)
        accums.append(to_accum(collected))

    merged, trace = tree_reduce(accums)
    final = finalize(merged)

    fvd = reduce_fvd(shards)
    if fvd is not None:
        final["fvd"] = fvd

    # Timing roll-up across shards. Wall clock is set by the SLOWEST shard
    # (that is what a barrier means), so the straggler is what we report.
    shard_walls = [s.get("compute_s", 0.0) for s in shards]
    cpu_totals = [s.get("timing", {}).get("cpu_total_s") or 0.0 for s in shards]
    gpu_busy = [s.get("timing", {}).get("gpu_busy_total_s") or 0.0 for s in shards]
    n_videos = sum(s.get("timing", {}).get("n", 0) for s in shards)
    steady = [
        s.get("timing_steady_state", {}).get("wall_mean_s")
        for s in shards
        if s.get("timing_steady_state", {}).get("wall_mean_s")
    ]

    summary = {
        "launcher": args.launcher,
        "world_size": args.world_size or len(shards),
        "threads_per_worker": args.threads,
        "n_shards": len(shards),
        "n_videos": n_videos,
        "wall_s": round(args.wall, 3) if args.wall else None,
        "peak_loadavg": args.peak_load,
        "shard_compute_s": [round(w, 3) for w in shard_walls],
        "straggler_s": round(max(shard_walls), 3) if shard_walls else None,
        "shard_imbalance": (
            round(max(shard_walls) / min(shard_walls), 3)
            if shard_walls and min(shard_walls) > 0
            else None
        ),
        "model_load_s": [s.get("model_load_s") for s in shards],
        "cpu_total_s": round(sum(cpu_totals), 3),
        "gpu_busy_total_s": round(sum(gpu_busy), 3),
        # The two numbers the whole GPU-count argument rests on.
        "cpu_seconds_per_video": (
            round(sum(cpu_totals) / n_videos, 3) if n_videos else None
        ),
        "gpu_busy_seconds_per_video": (
            round(sum(gpu_busy) / n_videos, 3) if n_videos else None
        ),
        "throughput_videos_per_s": (
            round(n_videos / args.wall, 4) if args.wall else None
        ),
        "steady_state_wall_mean_s": (
            round(sum(steady) / len(steady), 3) if steady else None
        ),
        "reduce_tree": trace,
        "final_scores": final,
        "per_video_scores": per_video,
        "thread_checks": [s.get("thread_check") for s in shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        print(f" merged       : {len(files)} shards, {n_videos} videos")
        for line in trace:
            print(f"   tree-reduce {line}")
        print(f" scores       : " + "  ".join(f"{k}={v:.4f}" for k, v in final.items()))
        if summary["shard_imbalance"]:
            print(
                f" imbalance    : {summary['shard_imbalance']}x "
                f"(slowest shard {summary['straggler_s']}s / fastest "
                f"{min(shard_walls):.1f}s)"
            )
        print(
            f" per video    : cpu={summary['cpu_seconds_per_video']}s  "
            f"gpu_busy={summary['gpu_busy_seconds_per_video']}s"
        )
        print(f" summary      : {args.out}")

    bad = [t for t in summary["thread_checks"] if t and t.get("mismatch")]
    if bad:
        print(f" !! {len(bad)} shard(s) had a THREAD MISMATCH -- timings suspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
