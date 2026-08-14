#!/usr/bin/env python
"""Ray launcher: an actor pool with a dynamic work queue.

Same work, same `score_one_video()`, different orchestration. What changes:

  bash-fork                          Ray actor pool
  ----------------------------------------------------------------------
  one OS process per GPU, forked     one actor per GPU, scheduled
  static stride video[r::W]          dynamic queue, next-free-actor wins
  shard count from a bash variable   shard count == len(pool), cannot desync
  dead worker -> silent missing      dead actor -> RayActorError
  merge by reading N JSON files      merge in-process (still tree-reduced)

Three thread arms, selectable with --threads-mode, because the interesting
result is the *difference between them*:

  auto    trust Ray's defaults. Ray reads the cgroup quota correctly (198, not
          256) but then gives every actor OMP_NUM_THREADS=1 regardless. For a
          CPU-bound workload this is a cliff.
  tuned   num_cpus=quota/G AND an explicit torch.set_num_threads(quota/G) inside
          the actor. The explicit call is not redundant: torch reads OMP once at
          import, and the actor's import may already have happened, so raising
          num_cpus alone is import-order dependent.
  fixed   an explicit --threads value, for the sweep.

Run:
  python run_ray.py --dataset data/eval81 --gpus 4 --threads-mode tuned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.resources import cpu_quota, threads_per_worker  # noqa: E402


def build_actor_cls(num_cpus: float | None):
    """Create the actor class. num_cpus is a *scheduling* token, not affinity."""
    import ray

    @ray.remote(num_gpus=1, num_cpus=num_cpus)
    class EvalActor:
        """One GPU's worth of scoring. Models load once, in __init__."""

        def __init__(self, threads: int, cfg_kwargs: dict, actor_id: int):
            self.actor_id = actor_id
            self.threads = threads

            # Must happen before torch's intra-op pool is first used. Ray has
            # already set OMP_NUM_THREADS=num_cpus (default 1) in this worker's
            # env, so we override BOTH the env and the live torch setting.
            if threads > 0:
                for var in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ):
                    os.environ[var] = str(threads)

            import torch

            from common.timing import SampleTimer, check_thread_mismatch
            from movebench.metrics import ScoreConfig, load_models

            if threads > 0:
                torch.set_num_threads(threads)

            self.cfg = ScoreConfig(**cfg_kwargs)
            t0 = time.perf_counter()
            self.models = load_models(self.cfg)
            self.load_s = time.perf_counter() - t0
            self.timer = SampleTimer(use_cuda=torch.cuda.is_available())
            self.check = check_thread_mismatch(threads) if threads > 0 else {}

        def info(self) -> dict:
            from common.timing import thread_env_report

            return {
                "actor_id": self.actor_id,
                "model_load_s": round(self.load_s, 3),
                "env": thread_env_report(),
                "thread_check": self.check,
            }

        def score(self, name: str, raw: str, gen: str) -> dict:
            from movebench.metrics import score_one_video

            with self.timer.sample(name):
                out = score_one_video(name, Path(raw), Path(gen), self.models, self.cfg)
            s = self.timer.samples[-1]
            out["_timing"] = s.as_dict()
            out["_actor"] = self.actor_id
            return out

        def drain(self) -> dict:
            self.timer.close()
            return {
                "actor_id": self.actor_id,
                "model_load_s": round(self.load_s, 3),
                "timing": self.timer.summary(),
                "timing_steady_state": self.timer.steady_state_summary(skip=1),
                "per_sample": [s.as_dict() for s in self.timer.samples],
                "thread_check": self.check,
            }

    return EvalActor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--gpus", type=int, default=None, help="pool size (default: all)")
    ap.add_argument(
        "--threads-mode", default="tuned", choices=["auto", "tuned", "fixed"]
    )
    ap.add_argument("--threads", type=int, default=None, help="for --threads-mode fixed")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metrics", default="clip,epe,lpips,ssim,psnr")
    ap.add_argument("--device-policy", default="original", choices=["original", "gpu"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pinned-memory", action="store_true")
    ap.add_argument("--tag", default="ray")
    args = ap.parse_args()

    import ray

    from movebench.metrics import list_pairs

    pairs = list_pairs(args.dataset)
    if args.limit:
        pairs = pairs[: args.limit]

    ray.init(log_to_driver=False, include_dashboard=False)
    avail = ray.available_resources()
    n_gpus = args.gpus or int(avail.get("GPU", 0))
    if n_gpus < 1:
        raise SystemExit("no GPUs available to Ray")

    quota = cpu_quota()
    # Ray's own CPU detection -- the point worth citing in its favour.
    ray_cpu = int(avail.get("CPU", 0))

    if args.threads_mode == "auto":
        # Trust Ray completely: no num_cpus request, no set_num_threads.
        num_cpus, threads = None, 0
    elif args.threads_mode == "fixed":
        threads = args.threads if args.threads is not None else threads_per_worker(n_gpus)
        num_cpus = float(threads)
    else:  # tuned
        threads = threads_per_worker(n_gpus, quota)
        num_cpus = float(threads)

    out_dir = args.out or REPO / f"results/movebench_{args.tag}_g{n_gpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==============================================================")
    print(" launcher     : Ray actor pool (dynamic work queue)")
    print(f" dataset      : {args.dataset}  ({len(pairs)} videos)")
    print(f" pool size    : {n_gpus} actors (1 GPU each)")
    print(f" CPU quota    : {quota}   | Ray detected: {ray_cpu}   | nproc: {os.cpu_count()}")
    print(f" threads mode : {args.threads_mode}", end="")
    if args.threads_mode == "auto":
        print("   (Ray default -> expect OMP_NUM_THREADS=1 per actor)")
    else:
        print(f"   num_cpus={num_cpus}  torch.set_num_threads({threads})")
    print(f" out          : {out_dir}")
    print("==============================================================")

    cfg_kwargs = dict(
        metrics=tuple(m.strip() for m in args.metrics.split(",") if m.strip()),
        device_policy=args.device_policy,
        max_frames=args.max_frames,
        pinned_memory=args.pinned_memory,
    )

    ActorCls = build_actor_cls(num_cpus)

    wall0 = time.perf_counter()
    actors = [
        ActorCls.remote(threads, cfg_kwargs, i) for i in range(n_gpus)
    ]
    infos = ray.get([a.info.remote() for a in actors])
    cold_start = time.perf_counter() - wall0
    print(f" actor pool ready in {cold_start:.1f}s")
    for inf in infos:
        env = inf["env"]
        flag = " !! MISMATCH" if inf["thread_check"].get("mismatch") else ""
        print(
            f"   actor {inf['actor_id']}: load={inf['model_load_s']}s "
            f"OMP={env.get('OMP_NUM_THREADS')} torch_threads={env.get('torch_num_threads')}"
            f"{flag}"
        )

    # ---------------------------------------------------------------------
    # dynamic work queue: hand the next video to whichever actor finished.
    # This is the piece the static stride cannot do. With equal-cost videos it
    # is worth nothing; with duration variance it is the whole ballgame.
    # ---------------------------------------------------------------------
    queue = list(pairs)
    inflight: dict = {}
    results = []
    idle = list(actors)

    def dispatch():
        while queue and idle:
            actor = idle.pop()
            name, raw, gen = queue.pop(0)
            ref = actor.score.remote(name, str(raw), str(gen))
            inflight[ref] = actor

    compute0 = time.perf_counter()
    dispatch()
    done_n = 0
    while inflight:
        ready, _ = ray.wait(list(inflight.keys()), num_returns=1)
        for ref in ready:
            actor = inflight.pop(ref)
            try:
                res = ray.get(ref)
            except ray.exceptions.RayActorError as e:
                # The fault-tolerance difference, made concrete: a dead actor is
                # a raised exception here, not a silently absent shard.
                print(f" !! actor died mid-run: {e}", file=sys.stderr)
                continue
            results.append(res)
            done_n += 1
            t = res["_timing"]
            print(
                f" [{done_n}/{len(pairs)}] {res['name']} actor={res['_actor']} "
                f"wall={t['wall_s']:.2f}s cpu={t['cpu_s']:.1f}s "
                f"gpu_util={t['gpu_util_mean']}% cores_busy={t['cpu_par']:.1f}",
                flush=True,
            )
            idle.append(actor)
        dispatch()
    compute_s = time.perf_counter() - compute0
    wall_s = time.perf_counter() - wall0

    drains = ray.get([a.drain.remote() for a in actors])

    # Write one shard file per actor so merge.py treats both launchers identically
    # -- the merged summary is then directly comparable, same code path.
    for d in drains:
        aid = d["actor_id"]
        payload = {
            "rank": aid,
            "world_size": n_gpus,
            "threads_requested": threads,
            "model_load_s": d["model_load_s"],
            "compute_s": d["timing"].get("wall_total_s", 0.0),
            "thread_check": d["thread_check"],
            "timing": d["timing"],
            "timing_steady_state": d["timing_steady_state"],
            "per_sample": d["per_sample"],
            "results": [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in results
                if r["_actor"] == aid
            ],
        }
        (out_dir / f"shard_{aid}.json").write_text(json.dumps(payload, indent=2))

    # Videos-per-actor: the load-balance evidence.
    per_actor = {}
    for r in results:
        per_actor[r["_actor"]] = per_actor.get(r["_actor"], 0) + 1

    (out_dir / "ray_run.json").write_text(
        json.dumps(
            {
                "launcher": "ray",
                "threads_mode": args.threads_mode,
                "pool_size": n_gpus,
                "threads_per_actor": threads,
                "num_cpus_per_actor": num_cpus,
                "ray_detected_cpu": ray_cpu,
                "cgroup_quota": quota,
                "host_cpu_count": os.cpu_count(),
                "cold_start_s": round(cold_start, 3),
                "compute_s": round(compute_s, 3),
                "wall_s": round(wall_s, 3),
                "videos_per_actor": per_actor,
                "actor_info": infos,
            },
            indent=2,
        )
    )

    print("--------------------------------------------------------------")
    print(f" cold start   : {cold_start:.2f}s (actor pool + model load)")
    print(f" compute      : {compute_s:.2f}s")
    print(f" wall clock   : {wall_s:.2f}s")
    print(f" per actor    : {dict(sorted(per_actor.items()))} videos")

    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "merge.py"),
            "--shards", str(out_dir),
            "--out", str(out_dir / "summary.json"),
            "--launcher", f"ray-{args.threads_mode}",
            "--world-size", str(n_gpus),
            "--threads", str(threads),
            "--wall", str(wall_s),
        ],
        check=True,
    )

    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
