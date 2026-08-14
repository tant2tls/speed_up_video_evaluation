#!/usr/bin/env python
"""ViPE Ray launcher: actor pool + dynamic work queue.

The counterpart to run_fork.sh. Same ViPE pipeline, same per-video work; the
difference is that videos are pulled from a queue by whichever actor is free
instead of being assigned up front by `videos[rank::world_size]`.

Why this matters more for ViPE than for the eval metrics: ViPE's per-video cost
scales with frame count and with how hard the scene is to track (bundle
adjustment iterates to convergence), so real video sets have genuine duration
variance -- which is precisely the condition under which a static stride leaves
GPUs idle and a dynamic queue does not.

Thread arms are the same three as movebench/run_ray.py:
  auto   Ray's defaults (correctly detects the 198-CPU quota, then gives each
         actor OMP_NUM_THREADS=1 anyway)
  tuned  num_cpus = quota/G AND an explicit torch.set_num_threads(quota/G)
  fixed  an explicit --threads value

Usage:
  python run_ray.py --videos test_video --gpus 4 --threads-mode tuned
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from common.resources import cpu_quota, threads_per_worker  # noqa: E402


def build_actor_cls(num_cpus: float | None):
    import ray

    @ray.remote(num_gpus=1, num_cpus=num_cpus)
    class VipeActor:
        """One GPU's ViPE pipeline. Built once in __init__, reused per video."""

        def __init__(self, threads: int, cfg_overrides: list[str], actor_id: int,
                     videos_dir: str, artifacts: str):
            self.actor_id = actor_id
            if threads > 0:
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    os.environ[var] = str(threads)

            import torch
            from hydra import compose, initialize_config_dir

            from common.timing import SampleTimer, check_thread_mismatch

            if threads > 0:
                torch.set_num_threads(threads)

            # Hydra's global state is per-process; each actor is its own process.
            self._hydra_ctx = initialize_config_dir(
                config_dir=str(HERE / "configs"), version_base=None
            )
            self._hydra_ctx.__enter__()
            cfg = compose(config_name="default", overrides=cfg_overrides)

            from vipe.pipeline import make_pipeline
            from vipe.streams.base import StreamList
            from vipe.utils.logging import configure_logging

            configure_logging()
            t0 = time.perf_counter()
            self.pipeline = make_pipeline(cfg.pipeline)
            self.load_s = time.perf_counter() - t0

            # Build the stream list once; index by video name at dispatch time.
            self.stream_list = StreamList.make(cfg.streams)
            self.by_name = {
                self.stream_list[i].name(): i for i in range(len(self.stream_list))
            }
            self.timer = SampleTimer(use_cuda=torch.cuda.is_available())
            self.check = check_thread_mismatch(threads) if threads > 0 else {}

        def info(self) -> dict:
            from common.timing import thread_env_report

            return {
                "actor_id": self.actor_id,
                "model_load_s": round(self.load_s, 3),
                "n_streams_visible": len(self.by_name),
                "env": thread_env_report(),
                "thread_check": self.check,
            }

        def run_video(self, name: str) -> dict:
            idx = self.by_name.get(name)
            if idx is None:
                return {"name": name, "error": "not in stream list"}
            with self.timer.sample(name):
                self.pipeline.run(self.stream_list[idx])
            s = self.timer.samples[-1]
            return {"name": name, "_timing": s.as_dict(), "_actor": self.actor_id}

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

    return VipeActor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--threads-mode", default="tuned", choices=["auto", "tuned", "fixed"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pose-only", action="store_true")
    ap.add_argument("--pipeline", default="default")
    ap.add_argument("--tag", default="ray")
    args = ap.parse_args()

    import ray

    videos = (
        [args.videos] if args.videos.is_file() else sorted(args.videos.glob("*.mp4"))
    )
    if not videos:
        raise SystemExit(f"no .mp4 under {args.videos}")
    if args.limit:
        videos = videos[: args.limit]

    ray.init(log_to_driver=False, include_dashboard=False)
    avail = ray.available_resources()
    n_gpus = args.gpus or int(avail.get("GPU", 0))
    if n_gpus < 1:
        raise SystemExit("no GPUs available to Ray")
    quota = cpu_quota()
    ray_cpu = int(avail.get("CPU", 0))

    if args.threads_mode == "auto":
        num_cpus, threads = None, 0
    elif args.threads_mode == "fixed":
        threads = args.threads if args.threads is not None else threads_per_worker(n_gpus)
        num_cpus = float(threads)
    else:
        threads = threads_per_worker(n_gpus, quota)
        num_cpus = float(threads)

    out_dir = args.out or REPO / f"results/vipe_{args.tag}_g{n_gpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"pipeline={args.pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={args.videos}",
        f"pipeline.output.path={artifacts}",
    ]
    if args.pose_only:
        overrides.append("pipeline.post.depth_align_model=null")

    print("==============================================================")
    print(" launcher     : Ray actor pool (dynamic work queue)")
    print(f" videos       : {args.videos}  ({len(videos)} videos)")
    print(f" pool size    : {n_gpus} actors (1 GPU each)")
    print(f" CPU quota    : {quota}   | Ray detected: {ray_cpu}   | nproc: {os.cpu_count()}")
    print(f" threads mode : {args.threads_mode}", end="")
    print("   (Ray default -> OMP=1/actor)" if args.threads_mode == "auto"
          else f"   num_cpus={num_cpus} torch.set_num_threads({threads})")
    print(f" out          : {out_dir}")
    print("==============================================================")

    ActorCls = build_actor_cls(num_cpus)
    wall0 = time.perf_counter()
    actors = [
        ActorCls.remote(threads, overrides, i, str(args.videos), str(artifacts))
        for i in range(n_gpus)
    ]
    infos = ray.get([a.info.remote() for a in actors])
    cold_start = time.perf_counter() - wall0
    print(f" actor pool ready in {cold_start:.1f}s")
    for inf in infos:
        env = inf["env"]
        flag = " !! MISMATCH" if inf["thread_check"].get("mismatch") else ""
        print(
            f"   actor {inf['actor_id']}: load={inf['model_load_s']}s "
            f"OMP={env.get('OMP_NUM_THREADS')} torch={env.get('torch_num_threads')}{flag}"
        )

    queue = [v.stem for v in videos]
    inflight: dict = {}
    idle = list(actors)
    results = []

    def dispatch():
        while queue and idle:
            a = idle.pop()
            ref = a.run_video.remote(queue.pop(0))
            inflight[ref] = a

    compute0 = time.perf_counter()
    dispatch()
    done_n = 0
    while inflight:
        ready, _ = ray.wait(list(inflight.keys()), num_returns=1)
        for ref in ready:
            a = inflight.pop(ref)
            try:
                res = ray.get(ref)
            except ray.exceptions.RayActorError as e:
                print(f" !! actor died: {e}", file=sys.stderr)
                continue
            results.append(res)
            done_n += 1
            t = res.get("_timing", {})
            print(
                f" [{done_n}/{len(videos)}] {res['name']} actor={res.get('_actor')} "
                f"wall={t.get('wall_s', 0):.2f}s cpu={t.get('cpu_s', 0):.1f}s "
                f"gpu_util={t.get('gpu_util_mean')}% cores_busy={t.get('cpu_par')}",
                flush=True,
            )
            idle.append(a)
        dispatch()
    compute_s = time.perf_counter() - compute0
    wall_s = time.perf_counter() - wall0

    drains = ray.get([a.drain.remote() for a in actors])
    for d in drains:
        aid = d["actor_id"]
        (out_dir / f"shard_{aid}.json").write_text(
            json.dumps(
                {
                    "rank": aid,
                    "world_size": n_gpus,
                    "threads_requested": threads,
                    "model_load_s": d["model_load_s"],
                    "compute_s": d["timing"].get("wall_total_s", 0.0),
                    "thread_check": d["thread_check"],
                    "timing": d["timing"],
                    "timing_steady_state": d["timing_steady_state"],
                    "per_sample": d["per_sample"],
                    "videos": [r["name"] for r in results if r.get("_actor") == aid],
                    "results": [
                        {"name": r["name"], "scores": {}}
                        for r in results
                        if r.get("_actor") == aid
                    ],
                },
                indent=2,
            )
        )

    per_actor: dict = {}
    for r in results:
        per_actor[r.get("_actor")] = per_actor.get(r.get("_actor"), 0) + 1

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
    print(f" cold start   : {cold_start:.2f}s")
    print(f" compute      : {compute_s:.2f}s")
    print(f" wall clock   : {wall_s:.2f}s")
    print(f" per actor    : {dict(sorted(per_actor.items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    subprocess.run(
        [
            sys.executable, str(HERE / "summarize.py"),
            "--shards", str(out_dir),
            "--out", str(out_dir / "vipe_summary.json"),
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
