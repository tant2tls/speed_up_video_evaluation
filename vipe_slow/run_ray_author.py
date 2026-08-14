#!/usr/bin/env python
"""The ViPE author's own Ray design, implemented faithfully — as a comparison arm.

Provenance
----------
The ViPE maintainer (`heiwang1997`, NVIDIA Toronto AI Lab) answered
nv-tlabs/vipe issue #41 ("Support for batch inference", Sep 2025) by confirming
that Ray is how ViPE is scaled internally, and posting the code they use:

    https://github.com/nv-tlabs/vipe/issues/41#issuecomment-3331003448

    "internally we've already scaled up ViPE mainly using `ray`. It allows
     pretty robust elastic computing using multiple GPUs."

This file reproduces that design so it can be *measured*, not just cited. Their
structure, kept faithfully:

  * `@ray.remote(num_gpus=1, num_cpus=4)` on a **task**, not an actor
  * **one task submitted per video** (`for stream_idx in range(len(stream_list))`)
  * the `StreamList` shipped to workers via `ray.put()` and indexed by position
  * `os.chdir(cwd)` inside the task so ViPE's relative path resolution works
  * `make_pipeline()` called **inside** the task
  * a `ray.wait()` drain loop with per-job `try/except RayTaskError`

What this repo adds is only instrumentation (per-video CPU/GPU attribution) and
the `--num-cpus` knob, so the arm can be compared against `run_ray.py`'s actor
pool under identical conditions. **No structural change** — that is the point.

The two designs and their predicted difference
----------------------------------------------
                        author's tasks              this repo's actor pool
  unit of scheduling    one task per video          one long-lived actor per GPU
  model weights         loaded per task             loaded once per actor
  elasticity            excellent (tasks are        fixed pool size
                        independent, requeueable)
  fault isolation       one task dies, others fine  actor death kills its queue
  cold start            paid N times                paid G times

The tradeoff is real and goes both ways: tasks buy elasticity and fault
granularity at the cost of re-paying model load per video. Whether that cost is
visible depends on `make_pipeline()`'s cost relative to per-video work — which is
exactly what we measure. ViPE caches model weights *process-globally*, so a Ray
worker process reused across tasks may hit a warm cache and pay far less than the
first call; the measurement tells us how often that reuse actually happens.

Usage
-----
    python run_ray_author.py --videos test_video --gpus 8 --pose-only
    python run_ray_author.py --videos test_video --num-cpus 24   # thread sweep
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


def build_task(num_gpus: float, num_cpus: float | None, threads: int):
    """The author's `run_video_stream` task, plus timing instrumentation."""
    import ray

    @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)
    def run_video_stream(cwd, stream_list, stream_idx, pipeline_args):
        # Author's line: ViPE resolves config/asset paths relative to cwd, and a
        # Ray worker starts in its own temp dir, so this chdir is load-bearing.
        os.chdir(str(cwd))

        if threads > 0:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[var] = str(threads)

        import torch

        from common.timing import SampleTimer, thread_env_report

        if threads > 0:
            torch.set_num_threads(threads)

        from vipe.pipeline import make_pipeline
        from vipe.utils.logging import configure_logging

        configure_logging()
        video_stream = stream_list[stream_idx]
        name = video_stream.name()

        # Fail loudly and legibly on the empty-stream trap described in main():
        # a 0-frame stream means the path did not resolve in THIS process, and
        # the native assert that would otherwise fire names neither the path nor
        # the cause.
        if len(video_stream) == 0:
            raise RuntimeError(
                f"stream {name!r} has 0 frames inside the worker (cwd={os.getcwd()}). "
                "The video path did not resolve in this process -- pass an absolute "
                "--videos path. cv2 reports 0 frames rather than raising, so this "
                "would otherwise surface as 'assert index < len(self)'."
            )

        # Instrumentation only: separate the per-task model build from the run,
        # because "model load per task" is the design's suspected cost.
        t0 = time.perf_counter()
        pipeline = make_pipeline(pipeline_args)
        build_s = time.perf_counter() - t0

        timer = SampleTimer(use_cuda=torch.cuda.is_available())
        with timer.sample(name):
            pipeline.run(video_stream)
        timer.close()
        s = timer.samples[-1]

        return {
            "name": name,
            "pipeline_build_s": round(build_s, 3),
            "worker_pid": os.getpid(),
            "_timing": s.as_dict(),
            "env": thread_env_report(),
        }

    return run_video_stream


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--gpus", type=int, default=None, help="cap concurrency (informational)")
    ap.add_argument(
        "--num-cpus",
        type=float,
        default=4.0,
        help="num_cpus per task. The author's code uses 4; pass quota/G to match "
        "this repo's tuned arm, or 0 to let Ray default.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="explicit torch.set_num_threads inside the task. Default: follow "
        "--num-cpus (what Ray's OMP_NUM_THREADS would be). 0 = leave torch alone.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pose-only", action="store_true")
    ap.add_argument("--pipeline", default="default")
    ap.add_argument("--tag", default="ray_author")
    args = ap.parse_args()

    import ray
    from hydra import compose, initialize_config_dir
    from ray.exceptions import RayTaskError

    # --------------------------------------------------------------------
    # Resolve --videos to an absolute path before it reaches the config.
    #
    # This is not cosmetic. The author's task does `os.chdir(cwd)` inside the
    # worker, but the `StreamList` is built on the DRIVER and shipped via
    # `ray.put()`. `RawMp4Stream.__init__` probes frame count with cv2 at
    # construction time, and **cv2 returns 0 frames for an unreachable path
    # instead of raising**. So a relative --videos that is valid on the driver
    # but not from the worker's cwd yields `len(stream) == 0`, and the failure
    # surfaces much later as a bare `assert index < len(self)` in
    # streams/base.py:360 -- with no mention of a path anywhere.
    #
    # Reproduced here before fixing it; see report.md §6 "A bug the task design hides".
    # --------------------------------------------------------------------
    args.videos = args.videos.resolve()
    if not args.videos.exists():
        raise SystemExit(f"no such path: {args.videos}")

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
    quota = cpu_quota()
    ray_cpu = int(avail.get("CPU", 0))

    num_cpus = args.num_cpus if args.num_cpus and args.num_cpus > 0 else None
    threads = args.threads
    if threads is None:
        threads = int(num_cpus) if num_cpus else 0

    out_dir = args.out or REPO / f"results/vipe_{args.tag}_g{n_gpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    print("==============================================================")
    print(" launcher     : Ray TASKS, one per video  (ViPE author's design,")
    print("                nv-tlabs/vipe issue #41)")
    print(f" videos       : {args.videos}  ({len(videos)} videos)")
    print(f" concurrency  : {n_gpus} GPUs -> {n_gpus} tasks in flight")
    print(f" tasks        : {len(videos)} (one per video)")
    print(f" CPU quota    : {quota}   | Ray detected: {ray_cpu}   | nproc: {os.cpu_count()}")
    print(f" num_cpus/task: {num_cpus}   torch threads: {threads or 'default'}")
    print(f" out          : {out_dir}")
    print("==============================================================")

    overrides = [
        f"pipeline={args.pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={args.videos}",
        f"pipeline.output.path={artifacts}",
    ]
    if args.pose_only:
        overrides.append("pipeline.post.depth_align_model=null")

    wall0 = time.perf_counter()
    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

        from vipe.streams.base import StreamList

        stream_list = StreamList.make(cfg.streams)
        # Restrict to the requested videos while keeping the author's
        # index-into-a-put-StreamList access pattern.
        wanted = {v.stem for v in videos}
        indices = [i for i in range(len(stream_list)) if stream_list[i].name() in wanted]

        task = build_task(1, num_cpus, threads)

        # The author's three ray.put() handles: ship each big object ONCE and
        # pass references, rather than re-serializing per task.
        stream_list_ref = ray.put(stream_list)
        pipeline_args_ref = ray.put(cfg.pipeline)
        cwd_ref = ray.put(HERE.resolve())

        submit0 = time.perf_counter()
        futures = [
            task.remote(cwd_ref, stream_list_ref, idx, pipeline_args_ref)
            for idx in indices
        ]
        submit_s = time.perf_counter() - submit0
        print(f" submitted {len(futures)} tasks in {submit_s:.2f}s", flush=True)

        results, failures = [], []
        done_n = 0
        compute0 = time.perf_counter()
        while len(futures):
            done_id, futures = ray.wait(futures)
            for obj_ref in done_id:
                try:
                    res = ray.get(obj_ref)
                except (RayTaskError, Exception) as e:  # author's handler
                    print(f"RayExecutor: Exception in job {e}", file=sys.stderr)
                    failures.append(str(e))
                    continue
                results.append(res)
                done_n += 1
                t = res["_timing"]
                print(
                    f" [{done_n}/{len(indices)}] {res['name']} "
                    f"pid={res['worker_pid']} build={res['pipeline_build_s']}s "
                    f"wall={t['wall_s']:.2f}s cpu={t['cpu_s']:.1f}s "
                    f"gpu_util={t['gpu_util_mean']}% cores_busy={t['cpu_par']}",
                    flush=True,
                )
        compute_s = time.perf_counter() - compute0
    wall_s = time.perf_counter() - wall0

    # How many distinct worker processes ran, and how often model load was re-paid.
    pids = {}
    for r in results:
        pids.setdefault(r["worker_pid"], []).append(r["pipeline_build_s"])
    builds = [r["pipeline_build_s"] for r in results]
    cold_builds = [b for b in builds if b > 1.0]

    # Emit shards in this repo's shape so summarize.py works unchanged.
    n_shards = max(1, len(pids))
    for i, (pid, _) in enumerate(sorted(pids.items())):
        mine = [r for r in results if r["worker_pid"] == pid]
        samples = [r["_timing"] for r in mine]
        wall_tot = sum(s["wall_s"] for s in samples)
        cpu_tot = sum(s["cpu_s"] for s in samples)
        busy = [s["gpu_busy_s"] for s in samples if s.get("gpu_busy_s") is not None]
        (out_dir / f"shard_{i}.json").write_text(
            json.dumps(
                {
                    "rank": i,
                    "worker_pid": pid,
                    "world_size": n_shards,
                    "threads_requested": threads,
                    "model_load_s": round(sum(r["pipeline_build_s"] for r in mine), 3),
                    "compute_s": round(wall_tot, 3),
                    "timing": {
                        "n": len(mine),
                        "wall_total_s": round(wall_tot, 3),
                        "wall_mean_s": round(wall_tot / len(mine), 3),
                        "cpu_total_s": round(cpu_tot, 3),
                        "cpu_mean_s": round(cpu_tot / len(mine), 3),
                        "gpu_busy_total_s": round(sum(busy), 3) if busy else None,
                        "gpu_busy_mean_s": round(sum(busy) / len(busy), 3) if busy else None,
                        "cpu_parallelism": round(cpu_tot / wall_tot, 3) if wall_tot else None,
                        "gpu_occupancy": round(sum(busy) / wall_tot, 3) if busy and wall_tot else None,
                    },
                    "timing_steady_state": {},
                    "per_sample": samples,
                    "thread_check": {},
                    "videos": [r["name"] for r in mine],
                    "results": [{"name": r["name"], "scores": {}} for r in mine],
                },
                indent=2,
            )
        )

    (out_dir / "ray_author_run.json").write_text(
        json.dumps(
            {
                "launcher": "ray-tasks-author",
                "provenance": "https://github.com/nv-tlabs/vipe/issues/41#issuecomment-3331003448",
                "n_videos": len(indices),
                "n_tasks": len(indices),
                "gpus": n_gpus,
                "num_cpus_per_task": num_cpus,
                "threads_per_task": threads,
                "ray_detected_cpu": ray_cpu,
                "cgroup_quota": quota,
                "submit_s": round(submit_s, 3),
                "compute_s": round(compute_s, 3),
                "wall_s": round(wall_s, 3),
                "n_worker_processes": len(pids),
                "tasks_per_worker_process": {str(k): len(v) for k, v in pids.items()},
                "pipeline_build_s_total": round(sum(builds), 3),
                "pipeline_build_s_mean": round(sum(builds) / len(builds), 3) if builds else None,
                "n_cold_builds_over_1s": len(cold_builds),
                "failures": failures,
            },
            indent=2,
        )
    )

    print("--------------------------------------------------------------")
    print(f" wall clock        : {wall_s:.2f}s   (compute {compute_s:.2f}s)")
    print(f" worker processes  : {len(pids)} for {len(indices)} tasks")
    print(
        f" make_pipeline()   : {sum(builds):.1f}s total, "
        f"{(sum(builds) / len(builds) if builds else 0):.2f}s mean, "
        f"{len(cold_builds)} cold (>1s)"
    )
    if failures:
        print(f" failures          : {len(failures)}")

    subprocess.run(
        [
            sys.executable, str(HERE / "summarize.py"),
            "--shards", str(out_dir),
            "--out", str(out_dir / "vipe_summary.json"),
            "--launcher", "ray-tasks-author",
            "--world-size", str(n_shards),
            "--threads", str(threads),
            "--wall", str(wall_s),
        ],
        check=True,
    )
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
