"""ViPE driver: annotate a folder of videos with camera pose + depth.

This is the original `run.py` from this checkout with two changes, both of which
matter for the benchmark and one of which is a real bug:

  1. **`make_pipeline()` moved OUT of the per-video loop.** The original built a
     fresh pipeline inside the loop, once per video. `make_pipeline()` costs
     ~9.7 s on its first call in a process and ~0 s afterwards (the model weights
     are cached process-globally), so this does not cost 9.7 s per video -- but it
     *does* fold the model-load time into the first video's measured time, which
     silently inflates any per-video average computed over a short run. Moving it
     out makes cold-start and steady-state separable, which is the whole point of
     the sweep. (See report.md §6 and vipe_slow/README.md's header note.)

  2. **Per-video timing with CPU/GPU attribution**, written to a JSON shard so the
     same `merge.py`-style roll-up works for both launchers.

Sharding is `--rank`/`--world-size` static stride, matching the production
pattern the Ray version is compared against.

Usage (single process, all videos):
    python run.py --videos test_video --out results/vipe_g1/shard_0.json

Hydra-style overrides still work via --override:
    python run.py --videos test_video --override pipeline.post.depth_align_model=null
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _pin_threads(n: int) -> None:
    """Must run before torch is imported: torch fixes its intra-op pool at import."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, type=Path, help="dir of .mp4 (or one .mp4)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="0 = leave torch default")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--output-path", type=Path, default=None, help="artifact dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pose-only", action="store_true", help="skip depth alignment")
    ap.add_argument("--pipeline", default="default")
    ap.add_argument(
        "--stream-mode",
        default="lazy",
        choices=["lazy", "cached"],
        help="lazy: StreamList streams frames as the pipeline consumes them. "
        "cached: wrap each video in ProcessedVideoStream(...).cache() first, "
        "decoding EVERY frame into host RAM before inference -- what the "
        "production launcher (eval_cam/vipe/run_multinode.py) actually does. "
        "~30 MB/frame at 1280x1280, so the cost multiplies by worker count.",
    )
    ap.add_argument("--override", action="append", default=[], help="hydra overrides")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from common.resources import cpu_quota, threads_per_worker

    if args.threads is None:
        args.threads = threads_per_worker(args.world_size)
    if args.threads > 0:
        _pin_threads(args.threads)

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from common.timing import SampleTimer, check_thread_mismatch, thread_env_report

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # ---- discover and shard the work ------------------------------------
    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 found under {args.videos}")
    if args.limit:
        videos = videos[: args.limit]
    shard = videos[args.rank :: args.world_size]

    out_artifacts = args.output_path or (REPO / "results" / "vipe_artifacts")
    out_artifacts.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"pipeline={args.pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={args.videos}",
        f"pipeline.output.path={out_artifacts}",
    ]
    if args.pose_only:
        overrides.append("pipeline.post.depth_align_model=null")
    overrides.extend(args.override)

    mismatch = check_thread_mismatch(args.threads) if args.threads > 0 else {}
    if mismatch.get("mismatch"):
        print(
            f"[rank {args.rank}] !! THREAD MISMATCH: requested {mismatch['requested']}, "
            f"torch reports {mismatch['actual_torch_threads']}",
            flush=True,
        )

    print(
        f"[rank {args.rank}/{args.world_size}] {len(shard)}/{len(videos)} videos | "
        f"threads={args.threads} (quota={cpu_quota()}) | torch={torch.get_num_threads()} | "
        f"cuda_devices={torch.cuda.device_count()}",
        flush=True,
    )

    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

        from vipe.pipeline import make_pipeline
        from vipe.streams.base import StreamList
        from vipe.utils.logging import configure_logging

        configure_logging()

        # THE FIX: build the pipeline ONCE, outside the loop. Model weights load
        # here, so cold-start is attributable and per-video times are steady-state.
        t0 = time.perf_counter()
        pipeline = make_pipeline(cfg.pipeline)
        build_s = time.perf_counter() - t0
        print(f"[rank {args.rank}] pipeline built in {build_s:.1f}s", flush=True)

        stream_list = StreamList.make(cfg.streams)
        want = {p.stem for p in shard}
        # StreamList enumerates the whole folder; select just this rank's shard.
        indices = [
            i for i in range(len(stream_list)) if stream_list[i].name() in want
        ]

        timer = SampleTimer(use_cuda=torch.cuda.is_available())
        done = []
        cache_times: list[float] = []
        t_compute0 = time.perf_counter()
        for k, idx in enumerate(indices):
            stream = stream_list[idx]
            name = stream.name()
            with timer.sample(name):
                if args.stream_mode == "cached":
                    # Faithful to the PRODUCTION launcher (eval_cam/vipe/
                    # run_multinode.py): every video is wrapped in
                    # ProcessedVideoStream(...).cache(), which decodes and holds
                    # EVERY FRAME in host RAM before inference starts. That is
                    # ~30 MB/frame at 1280x1280, so a per-worker host cost that
                    # multiplies by G -- and it is CPU+RAM work that the lazy
                    # path never pays. This is the arm that reproduces the
                    # "4 GPUs beat 8" observation.
                    from vipe.streams.base import ProcessedVideoStream

                    t_c = time.perf_counter()
                    stream = ProcessedVideoStream(stream, []).cache(
                        desc="Reading video stream"
                    )
                    cache_times.append(time.perf_counter() - t_c)
                pipeline.run(stream)
            s = timer.samples[-1]
            done.append(name)
            print(
                f"[rank {args.rank}] {k + 1}/{len(indices)} {name} "
                f"wall={s.wall_s:.2f}s cpu={s.cpu_s:.1f}s "
                + (f"cache={cache_times[-1]:.2f}s " if cache_times else "")
                + f"gpu_util={s.gpu_util_mean}% cores_busy={s.cpu_par:.1f}",
                flush=True,
            )
        compute_s = time.perf_counter() - t_compute0
        timer.close()

    payload = {
        "rank": args.rank,
        "world_size": args.world_size,
        "threads_requested": args.threads,
        "stream_mode": args.stream_mode,
        "model_load_s": round(build_s, 3),
        "compute_s": round(compute_s, 3),
        "cache_total_s": round(sum(cache_times), 3) if cache_times else None,
        "cache_mean_s": (
            round(sum(cache_times) / len(cache_times), 3) if cache_times else None
        ),
        "videos": done,
        "env": thread_env_report(),
        "thread_check": mismatch,
        "timing": timer.summary(),
        "timing_steady_state": timer.steady_state_summary(skip=1),
        "per_sample": [s.as_dict() for s in timer.samples],
        # `results` keeps the shard shape identical to movebench so merge.py works.
        "results": [{"name": n, "scores": {}} for n in done],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"[rank {args.rank}] DONE {len(done)} videos in {compute_s:.1f}s -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
