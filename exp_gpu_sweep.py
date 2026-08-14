#!/usr/bin/env python
"""The experiment: sweep GPU count with the CPU budget held fixed.

This is the harness behind the repo's central claim. It answers one question
per workload:

    Holding the container's CPU budget C fixed, what happens to throughput as
    the number of GPU workers G grows -- and where does it stop helping?

The model
---------
Each video needs W_cpu seconds of CPU work and W_gpu seconds of GPU work. With G
workers sharing a fixed budget of C CPUs:

    throughput(G)  =  min( G / W_gpu ,  C / W_cpu )

The first term is the GPU supply; it grows with G. The second is the CPU supply;
it does not grow at all -- C is a cgroup quota, not something a launcher can vote
on. So throughput rises linearly until the CPU term binds, at

    G*  =  C * W_gpu / W_cpu

and is flat after. Worse than flat, in practice: past G* each worker's thread
share (C/G) keeps shrinking, so the per-worker CPU stage gets slower even as the
count of workers rises, and the curve can bend *down*.

The prediction this makes, and why the two workloads are both run
----------------------------------------------------------------
  movebench eval -- LPIPS-VGG/SSIM/PSNR never leave the host, so W_cpu is huge
                    and W_gpu is small. Small G*: saturates almost immediately.
  ViPE inference -- SLAM/BA run on the GPU, so W_gpu dominates. Large G*: should
                    scale much further.

Same box, same fixed C, opposite verdicts. That contrast is the finding: the
right GPU count is a property of the *workload*, and `--gpus all` is a guess.

We measure W_cpu and W_gpu directly (per-sample CPU seconds and integrated NVML
utilization, see common/timing.py), so G* is *predicted from G=1 measurements*
and then checked against the measured curve rather than fitted to it.

Usage
-----
  python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --gpus 1 2 4 8
  python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --arms fork,ray-auto,ray-tuned
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from common.resources import cpu_quota, describe, threads_per_worker  # noqa: E402

PYTHON = os.environ.get("PYTHON", sys.executable)


def gpu_list(n: int) -> str:
    return ",".join(str(i) for i in range(n))


def run(cmd: list[str], log: Path) -> tuple[int, float]:
    """Run a launcher, tee its output to a log, return (rc, wall)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("w") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def arm_movebench_fork(g: int, args, out: Path, threads: int | None) -> list[str]:
    cmd = [
        str(REPO / "movebench" / "run_fork.sh"),
        "--dataset", str(args.dataset),
        "--gpus", gpu_list(g),
        "--out", str(out),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.device_policy != "original":
        cmd += ["--device-policy", args.device_policy]
    return cmd


def arm_movebench_ray(g: int, args, out: Path, mode: str) -> list[str]:
    cmd = [
        PYTHON, str(REPO / "movebench" / "run_ray.py"),
        "--dataset", str(args.dataset),
        "--gpus", str(g),
        "--threads-mode", mode,
        "--out", str(out),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.device_policy != "original":
        cmd += ["--device-policy", args.device_policy]
    return cmd


def arm_vipe_fork(g: int, args, out: Path, threads: int | None) -> list[str]:
    cmd = [
        str(REPO / "vipe_slow" / "run_fork.sh"),
        "--videos", str(args.dataset),
        "--gpus", gpu_list(g),
        "--out", str(out),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.stream_mode != "lazy":
        cmd += ["--stream-mode", args.stream_mode]
    if args.pose_only:
        cmd += ["--pose-only"]
    return cmd


def arm_vipe_ray(g: int, args, out: Path, mode: str) -> list[str]:
    cmd = [
        PYTHON, str(REPO / "vipe_slow" / "run_ray.py"),
        "--videos", str(args.dataset),
        "--gpus", str(g),
        "--threads-mode", mode,
        "--out", str(out),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.pose_only:
        cmd += ["--pose-only"]
    return cmd


def read_summary(out: Path) -> dict:
    for name in ("summary.json", "vipe_summary.json"):
        p = out / name
        if p.exists():
            return json.loads(p.read_text())
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", required=True, choices=["movebench", "vipe"])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument(
        "--arms",
        default="fork,ray-tuned",
        help="comma list of: fork, fork-unpinned, ray-auto, ray-tuned",
    )
    ap.add_argument("--limit", type=int, default=None, help="videos per run")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device-policy", default="original")
    ap.add_argument(
        "--stream-mode",
        default="lazy",
        choices=["lazy", "cached"],
        help="ViPE only. 'cached' reproduces the production launcher's "
        "ProcessedVideoStream(...).cache() -- decodes every frame into host RAM "
        "before inference (~30 MB/frame). This is the arm in which 4 GPUs beat 8.",
    )
    ap.add_argument(
        "--pose-only", action="store_true", help="ViPE: skip depth alignment"
    )
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="repeat each cell")
    args = ap.parse_args()

    root = args.out or REPO / "results" / f"{args.workload}_{args.tag}"
    root.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    quota = cpu_quota()

    env_info = describe()
    print("=" * 70)
    print(f" GPU-COUNT SWEEP  --  workload={args.workload}  arms={arms}")
    print(f" G values     : {args.gpus}")
    print(f" CPU budget C : {quota} (enforced)   nproc says {env_info['host_cpu_count']}")
    print(f" dataset      : {args.dataset}")
    print(f" out          : {root}")
    print("=" * 70)

    rows = []
    for arm in arms:
        for g in args.gpus:
            for rep in range(args.repeat):
                threads = threads_per_worker(g, quota)
                cell = f"{arm}_g{g}" + (f"_rep{rep}" if args.repeat > 1 else "")
                out = root / cell
                log = root / "logs" / f"{cell}.log"

                env = dict(os.environ)
                env["PYTHON"] = PYTHON
                if arm == "fork-unpinned":
                    env["THREADS"] = "0"
                elif arm == "fork":
                    env.pop("THREADS", None)

                if args.workload == "movebench":
                    if arm.startswith("fork"):
                        cmd = arm_movebench_fork(g, args, out, threads)
                    else:
                        cmd = arm_movebench_ray(g, args, out, arm.split("-", 1)[1])
                else:
                    if arm.startswith("fork"):
                        cmd = arm_vipe_fork(g, args, out, threads)
                    else:
                        cmd = arm_vipe_ray(g, args, out, arm.split("-", 1)[1])

                print(f"\n>>> {cell}: threads/worker={threads}  ({quota}/{g})", flush=True)
                t0 = time.perf_counter()
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as fh:
                    fh.write(f"$ {' '.join(cmd)}\n\n")
                    fh.flush()
                    rc = subprocess.run(
                        cmd, stdout=fh, stderr=subprocess.STDOUT, env=env
                    ).returncode
                wall = time.perf_counter() - t0

                s = read_summary(out)
                row = {
                    "arm": arm,
                    "G": g,
                    "rep": rep,
                    "threads_per_worker": 0 if arm == "fork-unpinned" else threads,
                    "rc": rc,
                    "wall_s": round(wall, 3),
                    "n_videos": s.get("n_videos"),
                    "throughput_v_per_s": (
                        round(s["n_videos"] / wall, 4) if s.get("n_videos") else None
                    ),
                    "cpu_s_per_video": s.get("cpu_seconds_per_video"),
                    "gpu_busy_s_per_video": s.get("gpu_busy_seconds_per_video"),
                    "straggler_s": s.get("straggler_s"),
                    "imbalance": s.get("shard_imbalance"),
                    "peak_loadavg": s.get("peak_loadavg"),
                    "steady_state_wall_mean_s": s.get("steady_state_wall_mean_s"),
                    "final_scores": s.get("final_scores"),
                    "log": str(log.relative_to(root)),
                }
                rows.append(row)
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(
                    f"    {status}  wall={wall:.1f}s  "
                    f"thr={row['throughput_v_per_s']} v/s  "
                    f"cpu/v={row['cpu_s_per_video']}s  gpu/v={row['gpu_busy_s_per_video']}s",
                    flush=True,
                )
                (root / "sweep.json").write_text(
                    json.dumps({"env": env_info, "args": vars(args) | {"dataset": str(args.dataset), "out": str(root)}, "rows": rows}, indent=2, default=str)
                )

    # -----------------------------------------------------------------
    # analysis: predict G* from the G=1 measurement, compare to the curve
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm and r["rc"] == 0]
        if not arm_rows:
            continue
        print(f"\n {arm}")
        print(f"   {'G':>2}  {'wall_s':>8}  {'v/s':>8}  {'speedup':>8}  "
              f"{'cpu_s/vid':>10}  {'gpu_s/vid':>10}  {'peak_load':>9}")
        base = next((r for r in arm_rows if r["G"] == min(r2["G"] for r2 in arm_rows)), None)
        for r in sorted(arm_rows, key=lambda x: (x["G"], x["rep"])):
            sp = (
                round(base["wall_s"] / r["wall_s"], 2)
                if base and r["wall_s"]
                else None
            )
            print(
                f"   {r['G']:>2}  {r['wall_s']:>8.1f}  "
                f"{(r['throughput_v_per_s'] or 0):>8.3f}  {str(sp):>8}  "
                f"{str(r['cpu_s_per_video']):>10}  {str(r['gpu_busy_s_per_video']):>10}  "
                f"{str(r['peak_loadavg']):>9}"
            )
        g1 = next((r for r in arm_rows if r["G"] == 1), None)
        if g1 and g1.get("cpu_s_per_video") and g1.get("gpu_busy_s_per_video"):
            w_cpu, w_gpu = g1["cpu_s_per_video"], g1["gpu_busy_s_per_video"]
            g_star = quota * w_gpu / w_cpu if w_cpu else float("inf")
            print(
                f"   model: W_cpu={w_cpu}s/vid  W_gpu={w_gpu}s/vid  C={quota}\n"
                f"          predicted G* = C*W_gpu/W_cpu = {g_star:.2f}  "
                f"-> scaling should stop helping past G~{max(1, round(g_star))}"
            )

    print(f"\n raw: {root / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
