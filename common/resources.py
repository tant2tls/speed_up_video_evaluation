"""Container-aware resource detection.

The one module every launcher in this repo imports before it decides how many
threads to give a worker.

Why this file exists
--------------------
On this box (and on every Docker/Kubernetes/RunAI container we have measured),
the number of CPUs a process is *allowed* to use and the number it *sees* are
different numbers:

    nproc / os.cpu_count() / len(os.sched_getaffinity(0))  ->  256   (the HOST)
    cgroup CFS quota                                       ->  198   (enforced)

cgroup v1 CPU throttling does not shrink the CPU affinity mask. It lets the
process spawn as many threads as it likes and then, once the cgroup has burned
`quota` CPU-seconds inside a `period`, freezes every thread until the next
period starts. So oversubscription does not raise an error. It just makes
everything slower, invisibly.

Any code that sizes a thread pool, a `DataLoader(num_workers=...)`, or
`OMP_NUM_THREADS` off `nproc` therefore oversubscribes by 256/198 = 1.29x on
this box, and by an arbitrary factor on a smaller container.

Use `cpu_quota()` instead. See README.md "Tip 4".
"""

from __future__ import annotations

import os
from pathlib import Path

# cgroup v1 and v2 expose the quota in different places/formats.
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
_CGROUP_V2_MAX = Path("/sys/fs/cgroup/cpu.max")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def cgroup_cpu_quota() -> float | None:
    """CPUs this container may actually use, or None if unlimited/unavailable.

    Returns a float because a quota need not be a whole number of CPUs
    (e.g. quota=150000 period=100000 -> 1.5 CPUs).
    """
    # cgroup v2: "<quota> <period>" or "max <period>"
    if _CGROUP_V2_MAX.exists():
        try:
            quota_s, period_s = _CGROUP_V2_MAX.read_text().split()
            if quota_s != "max":
                period = int(period_s)
                if period > 0:
                    return int(quota_s) / period
        except (OSError, ValueError):
            pass

    # cgroup v1: two files. quota == -1 means unlimited.
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    if quota is not None and period is not None and quota > 0 and period > 0:
        return quota / period

    return None


def host_cpu_count() -> int:
    """What naive auto-detection reports. Here for contrast, not for sizing."""
    return os.cpu_count() or 1


def affinity_cpu_count() -> int:
    """The CPU affinity mask size. Also wrong under CFS quota (it is not shrunk)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return host_cpu_count()


def cpu_quota() -> int:
    """The real, enforced CPU ceiling for this process. **Size pools off this.**

    Falls back to the affinity mask when no quota is set (bare metal, or a
    container run without `--cpus`).
    """
    quota = cgroup_cpu_quota()
    if quota is None:
        return affinity_cpu_count()
    # Floor, and never return 0: a 0.5-CPU container still gets one thread.
    return max(1, int(quota))


def threads_per_worker(num_workers: int, budget: int | None = None) -> int:
    """Split the CPU budget evenly across `num_workers` GPU workers.

    This is the number that goes into `OMP_NUM_THREADS` for each worker.
    The whole point: the denominator is the *quota* (198), never `nproc` (256).

        threads_per_worker(8) -> 24     # 198 // 8, not 256 // 8 == 32
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    budget = cpu_quota() if budget is None else budget
    return max(1, budget // num_workers)


def describe() -> dict:
    """Everything a benchmark should record about the CPU environment."""
    quota = cgroup_cpu_quota()
    return {
        "cpu_quota_enforced": cpu_quota(),
        "cgroup_quota_raw": quota,
        "host_cpu_count": host_cpu_count(),
        "affinity_cpu_count": affinity_cpu_count(),
        "lies_by_factor": (
            round(host_cpu_count() / cpu_quota(), 3) if cpu_quota() else None
        ),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
    }


def gpu_count() -> int:
    """Visible GPU count without importing torch (cheap; safe pre-fork)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip() != ""])
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
        return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


if __name__ == "__main__":
    import json

    info = describe()
    info["gpu_count"] = gpu_count()
    print(json.dumps(info, indent=2))
    q = info["cpu_quota_enforced"]
    print()
    print(f"  Real CPU ceiling (use this):      {q}")
    print(f"  What nproc/os.cpu_count() says:   {info['host_cpu_count']}")
    if info["lies_by_factor"] and info["lies_by_factor"] > 1.01:
        print(
            f"  -> naive auto-detect oversubscribes by {info['lies_by_factor']}x "
            "and gets silently CFS-throttled."
        )
    for g in (1, 2, 4, 8):
        print(f"  OMP_NUM_THREADS for G={g} workers: {threads_per_worker(g, q)}")
