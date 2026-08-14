"""Per-sample CPU-time / GPU-time attribution.

The instrument behind this repo's central claim: *splitting a workload across
every GPU is not always fastest, and you cannot tell which GPU count is fastest
without measuring where each sample's time actually goes.*

What we measure per sample
--------------------------
wall_s        elapsed wall-clock for the sample
cpu_s         process CPU time (user+sys) consumed during it, across all threads
gpu_s         time the GPU spent executing this sample's kernels
cpu_par       cpu_s / wall_s -- the mean number of CPU cores kept busy

`cpu_par` is the number that decides everything. A sample with cpu_par ~= 20 is
using 20 cores; run G of those workers concurrently and you need 20*G cores. On
a 198-CPU container, G=8 needs 160 cores of CPU headroom, and if each worker's
OMP pool is only 24 threads wide it cannot even reach its own demand. That is
the mechanism by which adding GPUs makes a pipeline slower.

How gpu_s is measured
---------------------
Two different numbers, because they answer two different questions and
conflating them is how people talk themselves into "the GPU is the bottleneck":

  gpu_s       CUDA-event elapsed time across the sample. Events are recorded on
              the stream, so this is device-side *elapsed* time -- it includes
              every gap where the GPU sat idle waiting for the host. For a
              synchronous single-stream workload it therefore tracks wall time
              closely, and on its own tells you almost nothing.

  gpu_busy_s  integrated SM utilization, sampled from NVML in a background
              thread at ~50 Hz. This is the honest "how much of the sample did
              the GPU actually work" number, and `gpu_busy_s / wall_s` is the
              occupancy that decides whether a workload is GPU-bound.

The pair is what makes the argument: a sample with gpu_s ~= wall_s but
gpu_busy_s / wall_s ~= 0.2 is a *host-bound* sample that merely looks
GPU-resident. Every CPU-bound metric in this suite looks like that.

Caveat kept honest: `cpu_s` is process-wide, so in a single-process-per-GPU
launcher it is that worker's CPU consumption (what we want). In the Ray actor
pool, one process hosts one actor, so it is still per-worker -- but a Ray worker
process also runs Ray's own IPC threads, which inflates cpu_s slightly. We
report it and do not correct for it. NVML utilization is likewise *per device*,
so it is only attributable to one worker when that worker owns the GPU alone --
true for every configuration in this repo (one worker per GPU, pinned).
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, field, asdict


@dataclass
class SampleTiming:
    """One unit of work (one video)."""

    name: str
    wall_s: float
    cpu_s: float
    gpu_s: float | None = None
    gpu_busy_s: float | None = None
    cpu_par: float = 0.0
    gpu_util_mean: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cpu_par = round(self.cpu_s / self.wall_s, 3) if self.wall_s > 0 else 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _process_cpu_time() -> float:
    """User+sys CPU seconds for this process, summed over all threads."""
    t = os.times()
    return t.user + t.system


class _GpuUtilSampler:
    """Background NVML poller: integrates SM utilization over a window.

    Runs in a daemon thread at `hz`, costs ~one NVML call per tick (tens of
    microseconds), and is the only way to separate "GPU resident" from "GPU
    busy". Degrades to a no-op if NVML is unavailable.
    """

    def __init__(self, device_index: int = 0, hz: float = 50.0):
        self._handle = None
        self._nvml = None
        self._interval = 1.0 / hz
        self._lock = threading.Lock()
        self._util_sum = 0.0
        self._ticks = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            # Respect CUDA_VISIBLE_DEVICES: index 0 of the visible set.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            phys = device_index
            if visible:
                ids = [d for d in visible.split(",") if d.strip() != ""]
                if device_index < len(ids):
                    try:
                        phys = int(ids[device_index])
                    except ValueError:
                        phys = device_index
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(phys)
        except Exception:
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            except Exception:
                continue
            with self._lock:
                self._util_sum += float(u)
                self._ticks += 1

    def start(self):
        if not self.available or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def snapshot(self) -> tuple[float, int]:
        with self._lock:
            return self._util_sum, self._ticks


class SampleTimer:
    """Context manager measuring one sample.

    Usage:
        timer = SampleTimer(use_cuda=True)
        with timer.sample("video_003") as s:
            score_one_video(...)
        print(s.wall_s, s.cpu_s, s.gpu_busy_s, s.cpu_par)
        ...
        timer.summary()   # aggregate
    """

    def __init__(self, use_cuda: bool = True, sample_gpu_util: bool = True):
        self.samples: list[SampleTiming] = []
        self._cuda = False
        self._util: _GpuUtilSampler | None = None
        if use_cuda:
            try:
                import torch

                self._cuda = torch.cuda.is_available()
            except Exception:
                self._cuda = False
        if self._cuda and sample_gpu_util:
            self._util = _GpuUtilSampler()
            self._util.start()

    def close(self):
        if self._util is not None:
            self._util.stop()

    @contextlib.contextmanager
    def sample(self, name: str, **extra):
        import_torch = None
        start_evt = end_evt = None
        if self._cuda:
            import torch

            import_torch = torch
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        util0, ticks0 = self._util.snapshot() if self._util else (0.0, 0)
        cpu0 = _process_cpu_time()
        wall0 = time.perf_counter()

        holder = SampleTiming(name=name, wall_s=0.0, cpu_s=0.0, extra=dict(extra))
        try:
            yield holder
        finally:
            gpu_s = None
            if self._cuda and import_torch is not None:
                end_evt.record()
                import_torch.cuda.synchronize()
                gpu_s = start_evt.elapsed_time(end_evt) / 1000.0

            wall = time.perf_counter() - wall0
            cpu = _process_cpu_time() - cpu0

            util_mean = busy = None
            if self._util:
                util1, ticks1 = self._util.snapshot()
                dt = ticks1 - ticks0
                if dt > 0:
                    util_mean = (util1 - util0) / dt  # percent
                    busy = wall * util_mean / 100.0

            holder.wall_s = wall
            holder.cpu_s = cpu
            holder.gpu_s = gpu_s
            holder.gpu_busy_s = round(busy, 4) if busy is not None else None
            holder.gpu_util_mean = round(util_mean, 2) if util_mean is not None else None
            holder.__post_init__()  # recompute cpu_par
            self.samples.append(holder)

    def summary(self) -> dict:
        """Aggregate; the per-sample means are what the sweep compares."""
        if not self.samples:
            return {"n": 0}
        n = len(self.samples)
        wall = sum(s.wall_s for s in self.samples)
        cpu = sum(s.cpu_s for s in self.samples)
        gpus = [s.gpu_s for s in self.samples if s.gpu_s is not None]
        busy = [s.gpu_busy_s for s in self.samples if s.gpu_busy_s is not None]
        return {
            "n": n,
            "wall_total_s": round(wall, 3),
            "wall_mean_s": round(wall / n, 3),
            "cpu_total_s": round(cpu, 3),
            "cpu_mean_s": round(cpu / n, 3),
            "gpu_total_s": round(sum(gpus), 3) if gpus else None,
            "gpu_mean_s": round(sum(gpus) / len(gpus), 3) if gpus else None,
            "gpu_busy_total_s": round(sum(busy), 3) if busy else None,
            "gpu_busy_mean_s": round(sum(busy) / len(busy), 3) if busy else None,
            # The headline pair: mean cores busy, and the honest GPU occupancy.
            "cpu_parallelism": round(cpu / wall, 3) if wall > 0 else None,
            "gpu_occupancy": round(sum(busy) / wall, 3) if busy and wall > 0 else None,
        }

    def steady_state_summary(self, skip: int = 1) -> dict:
        """Same, dropping the first `skip` samples (model load / cudnn autotune)."""
        if len(self.samples) <= skip:
            return self.summary()
        warm = SampleTimer(use_cuda=False, sample_gpu_util=False)
        warm.samples = self.samples[skip:]
        out = warm.summary()
        out["skipped_warmup"] = skip
        return out


def thread_env_report() -> dict:
    """What the thread knobs actually ended up as, inside this worker.

    Recording this next to every benchmark is the discipline that catches the
    class of bug this whole repo is about: a launcher that *requests* N threads
    and a library that quietly uses a different number.
    """
    out = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    }
    try:
        import torch

        out["torch_num_threads"] = torch.get_num_threads()
        out["torch_num_interop_threads"] = torch.get_num_interop_threads()
        out["cuda_device_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return out


def check_thread_mismatch(requested: int) -> dict:
    """Compare what we asked for against what torch actually did.

    Returns a dict with `mismatch: bool`. Callers should print loudly on True --
    a silent mismatch here invalidates every number in the run.
    """
    rep = thread_env_report()
    actual = rep.get("torch_num_threads")
    mismatch = actual is not None and int(actual) != int(requested)
    return {
        "requested": requested,
        "actual_torch_threads": actual,
        "omp_env": rep.get("OMP_NUM_THREADS"),
        "mismatch": mismatch,
    }
