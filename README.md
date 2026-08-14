# Faster Video Evaluation Through Better Resource Scheduling

This repository measures and improves multi-GPU video evaluation. Its central
lesson is simple: adding GPUs only helps when the host CPU can keep their workers
fed. For CPU-heavy pipelines, an uncontrolled thread pool can make eight GPUs
slower than one.

## Results at a glance

Measurements were made on 8× NVIDIA H100 80 GB GPUs with a Docker CPU quota of
198 cores. The MoveBench evaluation ran 48 video pairs, using 24 frames per
video.

| 8-GPU configuration | Wall time | Improvement |
|---|---:|---:|
| Default CPU threading; metrics on CPU | 1,885.7 s | baseline |
| CPU threads divided across workers | 180.1 s | 10.47× faster |
| Thread tuning + metrics placed on GPU | **34.2 s** | **55.10× faster** |

The original bottleneck was host-side work, not GPU compute: the default setup
requested 128 CPU threads per worker—1,024 threads across eight workers—despite
a 198-CPU quota. GPU utilization fell to 0.33%.

> Read the [detailed measurement report](report.md) for
> methodology, all sweep results, caveats, and the engineering lessons behind
> these numbers.

## Key findings

- Treat GPU count and CPU threads per worker as one tuning decision. Start with:
  `floor(available CPU quota / GPU workers)`.
- Detect the CPU quota available to the job. In containers, `nproc` can report
  host CPUs rather than the cgroup limit; `common/resources.py` handles this.
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` before
  importing PyTorch or NumPy.
- Do not rely on launcher defaults. `torchrun` may set one thread per process,
  while Ray's default actor allocation may also underuse the CPU.
- Profile actual device activity. CUDA event duration can include host waiting;
  this project records NVML GPU-busy time to distinguish active compute from
  an idle GPU holding a stream.

## Workloads and tools

| Path | Purpose |
|---|---|
| `movebench/` | Video-quality and motion evaluation: CLIP, EPE, LPIPS, SSIM, PSNR, and FVD |
| `vipe_slow/` | NVIDIA ViPE camera-pose and metric-depth evaluation |
| `common/resources.py` | Cgroup-aware CPU-quota detection and per-worker thread calculation |
| `exp_gpu_sweep.py` | Reproducible GPU-count and launcher comparison sweep |
| `results/` | Committed summaries, worker shards, and load traces behind the report |

The two workloads scale differently. With threads capped, the CPU-heavier
MoveBench evaluation reached 3.93× speedup on eight GPUs, while ViPE reached
5.45×. In ViPE's production configuration, four GPUs initially beat eight;
the report reproduces that result and isolates its cause.

## Run on your hardware

Inspect the CPUs that your job can really use:

```bash
python common/resources.py
```

Sweep the GPU counts available on your machine:

```bash
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 \
  --gpus 1 2 4 --arms fork --limit 48 --max-frames 24 --tag my-hardware
```

For a direct four-GPU run with a 32-CPU allocation, give each worker eight
threads:

```bash
THREADS=8 movebench/run_fork.sh --dataset data/eval81 --gpus 0,1,2,3
```

`exp_gpu_sweep.py --gpus 1 2 4` specifies GPU *counts*. In contrast,
`run_fork.sh --gpus 0,1,2,3` specifies GPU *IDs*. If `THREADS` is unset,
`run_fork.sh` selects `floor(available CPUs / selected GPUs)` automatically.

## Installation

```bash
conda create -n vipe-new python=3.11 -y
conda activate vipe-new
pip install -U pip uv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
cd vipe_slow && pip install -e . --no-build-isolation && cd ..
```

Your best GPU count depends on the CPU quota, GPU model, dataset, and which
stages execute on the host. Measure one GPU first, then add GPUs only while
end-to-end wall time improves.
