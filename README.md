# ⚡ More GPUs Is Not Always Faster

This repository measures how video evaluation scales across GPUs. Its purpose is
simple: help you choose a GPU count and CPU-thread budget that make the pipeline
faster, instead of assuming that every available GPU should run.

## ⭐ The finding

Two video pipelines were tested on 1, 2, 4, and 8 NVIDIA H100 GPUs. The bottleneck
was often the host CPU, not the GPU. Starting one worker per GPU without limiting
CPU threads made workers compete for the same cores; in one case, the GPUs were only
7.3% busy.

| 48-video evaluation on 8 H100s | Wall time | Speedup |
|---|---:|---:|
| Default threads; metrics on CPU | 1885.7 s | 1.0× |
| CPU threads split across workers | 180.1 s | 10.5× |
| Metrics moved to GPU | **34.2 s** | **55.1×** |

> 💡 **Takeaway:** profile the workload first. More GPUs help only while the CPU
> can keep every GPU worker supplied with work.

## 🧠 Why this happens

PyTorch and NumPy create CPU thread pools. With eight GPU workers, letting every
worker use the machine-wide default can request far more threads than the job has
CPUs. The result is CPU throttling, cache contention, and idle GPUs.

| Configuration | Threads per worker | Total requested | Result |
|---|---:|---:|---|
| Uncapped default | 128 | 1024 | 💥 5× over a 198-CPU quota |
| Capped | 24 | 192 | ✅ fits the quota |

Use this rule for one process per GPU:

```text
threads per worker = floor(CPU budget / number of GPU workers)
```

`common/resources.py` reads the CPU budget actually available to the job, including
container cgroup limits. Do not size the pool from `nproc` alone.

## 💡 Helpful tips

- **Set thread limits before importing PyTorch.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  and `OPENBLAS_NUM_THREADS` must be set before a worker imports its numerical libraries.
  Changing them later may not resize an existing thread pool.
- **Check `torchrun` explicitly.** It commonly sets `OMP_NUM_THREADS=1` when the variable
  is unset. This prevents oversubscription, but it can leave CPU-heavy work underused.
- **Do not trust Ray's defaults.** Ray's `auto` mode in this project gives actors one
  CPU thread. Use the `ray-tuned` arm, which requests `quota / GPU workers` CPUs and sets
  PyTorch's thread count inside each actor.
- **Warm and persist model caches.** A first run may download checkpoints or compile
  extensions. Keep caches on node-local persistent storage (including when using tools
  such as DiffSynth) and exclude that cold-start work from performance timing.
- **Record the effective settings.** Save the GPU count, CPU quota, thread count, and
  cache state with each result; otherwise two runs with the same command can be misleading.

## 🛠️ Run on your hardware

First, inspect the CPU allocation visible to the job:

```bash
python common/resources.py
```

Then sweep only GPU counts that exist on your machine:

```bash
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 \
  --gpus 1 2 4 --arms fork --limit 48 --max-frames 24 --tag my-hardware
```

For a direct run, select GPU IDs and set threads per worker. For example, with
32 available CPUs and four GPU workers, use eight threads each:

```bash
THREADS=8 movebench/run_fork.sh --dataset data/eval81 --gpus 0,1,2,3
```

> ⚠️ In `exp_gpu_sweep.py`, `--gpus 1 2 4` means GPU **counts** to test. In
> `movebench/run_fork.sh`, `--gpus 0,1,2,3` means GPU **IDs** to use.

If `THREADS` is unset, `run_fork.sh` automatically uses
`floor(available CPUs / selected GPUs)`. Start with one GPU and increase the count
only while the measured wall time improves.

## 📊 What is included

- `movebench/` — video-quality and motion evaluation (CLIP, EPE, LPIPS, SSIM, PSNR, FVD)
- `vipe_slow/` — NVIDIA ViPE camera-pose and metric-depth pipeline
- `exp_gpu_sweep.py` — runner for GPU-count and launcher comparisons
- `common/resources.py` — CPU quota detection and per-worker thread calculation
- `results/` — committed summaries, worker shards, and load traces behind the numbers

The measured results show two useful patterns:

1. Capping CPU threads made the evaluation pipeline scale monotonically to eight GPUs.
2. In the ViPE production configuration, four GPUs initially beat eight because every
   worker decoded video at once; after capping threads, eight GPUs became faster.

## 🚀 Install

```bash
conda create -n vipe-new python=3.11 -y
conda activate vipe-new
pip install -U pip uv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
cd vipe_slow && pip install -e . --no-build-isolation && cd ..
```

The original measurements used 8× H100 80 GB GPUs and a 198-CPU Docker cgroup quota.
Your best GPU count will depend on your own CPU allocation, GPU type, dataset, and
whether the CPU-heavy stages run on the host or the GPU.
