# 🚀 More GPUs Is Not Linearly Faster

### Finding the *death-linear point* of two real video pipelines on 8× H100

🎯 **The purpose.** Two video pipelines from a real project ([UniCaMo](https://phongnhhn.info/Unicamo/)) —
a 6-metric evaluation suite and NVIDIA ViPE for camera pose + depth — run on 1, 2, 4 and 8 GPUs
to find **where adding a GPU stops paying for itself**. Both are embarrassingly parallel over
video clips, so the obvious move is to shard them across every GPU you own. For both, that was
the wrong move — in opposite directions.

💡 **Why it happens.** A "GPU pipeline" usually isn't GPU-bound. One of these two burned **1390×
more CPU time than GPU time** and left the GPU **93% idle**. Once that's true your limit is *CPU
cores per worker* — and **adding GPUs divides them**.

> 📖 This README is the overview: the transferable rules and the findings.
> **Every measurement, method, reversal and caveat is in [`report.md`](report.md).**

---

## ⭐ Results at a glance

**How long one video takes, at each GPU count.** Same work, same box — only the GPU count and
the host configuration change. 🟢 marks each row's best, and the row that *gets worse* as you add
GPUs is the whole point of this repo:

| | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs | best |
|---|---:|---:|---:|---:|---|
| **movebench** — no thread limit *(the obvious way)* | 14.85 s | 🟢 **9.09 s** | 32.80 s 💥 | 39.29 s 💥 | **2 GPUs** |
| **movebench** — threads capped to `quota ÷ GPUs` | 14.77 s | 7.33 s | 4.85 s | 🟢 **3.75 s** | 8 GPUs |
| **movebench** — capped **+** metrics moved onto the GPU | 3.14 s | 1.69 s | 1.02 s | 🟢 **0.71 s** 🚀 | 8 GPUs |
| **ViPE** — as production ships it *(no thread limit)* | 56.03 s | 30.67 s | 🟢 **23.07 s** | 24.27 s ⚠️ | **4 GPUs** |
| **ViPE** — threads capped | 55.02 s | 29.40 s | 15.99 s | 🟢 **10.08 s** | 8 GPUs |

*movebench = 48 videos × 24 frames; ViPE = 8 videos × 81 frames. Per-video times aren't
comparable between the two workloads — the ratios and the shape of each row are.*

Two rows peak before 8 GPUs and then get *worse* — **movebench at 2, ViPE at 4** — and in both
cases the cause is the host, not the GPUs. Cap the threads and both rows become monotonic. 📉 That
turning point is the **death-linear point** ([§2](#2--the-death-linear-point-)).

### 🚀 And on a fixed 8 GPUs, the host config is worth 55×

| 48 videos, 8 GPUs | wall | per video | cumulative |
|---|---:|---:|---:|
| no thread limit, metrics on CPU — *the obvious way* | 1885.7 s | 39.29 s | 1.00× |
| **+** threads capped to `quota ÷ GPUs` | 180.1 s | 3.75 s | **10.47×** ⚡ |
| **+** LPIPS/SSIM/PSNR moved onto the accelerator | **34.2 s** | **0.71 s** | **55.10×** 🚀 |

The two factors multiply almost exactly (10.47 × 5.26 = 55.10), which is the evidence they fix
*independent* problems. **Adding the other 7 GPUs was worth 3.94× — the host fixes were ~14× more
valuable than the hardware.**

---

## 💡 Helpful tips

Six things worth checking before you blame the GPU — none of them specific to these pipelines.
*Full write-ups with the numbers: [`report.md` §13](report.md#13--nine-engineering-lessons).*

**🧵 Split the CPU budget across workers yourself — before `import torch`.**
Each worker sizes its thread pool for the whole machine, unaware the others exist, so `N` workers
ask for `N ×` the box. Set `OMP_NUM_THREADS = quota ÷ workers`; after torch loads it's a silent
no-op. The two defaults sit at opposite extremes: `torchrun` quietly forces 1 thread, plain
`python` lets torch grab half the host.

**⚡ Ray defaults to one CPU per actor — which means one thread.**
It reads the cgroup quota correctly, then maps `num_cpus` (default **1**) onto
`OMP_NUM_THREADS`, so every actor is single-threaded while most of the box sits idle. Set
`num_cpus` explicitly *and* call `torch.set_num_threads()` inside the actor — `num_cpus` alone is
import-order dependent.

**🎯 Ask *where* your math runs, not just how it's scheduled.**
Scheduling around a stage that's on the wrong device treats the symptom. Look for tensors that
never leave the host; one unnoticed CPU-resident stage can dominate everything else.

**🔍 In a container, `nproc` doesn't report your quota.**
It reports the host. Throttling doesn't shrink the affinity mask, so you can spawn far more threads
than you may actually run — then all of them freeze once the quota burns. No error, just slowness.
Read the cgroup quota (Docker, Kubernetes, Slurm and RunAI all have this gap).

**🌐 On multi-node, check what else is on the network before you trust a timing.**
Anything shared — NFS, an object store, a download lock, the interconnect — behaves like a
different machine depending on who else is using it. On a quiet network a job flies; on a busy one
the same binary crawls or dies, because contention grows *super-linearly* in rank count while
collective timeouts stay fixed. What merely wasted time at low load starts tripping watchdogs at
high load. **Measure at different times of day, keep per-rank timestamps, and treat one run on a
shared filesystem as a lower bound.**

**📦 Check what your framework does at *every* rank start.**
Loaders often re-query a model hub even when the weights are already local, and hold a lock while
doing it. Put that lock on a shared filesystem and it becomes fleet-global: ranks block on each
other waiting to download nothing, until a watchdog kills the job. Keep caches node-local, pass
explicit local paths, and verify the load path is a no-op offline. *(DiffSynth specifically:
`export DIFFSYNTH_SKIP_DOWNLOAD=true` — only the literal `true` parses.)*

> 🧠 **Profile before you scale.** The GPU count belongs in the profile, not in the launch script
> as `--gpus all` — the resource that limits an ML workload is usually not the one it's named after.

---

## 🧵 What "capped" and "uncapped" mean

Every table below is labelled *capped* or *uncapped*, answering one question: **did anyone tell
each worker how many CPU threads it may use?** On this box the enforced quota was **198 CPUs**
across 8 workers:

| | what you run | threads/worker | 8 workers ask for | vs the quota |
|---|---|---|---|---|
| **uncapped** (the default) | nothing — you set nothing | 128 (torch's guess) | **1024** | **5× oversubscribed** 💥 |
| **capped** (the fix) | `export OMP_NUM_THREADS=24` | 24 = `198 ÷ 8` | **192** | fits ✅ |

Oversubscription is worse than proportionally slower: threads spend their time being switched on
and off the CPU and fighting over cache, and the cgroup freezes all of them once the quota burns.

⚠️ **The trap: uncapped is what you get by not thinking about it, and it produces *correct*
numbers — just many times slower, with no error and no warning.** That is why it survives in
production.

*This limits thread **count**, not CPU affinity — nothing here uses `taskset`.
[`report.md`](report.md) calls the same two configurations **pinned** / **unpinned**.*

---

## 📊 The seven findings

| # | Finding | details |
|---|---|---|
| 1 | ⭐ **55.1× faster on the same 8 GPUs** (1885.7 → 34.2 s) from two host-side config changes; the 7 extra GPUs were worth 3.94×. | [§1](#1--the-bottleneck-was-the-host-not-the-gpu-) · [report §9](report.md#9--device_policygpu--fixing-the-placement-instead-of-scheduling-around-it) |
| 2 | A "GPU evaluation pipeline" spent **1390× more CPU than GPU time**, GPU **7.3% busy**. Its bottleneck was never the accelerator. | [§1](#1--the-bottleneck-was-the-host-not-the-gpu-) · [report §2](report.md#2--the-two-workloads-are-on-opposite-sides-of-the-cpugpu-balance) |
| 3 | 🎯 **Fixing *placement* beats tuning the schedule** — 3 metrics onto the GPU cut CPU-s/video **57×** and made compute scale at **93%** efficiency. | [§1](#1--the-bottleneck-was-the-host-not-the-gpu-) · [report §9](report.md#9--device_policygpu--fixing-the-placement-instead-of-scheduling-around-it) |
| 4 | 📉 Adding GPUs helps **only as far as the host keeps up**: capped, eval scales 3.94× and ViPE 5.46×; uncapped, eval **collapses 10.47×**. | [§2](#2--the-death-linear-point-) · [report §4](report.md#4--the-unpinned-arm-the-thread-storm-measured) |
| 5 | 🔬 The `min(GPU, CPU)` model this started from **was wrong, and being wrong is the finding** — `W_cpu` is a function of thread count. An Amdahl model *does* hold after the fix. | [§2](#2--the-death-linear-point-) · [report §3](report.md#3--gpu-count-sweep-threads-pinned-to-198g--the-two-workloads-scale-differently) |
| 6 | ⭐ **For ViPE, 4 GPUs run ~1.25× faster than 8** (n=3) in the config production ships — because at G=8 every worker decodes at once, not anything about GPUs. | [§3](#3--vipe-4-gpus-really-are-faster-than-8-) · [report §12](report.md#12--vipe-4-gpus-really-is-faster-than-8--reproducing-the-production-result) |
| 7 | ⚖️ **The launcher is worth ~3%; the thread configuration is worth 10×.** Two correct launchers land within 3%; two misconfigured ones are 6.6× and 10.5× slower, in *opposite* directions. | [§4](#4--ray-vs-bash-fork-worth-3-) · [report §5](report.md#5--ray-vs-bash-fork) |

📁 Every number comes from committed JSON under `results/` — **402 files, ~1.5 MB** — so the
tables recompute **without an 8-GPU box**. Four thin claims were re-measured **n=3**
(`results/*_rep/`): ViPE 4-beats-8 got *stronger*; the load-balancing claim narrowed to the
metric that survived.

⚠️ **Two caveats up front.** The 1885.7 s baseline is a real configuration but also the worst of
the four launchers in [§4](#4--ray-vs-bash-fork-worth-3-), so 55.1× spans the naive-to-tuned
distance, not headroom every deployment is sitting on. And moving metrics to the GPU shifts LPIPS
by up to **6.7e-4 relative** (cuDNN picks different conv algorithms); every aggregate still
matches to 4 decimals.

---

## 🎥 Where this comes from

**[UniCaMo](https://phongnhhn.info/Unicamo/)** (*Unifying Camera and Motion Control for Video
Generation*) generates video under two controls at once — what the subject does, and where the
camera goes. Controlling both means measuring both:

| folder | role in UniCaMo | what it computes |
|---|---|---|
| **`movebench/`** | evaluation — **motion + quality** | CLIP-I · optical-flow EPE (RAFT) · LPIPS · SSIM · PSNR · FVD (I3D) |
| **`vipe_slow/`** | **preprocessing** (labels for every training clip) **and** evaluation — **camera** | NVIDIA ViPE — camera pose + metric depth, SLAM with learned priors |

⭐ **ViPE sits on the critical path twice** — once over the training corpus to produce the
depth/pose labels the model conditions on, and again over generated video to check whether the
camera went where it was told. Its throughput gates dataset prep *and* every evaluation round,
which is why [§3](#3--vipe-4-gpus-really-are-faster-than-8-) gets its own section.

Each pipeline runs under **four launchers** doing identical arithmetic and differing only in
orchestration, so any wall-clock difference is attributable to scheduling. Parity is structural:
every launcher calls the same `score_one_video()` / `pipeline.run()`.

---

## 1 · The bottleneck was the host, not the GPU 🖥️

Per video at G=1, full thread budget:

| | movebench eval (24 fr) | ViPE pose+depth (81 fr) |
|---|---|---|
| **CPU-seconds consumed** | **1469 s** | **1056 s** |
| **GPU-busy-seconds** | **1.06 s** | **12.38 s** |
| GPU utilization | **7.3%** 😴 | **23.3%** |
| **CPU : GPU time ratio** | **1390 : 1** | **85 : 1** |

The cause is **placement, not volume**: LPIPS-VGG, SSIM and PSNR are handed CPU tensors and never
moved to the accelerator, and LPIPS (a VGG16 forward per frame) dominates. Two pipelines over the
same videos on the same hardware land on opposite sides of the CPU/GPU balance.

🔬 CUDA events report `gpu_s ≈ wall_s` because the stream is resident the whole time; integrated
NVML utilization says the GPU actually worked 7% of it. **Event-elapsed time would have made a
host-bound workload look GPU-bound**, so every "it's the host" claim rests on `gpu_busy_s`.

So the repair is placement (`--device-policy gpu`), not scheduling:

| G | original | **device_policy=gpu** | speedup | CPU-s/video |
|---|---|---|---|---|
| 1 | 708.9 s | **151.0 s** | 4.70× | 1469 → 25.7 (**57×**) |
| 2 | 351.9 s | **81.1 s** | 4.34× | 832 → 15.0 (56×) |
| 4 | 232.9 s | **49.0 s** | 4.75× | 557 → 9.8 (57×) |
| 8 | 180.1 s | **34.2 s** | **5.26×** 🚀 | 431 → 7.5 (**57×**) |

The *shape* of the curve changes, not just the constant: compute efficiency at G=8 goes **53% →
93%**, GPU occupancy **7.3% → ~66%**, and the sublinearity that remains is a fixed **≈14.0 s**
serial cost (model load + merge). A one-parameter Amdahl model predicts the whole curve within
**1–9%** — and predicts something falsifiable: **the ceiling is 10.77×** however many GPUs you
add. Past this fix, optimize the serial cost, not the GPU count.
→ [report §9](report.md#9--device_policygpu--fixing-the-placement-instead-of-scheduling-around-it)

---

## 2 · The death-linear point 📉

The **death-linear point** is the worker count past which adding a GPU stops buying proportional
throughput, because a fixed per-item *host* cost has become the binding constraint instead of the
accelerator. It's a property of the workload, not the hardware.

**Capped** — both curves monotonic. ViPE holds 68% efficiency against eval's 49% because it has
12.4 GPU-s per video (vs 1.06) to hide the host stage behind:

| G | threads | **eval** wall | speedup | eff. | **ViPE** wall | speedup | eff. |
|---|---|---|---|---|---|---|---|
| 1 | 198 | 708.9 s | 1.00× | 100% | 440.2 s | 1.00× | 100% |
| 2 | 99 | 351.9 s | 2.01× | 101% | 235.2 s | 1.87× | 94% |
| 4 | 49 | 232.9 s | 3.04× | 76% | 127.9 s | 3.44× | 86% |
| 8 | 24 | 180.1 s | **3.94×** | **49%** | 80.6 s | **5.46×** | **68%** |

**Uncapped** — what you get by setting nothing. The eval curve falls apart 💥:

| G | total threads | uncapped | capped | penalty | peak loadavg (quota **198**) |
|---|---|---|---|---|---|
| 1 | 128 | 712.9 s | 708.9 s | 1.01× | 126 |
| 2 | 256 | 436.5 s | 351.9 s | 1.24× | **202** |
| 4 | 512 | 1574.2 s | 232.9 s | **6.76×** | **377** |
| 8 | 1024 | **1885.7 s** | 180.1 s | **10.47×** | **502** 🔥 |

This is the genuine "more GPUs made it slower" curve: throughput peaks at **G=2** and collapses.
Two controls identify the cause — `gpu_busy_s` stays ≈1 s/video in *every* cell (the GPU does
identical work; only host time inflates), and peak load crosses the quota exactly at G=2, the
first configuration where `G × 128` exceeds it. That's CFS throttling, not GPU contention. Scores
were identical in every cell: it costs throughput, not correctness.

🔬 **The model this started from was wrong, and that's more useful than if it had worked.**
`throughput(G) = min(G/W_gpu, C/W_cpu)` gives `G* = C · W_gpu / W_cpu`, predicting 0.14 (eval) and
2.32 (ViPE). **Both refuted; both scaled to G=8** — because CPU-s/video *falls* as G rises (eval
1469 → 431) when threads per worker fall 198 → 24. `W_cpu` isn't a constant being divided; it's a
function of thread count. `G*` gets the *ordering* between workloads right and the absolute value
wrong. The refuted prediction stays in deliberately: the Amdahl model in
[§1](#1--the-bottleneck-was-the-host-not-the-gpu-) *does* hold, and the contrast is worth more
than either model alone. → [report §3](report.md#3--gpu-count-sweep-threads-pinned-to-198g--the-two-workloads-scale-differently)

**Topology consequence.** When each container gets its own CPU quota, `N` containers of `G` GPUs
command `N ×` that quota. Normalized to a fixed 8-GPU fleet, **4 × 2 GPU projects to 2.05× the
throughput of 1 × 8** — which explains the launcher the suite shipped with (`for gpu in 0 1`). It
was never that 2 GPUs beat 8; it's that **2 GPUs per 198 CPUs is the point where another GPU is
still worth adding.**

> ⚠️ **Projection, not measurement.** Only `1×8` ran as an actual fleet; the rest is `N ×` a
> per-container wall measured alone on an idle box, assuming own-quota and zero cross-container
> interference. **2.05× is an upper bound** — scaling out converts an idle CPU quota you already
> pay for into throughput; it isn't free speedup.
> → [report §11](report.md#11--scale-out-vs-scale-up-the-arithmetic-that-explains-the-production-launcher)

---

## 3 · ViPE: 4 GPUs really are faster than 8 ⭐

This is where the project started, and it holds up — **but only in the configuration production
actually runs**, which is not what a careful benchmark defaults to. Getting it right meant reading
the shipped launcher instead of trusting the harness. Two differences, both load-bearing:
production **never sets `OMP_NUM_THREADS`**, and it wraps every video in
`ProcessedVideoStream(...).cache()`, decoding every frame into host RAM before inference
(30.5 MB/frame, 2474 MB resident, 1.35 s per clip). That's per-worker *host* work, so it
multiplies by the worker count.

Crossing thread cap × stream mode turns one inverted curve into a controlled experiment:

| G | uncapped+cached **(production)** | uncapped+lazy | capped+cached | capped+lazy |
|---|---|---|---|---|
| 1 | 448.2 s | 435.8 s | 444.5 s | 440.2 s |
| 2 | 245.3 s | 236.9 s | 234.6 s | 235.2 s |
| **4** | **184.6 s** ← best | **177.0 s** ← best | 133.4 s | 127.9 s |
| 8 | 194.2 s | 213.7 s | **79.2 s** ← best | **80.6 s** ← best |

🎯 **Both uncapped arms invert at G=4; neither capped arm does** — including the one still paying
full `.cache()` cost. So **the thread cap is the cause**, and `.cache()` only sets the magnitude.
Mechanism from G=4 → 8: videos per worker drops 2 → **1**, so all eight workers decode
simultaneously — decode/video **3.25 → 9.10 s**, peak load **150 → 288** — while `gpu_busy_s`/video
stays flat (12.94 → 13.74 s). That flat GPU number rules out GPU contention.

✅ **It got stronger on repeat.** n=1 put G=8 at 1.05× (cached) / 1.21× (lazy) slower than G=4;
n=3 puts **both arms at ≈1.25×**, a ≈45 s gap well outside the ≈16–21 s pooled spread.

> ⚠️ **State it in full.** Under the shipped configuration the optimum *is* 4 GPUs. But capping
> the threads moves the optimum to 8 and lands at **80.6 s — 2.29× faster than the 4-GPU optimum
> ever was**. "Use 4 GPUs" answers the question as posed; "cap your thread count" answers the one
> underneath it. **Never quote "8 GPUs is slower than 4" without naming the thread cap.**
> → [report §12](report.md#12--vipe-4-gpus-really-is-faster-than-8--reproducing-the-production-result)

---

## 4 · Ray vs. bash-fork: worth 3% ⚖️

All four launchers, same 48 videos, same 8 GPUs, identical scores:

| launcher | thread config | wall | vs best |
|---|---|---|---|
| bash-fork, capped | `OMP=24` per process | **180.1 s** | 1.00× |
| Ray actor pool, tuned | `num_cpus=24` + `set_num_threads(24)` | 186.3 s | 1.03× |
| Ray actor pool, **default** | Ray's default (1 thread/actor) | 1195.2 s | **6.64×** 😴 |
| bash-fork, **uncapped** | torch's default (128 threads) | 1885.7 s | **10.47×** 🔥 |

The two correct launchers land within **3%**. The misconfigured two fail in *opposite* directions:
Ray starves the workload at 1 thread/actor (94% of the box idle), uncapped fork floods it with
1024 threads. Opposite ends of the same thread U-curve, and neither prints a warning.

So **the launcher is worth 3%, the thread configuration is worth 10×** — not that Ray or fork is
faster. This comparison reversed three times as the thread configuration changed, and at n=1 the
winner sits *inside* the measurement boundary (the two launchers' timers disagree by more than the
gap). What does favour Ray once throughput is off the table: elasticity, fault propagation
(`RayActorError` vs a bare `wait` that exits 0 with a shard missing), and its dynamic queue on
uneven work — on a 4×-variance dataset at G=8, n=3, the queue holds shard imbalance to
**1.16 ± 0.01** vs the static stride's **1.43 ± 0.04**. → [report §5](report.md#5--ray-vs-bash-fork)

---

## 🔁 Reproduce it

```bash
python common/resources.py      # your container's REAL cpu budget (not what nproc says)
python common/make_dataset.py --source vipe_slow/test_video/recam1.mp4 --out data/eval81 --num 48

# the headline (capped scaling), the fix (metrics on GPU), and ViPE as production ships it
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --gpus 1 2 4 8 --arms fork --limit 48 --max-frames 24 --tag main
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --gpus 1 2 4 8 --arms fork --limit 48 --max-frames 24 --device-policy gpu --tag gpufix
python exp_gpu_sweep.py --workload vipe --dataset vipe_slow/test_video --gpus 1 2 4 8 --arms fork-unpinned,fork --limit 8 --stream-mode cached --tag prodcached
```

`--gpus 1 2 4` means GPU *counts*; the launchers' own `--gpus 0,1,2,3` means GPU *IDs*. Each run
writes a `summary.json` with per-sample CPU/GPU attribution, shard imbalance, peak load and merged
scores — that's where every table above is read from. The arm that sets no thread limit is
`fork-unpinned`. ⚠️ **Warm the weight cache first** — a cold run downloads several GB (CLIP, RAFT,
I3D, VGG16, DROID, UniDepth) and that lands inside whatever you're timing.

## 📦 Install

Measured on Python 3.11.15. torch/torchvision need the index matching your driver's CUDA
(`nvidia-smi` will tell you; it's cu128 here).

```bash
conda create -n vipe-new python=3.11 -y && conda activate vipe-new
pip install -U pip uv

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
cd vipe_slow && pip install -e . --no-build-isolation && cd ..
```

## 📁 Repo layout

| path | what's in it |
|---|---|
| `report.md` | 📖 **every** measurement, caveat and reversal + the 9 engineering lessons |
| `exp_gpu_sweep.py` | ★ the experiment: sweep GPU count, hold the CPU quota fixed |
| `common/` | `resources.py` (cgroup-aware CPU budget) · `timing.py` · `make_dataset.py` |
| `movebench/` | 6-metric eval — `metrics/core.py` shared by ALL launchers |
| `vipe_slow/` | vendored NVIDIA ViPE + the four launchers |
| `results/` | every run's shards, logs, loadavg, `summary.json` |

The **evidence is committed**, not just the tables: **58** summaries, **12** sweeps and **321**
per-worker `shard_*.json` with per-sample `cpu_s` / `gpu_busy_s`, plus every 1 Hz `loadavg.txt` —
3.3 MB in the repo out of ~7.5 GB on disk. Every number above recomputes without an 8-GPU box.
*(Intended GitHub repo name: `gpu-scaling-death-point`.)*

**🖥️ Hardware.** 8× NVIDIA H100 80GB HBM3 (NVLink mesh) · 2× AMD EPYC 9534 · Docker, cgroup CFS
quota = **198** CPUs. Three CPU numbers matter and only one is usable: 128 physical (AVX-512
saturation) < **198 quota** (the enforced wall) < 256 host logical — the last is what naive
auto-detection reports and what you should never size from.

---

## 📖 Want the details? → [`report.md`](report.md)

| in `report.md` | what's there |
|---|---|
| [§2](report.md#2--the-two-workloads-are-on-opposite-sides-of-the-cpugpu-balance) · [§3](report.md#3--gpu-count-sweep-threads-pinned-to-198g--the-two-workloads-scale-differently) · [§4](report.md#4--the-unpinned-arm-the-thread-storm-measured) | CPU/GPU attribution, the full scaling sweeps, the thread-storm |
| [§5](report.md#5--ray-vs-bash-fork) · [§6](report.md#6--the-vipe-authors-ray-design-measured) | all four launchers; the ViPE author's own Ray design (tasks vs actors — a real elasticity-vs-warm-models tradeoff, not a defect) |
| [§7](report.md#7--method-notes-and-caveats) · [§8](report.md#8--status) | method notes, every caveat, and **what is still un-run** |
| [§9](report.md#9--device_policygpu--fixing-the-placement-instead-of-scheduling-around-it) · [§10](report.md#10--the-vipe-ray-auto-cliff--a-prediction-then-the-measurement) · [§11](report.md#11--scale-out-vs-scale-up-the-arithmetic-that-explains-the-production-launcher) · [§12](report.md#12--vipe-4-gpus-really-is-faster-than-8--reproducing-the-production-result) | the placement fix + Amdahl model, the ViPE `ray-auto` cliff, scale-out arithmetic, ViPE 4-vs-8 |
| [§13](report.md#13--nine-engineering-lessons) | 🧠 **all nine engineering lessons**, in full |

*Method, briefly: parity is structural (one shared `score_one_video()`), scores bit-identical
across every scheduling change over 10 runs, with two measured exceptions reported where they arise
(`device_policy=gpu` LPIPS ≤6.7e-4 relative; ViPE pose ≤4.5e-4 abs across thread configs, from a
nondeterministic bundle-adjustment reduction). `--max-frames 24` keeps the metric sweeps inside one
session while ViPE uses all 81, so ratios compare across the two but absolute times don't.*
