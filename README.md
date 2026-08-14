# ⚡ More GPUs Is Not Linearly Faster

### Finding the *death-linear point* of two real video pipelines on 8× H100

This repo asks a practical question: when does adding another GPU stop making video
evaluation faster? I ran two real pipelines on 1, 2, 4, and 8 H100s. Both can process
video clips independently, so using every GPU looks like the obvious choice. In practice,
that was not always the fastest option — and the reason was the CPU, not the GPU.

The reason is that a "GPU pipeline" often isn't GPU-bound. One of these two spent 1390×
more CPU time than GPU time and left the GPU 93% idle. Once that's true, what limits you
is CPU cores per worker, and adding GPUs divides them.

The headline number: the evaluation suite went from 1885.7 s to 34.2 s on the same 8
GPUs, a 55.1× speedup, from two changes to how the host was configured. No GPU code was
touched. Adding the other 7 GPUs was worth 3.94× by comparison, which makes the host
fixes roughly 14× more valuable than the hardware.

| 48 videos on 8× H100 | wall | per video | cumulative |
|---|---|---|---|
| `--gpus all`, thread defaults, metrics on CPU | 1885.7 s | 39.3 s | 1.00× |
| **+** threads capped to `quota/G` | 180.1 s | 3.75 s | **10.47×** |
| **+** LPIPS/SSIM/PSNR moved to the GPU | **34.2 s** | **0.71 s** | **55.10×** |

> 💡 The resource that limits an ML workload is often not the one in its name. Measure
> first; choose the GPU count from that profile instead of defaulting to `--gpus all`.

---

## What "capped" and "uncapped" mean

Every table below is labelled *capped* or *uncapped*, and both refer to one question:
did anyone tell each worker how many CPU threads it was allowed to use?

PyTorch and NumPy speed up CPU math by splitting it across threads. If you don't say
otherwise, each worker picks a thread count sized for the whole machine, because it has
no idea the other seven workers exist.

| | what you run | threads per worker | 8 workers ask for | vs the 198-CPU limit |
|---|---|---|---|---|
| **uncapped** (the default) | nothing — you set nothing | 128 (torch's own guess) | **1024** | **5× oversubscribed** 💥 |
| **capped** (the fix) | `export OMP_NUM_THREADS=24` | 24 = `198 ÷ 8` | **192** | fits ✅ |

The same two configurations as code:

```bash
# uncapped: every worker thinks it owns the whole box
for gpu in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$gpu python worker.py & done

# capped: divide the CPU budget by the number of workers, then tell each worker
export OMP_NUM_THREADS=24            # 198 CPUs / 8 workers
for gpu in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$gpu python worker.py & done
```

Oversubscribing doesn't just spread the CPUs thinner. 1024 threads on 198 CPUs do worse
than run 5× slower each: they spend their time being switched on and off the CPU and
fighting over cache instead of computing, and inside a container the cgroup freezes all
of them once they burn through the quota. That's where the 10.47× penalty in
[§2](#2--adding-gpus-helps-only-if-the-host-can-keep-up) comes from.

The trap is that uncapped is what you get by not thinking about it, and it produces
correct numbers. It's just many times slower, with no error and no warning, which is
exactly why it survives in production code. Capping the count is one `export`, and it's
the highest-value line in this repo.

Worth one clarification, since "pin" is the usual word for this and it invites the wrong
guess: this is a limit on the *number* of threads, not CPU affinity. Nothing here binds
a worker to specific cores with `taskset`. In this README, *capped* and *uncapped* always
refer to the number of CPU threads, not CPU affinity.

---

## Results

| # | Finding | § |
|---|---|---|
| 1 | **55.1× faster on the same 8 GPUs** (1885.7 → 34.2 s) from two host-side config changes. The 7 extra GPUs were worth 3.94×. | [§1](#1--the-551-came-from-the-host-not-the-gpus) |
| 2 | A "GPU evaluation pipeline" spent **1390× more CPU time than GPU time**, GPU **7.3% busy**. Its bottleneck was never the accelerator. | [§1](#1--the-551-came-from-the-host-not-the-gpus) |
| 3 | **Fixing *placement* beats tuning the schedule** — moving 3 metrics onto the GPU cut CPU-s/video **57×** and made compute scale at **93%** efficiency. | [§1](#1--the-551-came-from-the-host-not-the-gpus) |
| 4 | Adding GPUs helps **only as far as the host keeps up**: with threads capped, eval scales 3.94× and ViPE 5.46×; uncapped, eval **collapses 10.47×**. | [§2](#2--adding-gpus-helps-only-if-the-host-can-keep-up) |
| 5 | The `min(GPU, CPU)` model I started from **was wrong, and being wrong is the finding** — `W_cpu` is a function of thread count. An Amdahl model *does* hold after the fix. | [§2](#2--adding-gpus-helps-only-if-the-host-can-keep-up) |
| 6 | **For ViPE, 4 GPUs run ~1.25× faster than 8** (n=3) in the config production ships — because at G=8 every worker decodes at once, not anything about GPUs. | [§3](#3--for-vipe-4-gpus-really-are-faster-than-8) |
| 7 | **The launcher choice is worth ~3%; the thread configuration is worth 10×.** The two correct launchers land within 3% at n=1; the two misconfigured ones are 6.6× and 10.5× slower in *opposite* directions. | [§4](#4--ray-vs-bash-fork) |

Every number in this README comes out of the committed JSON under `results/` — 246
files, about 1.2 MB — so the tables recompute without an 8-GPU box. Each run's
`summary.json` carries the wall clock, per-sample CPU and GPU attribution, shard
imbalance and peak load that the rows below are read from.

Four of the claims were small enough that run-to-run noise could have explained them, so
I re-measured those three times each; the results live in `results/*_rep/`. The ViPE
4-beats-8 result got stronger, the load-balancing claim narrowed to the one metric that
survived, and the launcher comparison is discussed in [§4](#4--ray-vs-bash-fork).

Two caveats belong up front. The 1885.7 s baseline is a real configuration — torch's
thread default plus the placement the suite ships with — but it's also the worst of the
four launchers in [§4](#4--ray-vs-bash-fork), so 55.1× is the distance between the naive
and the tuned end of a spectrum rather than headroom every deployment is sitting on. And
while capping threads leaves scores untouched, moving the metrics to the GPU shifts LPIPS
by up to 6.7e-4 relative because cuDNN picks different convolution algorithms; every
aggregate still matches to 4 decimals.

---

## Where this comes from: the UniCaMo pipeline

The work is inspired by **[UniCaMo](https://phongnhhn.info/Unicamo/)**, *Unifying Camera
and Motion Control for Video Generation*. UniCaMo generates video under two controls at
once: what the subject does, and where the camera goes. Controlling both means measuring
both, which is where these two workloads come from.

```
  raw training       ┌─ PREPROCESSING ──────────────────────────────┐
  video ────────────▶│  ViPE: per-clip camera pose + metric depth,   │──┐
                     │  precomputed for every clip → the conditioning│  │
                     └───────────────────────────────────────────────┘  ▼
                                                              train UniCaMo
                                        generated video                 │
                     ┌─ EVALUATION — two axes ──────────────────────────▼──┐
                     │  motion + quality │ movebench: CLIP-I · EPE · LPIPS  │
                     │                   │ · SSIM · PSNR · FVD              │
                     │  camera control   │ ViPE again: pose of generated     │
                     │                   │ video vs. the requested traj.     │
                     └──────────────────────────────────────────────────────┘
```

| folder | role in UniCaMo | what it computes |
|---|---|---|
| **`movebench/`** | evaluation — motion + quality | CLIP-I · optical-flow EPE (RAFT) · LPIPS · SSIM · PSNR · FVD (I3D) |
| **`vipe_slow/`** | **preprocessing** (labels for every training clip) **and** evaluation — camera axis | NVIDIA ViPE — camera pose + metric depth, SLAM with learned priors |

ViPE sits on the critical path twice: once over the training corpus to produce the
depth and pose labels the model conditions on, and again over generated video to check
whether the camera went where it was told. Its throughput gates dataset prep as well as
every evaluation round, which is why
[§3](#3--for-vipe-4-gpus-really-are-faster-than-8) gets a section of its own instead of
a footnote.

Each pipeline runs under four launchers doing identical arithmetic and differing only in
orchestration, so any difference in wall clock is attributable to scheduling. Parity is
structural rather than asserted: every launcher calls the same `score_one_video()` /
`pipeline.run()`. If two of them disagree, orchestration is what broke.

---

## 1 · The 55.1× came from the host, not the GPUs

Same 48 videos, same 8 H100s, same arithmetic:

| configuration | wall at G=8 | per video | cumulative |
|---|---|---|---|
| `--gpus all`, thread defaults, metrics on CPU — *the obvious way* | 1885.7 s | 39.3 s | 1.00× |
| **+** cap each worker's threads to `quota/G` ([§2](#2--adding-gpus-helps-only-if-the-host-can-keep-up)) | 180.1 s | 3.75 s | **10.47×** |
| **+** move LPIPS/SSIM/PSNR onto the accelerator | **34.2 s** | **0.71 s** | **55.10×** |

The two factors multiply almost exactly — 10.47 × 5.26 = 55.10, which is what the
measured endpoints give — and that's the evidence they're fixing independent problems.
One is CFS throttling from oversubscribed threads; the other is convolution running on
the host.

Here's why the pipeline was host-bound to begin with, measured per video at G=1 with the
full 198-thread budget:

| | movebench eval (24 fr) | ViPE pose+depth (81 fr) |
|---|---|---|
| **CPU-seconds consumed** | **1469 s** | **1056 s** |
| **GPU-busy-seconds** | **1.06 s** | **12.38 s** |
| GPU utilization | **7.3%** | **23.3%** |
| **CPU : GPU time ratio** | **1390 : 1** | **85 : 1** |
| where the hot loop lives | host (VGG / SSIM / LPIPS) | device (SLAM + bundle adjustment) |

The cause is placement, not volume. LPIPS-VGG, SSIM and PSNR are all handed CPU tensors
and never moved to the accelerator, and LPIPS — a VGG16 forward pass per frame —
dominates. Two pipelines over the same videos on the same hardware end up on opposite
sides of the CPU/GPU balance, a 16× difference in ratio.

One note on how that was measured, since it's the number everything else leans on.
Wrapping a sample in CUDA events reports `gpu_s ≈ wall_s`, because the stream is resident
the whole time. Integrated NVML utilization says the GPU actually worked 7% of it.
Event-elapsed time would have made a host-bound workload look GPU-bound, so every "it's
the host" claim here rests on `gpu_busy_s` instead.

Which makes the real repair placement rather than scheduling (`--device-policy gpu`):

| G | original placement | **device_policy=gpu** | speedup | CPU-s/video (orig → fix) |
|---|---|---|---|---|
| 1 | 708.9 s | **151.0 s** | 4.70× | 1469 → 25.7 (**57×**) |
| 2 | 351.9 s | **81.1 s** | 4.34× | 832 → 15.0 (56×) |
| 4 | 232.9 s | **49.0 s** | 4.75× | 557 → 9.8 (57×) |
| 8 | 180.1 s | **34.2 s** | **5.26×** | 431 → 7.5 (**57×**) |

That's 57× less CPU time per video at every G, with GPU occupancy going from 7.3% to
around 34%. The more interesting change is in the shape of the curve. Compute efficiency
at G=8 goes from 53%, where workers were fighting over the CPU quota, to 93%, and what
remains of the sublinearity isn't contention any more — it's a fixed serial cost of about
14.0 s for model load and merge. A one-parameter Amdahl model, `wall(G) = S +
compute(1)/G` with `S = 14.0 s`, predicts the whole curve to within 1–9%:

```
G=1: 152.2 predicted vs 151.0 measured   G=4: 48.6 vs 49.0
G=2:  83.1              vs  81.1         G=8: 31.3 vs 34.2
```

It also predicts something falsifiable: with `S` fixed, the ceiling is 10.77× no matter
how many GPUs you add. Past this fix, the thing to optimize is the serial cost, not the
GPU count.

> ⚠️ This is the one change in the repo that isn't bit-identical. Moving LPIPS to the GPU
> shifts its result by up to **6.7e-4 relative** (48 videos, mean 3.4e-4), because cuDNN
> picks different convolution algorithms. CLIP and EPE come out exactly equal, SSIM and
> PSNR agree to ~1e-8, and every aggregate matches to 4 decimals. For a ranking metric
> 6.7e-4 is harmless, but "we moved it to the GPU and nothing changed" would be false.

---

## 2 · Adding GPUs helps only if the host can keep up

The *death-linear point* is the worker count past which adding a GPU stops buying
proportional throughput, because a fixed per-item host cost has become the binding
constraint instead of the accelerator. It's a property of the workload, not of the
hardware.

With threads capped to `198/G` and one process per GPU — the fixed configuration, in the
sense of [what "capped" means](#what-capped-and-uncapped-mean):

| G | threads | **eval** wall | speedup | eff. | **ViPE** wall | speedup | eff. |
|---|---|---|---|---|---|---|---|
| 1 | 198 | 708.9 s | 1.00× | 100% | 440.2 s | 1.00× | 100% |
| 2 | 99 | 351.9 s | 2.01× | 101% | 235.2 s | 1.87× | 94% |
| 4 | 49 | 232.9 s | 3.04× | 76% | 127.9 s | 3.44× | 86% |
| 8 | 24 | 180.1 s | **3.94×** | **49%** | 80.6 s | **5.46×** | **68%** |

Once the threads are capped both curves are monotonic, and they lose efficiency at about
the rate the CPU:GPU ratio predicts. ViPE has 12.4 GPU-s per video against eval's 1.06,
so it has far more device work to hide the host stage behind: 68% efficiency at G=8
versus 49%.

Leave the threads at torch's default, though, and the eval curve falls apart. This is
what you get by setting nothing:

| G | total threads | uncapped | capped | penalty | peak loadavg (quota **198**) |
|---|---|---|---|---|---|
| 1 | 128 | 712.9 s | 708.9 s | 1.01× | 126 |
| 2 | 256 | 436.5 s | 351.9 s | 1.24× | **202** |
| 4 | 512 | 1574.2 s | 232.9 s | **6.76×** | **377** |
| 8 | 1024 | **1885.7 s** | 180.1 s | **10.47×** | **502** |

This is the genuine "more GPUs made it slower" curve: throughput peaks at G=2 and
collapses after it. What identifies the cause is that `gpu_busy_s` stays around
1 s/video in every cell, capped or not, so the GPU is doing identical work and only host
time inflates. Peak load crosses 198 exactly at G=2, the first configuration where
`G × 128` exceeds the quota, which is the signature of CFS throttling rather than GPU
contention. Scores came out identical in every cell, so this costs throughput and not
correctness — which is why it survives in production with nobody noticing.

The model I started from was wrong, and that turned out to be more useful than if it had
worked. With `C` CPUs fixed and `G` workers each getting `C/G` threads:

```
throughput(G) = min( G/W_gpu , C/W_cpu )       G* = C · W_gpu / W_cpu
```

`W_cpu` and `W_gpu` are measured at G=1, so `G*` is predicted and then checked rather
than fitted. It predicts 0.14 for eval and 2.32 for ViPE, and both are refuted — both
scaled all the way to G=8. The CPU-s/video column shows why: it falls as G rises (eval
1469 → 431) because threads per worker fall from 198 to 24. `W_cpu` isn't a constant
being divided, it's a function of the thread count, since a 198-thread convolution
spends most of those threads synchronizing rather than doing arithmetic. So `G*` gets the
ordering between the two workloads right and the absolute value wrong. I've kept the
refuted prediction here deliberately, because the Amdahl model in
[§1](#1--the-551-came-from-the-host-not-the-gpus) is the one that holds, and the contrast
between the two is worth more than either on its own.

There's a topology consequence too. In the real deployment each container gets its own
`C=198` quota, so `N` containers of `G` GPUs command `N × 198` CPUs. Normalizing to a
fixed 8-GPU fleet, using the per-container walls already measured:

| topology | per-container wall | **projected** fleet throughput | vs 1×8 | CPU quota |
|---|---|---|---|---|
| 8 × 1 GPU | 708.9 s | 0.5417 v/s | 2.03× | 1584 |
| **4 × 2 GPU** | 351.9 s | **0.5456 v/s** | **2.05×** | 792 |
| 2 × 4 GPU | 232.9 s | 0.4122 v/s | 1.55× | 396 |
| 1 × 8 GPU | 180.1 s | 0.2665 v/s | 1.00× | 198 |

This explains the launcher the suite shipped with. `for gpu in 0 1`, two GPUs per
container, was the right call: at G=2 each worker still gets 99 threads and per-GPU
throughput is undiminished. It was never that 2 GPUs beat 8. It was that 2 GPUs per 198
CPUs is the point where another GPU is still worth adding.

> ⚠️ This table is a projection, not a measurement. Only the `1×8` row was run as an
> actual fleet; the rest is `N ×` a per-container wall measured alone on an idle box. It
> assumes each container gets its own full 198-CPU quota, true for the multi-node RunAI
> topology but false if one box is subdivided, and it assumes zero cross-container
> interference, which is untested and harder to defend — concurrent containers contend
> for memory bandwidth, L3, PCIe and NVLink. **2.05× is an upper bound.** The honest
> framing is that scaling out converts an idle CPU quota you already pay for into
> throughput, not that it's free speedup.

---

## 3 · For ViPE, 4 GPUs really are faster than 8

This is where the project started, and it holds up under measurement — but only in the
configuration production actually runs, which isn't the one a careful benchmark defaults
to. Getting it right meant reading the shipped launcher
(`eval_cam/vipe/vipe_unicamo.sh` → `run_multinode.py`) instead of trusting my own
harness. It differed in two ways, and both of them mattered:

| | production | a "clean" benchmark |
|---|---|---|
| `OMP_NUM_THREADS` | **never set** — uncapped | capped to `198/G` |
| stream construction | `ProcessedVideoStream(...).cache()` | `StreamList.make()` — lazy |

`.cache()` decodes every frame into host RAM before inference: 30.5 MB per frame,
2474 MB resident, 1.35 s per clip, measured. That's per-worker host work, so it
multiplies by the worker count.

Crossing the thread cap with the stream mode turns one inverted curve into a controlled
experiment, and it moves the blame off `.cache()` entirely. Reading the columns:
*uncapped* means nobody set the thread count and *capped* means it was set to `198/G`;
*cached* means the whole clip is decoded into RAM first and *lazy* means frames are
decoded as they're needed. Only the leftmost column is what production runs.

| G | uncapped+cached **(production)** | uncapped+lazy | capped+cached | capped+lazy |
|---|---|---|---|---|
| 1 | 448.2 s | 435.8 s | 444.5 s | 440.2 s |
| 2 | 245.3 s | 236.9 s | 234.6 s | 235.2 s |
| **4** | **184.6 s** ← best | **177.0 s** ← best | 133.4 s | 127.9 s |
| 8 | 194.2 s | 213.7 s | **79.2 s** ← best | **80.6 s** ← best |

Both uncapped arms invert at G=4 and neither capped arm does, including the one still
paying the full `.cache()` cost. So the thread cap is the cause, and `.cache()` only
changes the magnitude. The mechanism going from G=4 to G=8: videos per worker drops from
2 to 1, so all eight workers decode at the same time. Decode time per video goes from
3.25 s to 9.10 s and peak load from 150 to 288 against a 198 quota, while `gpu_busy_s`
per video stays flat at 12.94 → 13.74 s. That flat GPU number is what rules out GPU
contention.

> This one got stronger when I repeated it. The single-shot table puts G=8 at 1.05×
> (cached) and 1.21× (lazy) slower than G=4. Re-running G=4 and G=8 three times each
> (`results/vipe_prodcached_rep/`, `results/vipe_unpinned_rep/`) put both arms at
> **G=8 ≈ 1.25× slower than G=4**:
> 223.9 ± 14.0 s vs 179.0 ± 8.3 s cached, 220.0 ± 17.8 s vs 176.6 ± 10.8 s lazy. The
> ≈45 s gap sits well outside the pooled run-to-run spread of ≈16–21 s, so it's a real
> effect and not the noise it might have been at n=1.

> The full statement matters more than the headline. Under the shipped configuration the
> optimum is 4 GPUs, reproduced at n=3. But capping the threads moves the optimum to 8
> and lands at **80.6 s, 2.29× faster than the 4-GPU optimum ever was**. "Use 4 GPUs" is
> the right answer to the question as posed; "cap your thread count" is the better answer
> to the question underneath it. Never quote "8 GPUs is slower than 4" without naming the
> thread cap.

---

## 4 · Ray vs. bash-fork

All four launchers, same 48 videos, same 8 GPUs:

| launcher | thread config | wall | vs best |
|---|---|---|---|
| bash-fork, capped | `OMP=24` per process | **180.1 s** | 1.00× |
| Ray actor pool, tuned | `num_cpus=24` + `set_num_threads(24)` | 186.3 s | 1.03× |
| Ray actor pool, **default** | Ray's own default (1 thread/actor) | 1195.2 s | **6.64×** |
| bash-fork, **uncapped** | torch's default (128 threads) | 1885.7 s | **10.47×** |

The two correctly configured launchers land within 3% of each other. The two
misconfigured ones are 6.6× and 10.5× slower, and they fail in opposite directions: Ray
starves the workload with 1 thread per actor, leaving 94% of the box idle, while uncapped
bash-fork floods it with 1024 threads against a 198 quota at load 502. Opposite ends of
the same thread U-curve. Neither prints a warning, and all four produce identical scores.

The conclusion I'd stand behind is that the launcher choice is worth 3% and the thread
configuration is worth 10× — not that Ray is faster, or that bash-fork is. This
comparison reversed three times across sessions and every reversal came down to the
thread cap rather than the scheduler. At n=1 the winner sits inside the measurement
boundary: by the sweep driver's wall fork wins by 3.3%, and by each launcher's own
internal wall, which excludes about 12 s of `ray.init()`, Ray wins by 3.5%.

Repeating that cell three times (`results/movebench_launcher_rep/`) complicates the
picture rather than settling it. Ray came out at **187.7 ± 3.6 s** against fork's
**235.7 ± 21.9 s** — a 26% gap, which is larger than the pooled spread, so it isn't
simply noise. But the whole difference is fork's variance: its three runs were 219.3,
227.2 and 260.5 s while Ray's were 184.2, 187.4 and 191.5 s. The box was contended
during that sweep, and fork's static stride has no way to absorb a slow worker, while
Ray's queue does. So the defensible reading is that the two are level when the machine
is quiet and Ray degrades more gracefully when it isn't — not that Ray is 26% faster.
Pinning that down properly needs a quiet box; that experiment remains to be run.

What does favour Ray, once throughput is off the table: elasticity, fault propagation
(`RayActorError` versus a bare `wait` that exits 0 with a shard missing), the fact that
the shard-count bug is structurally impossible, and its dynamic queue when the work is
uneven. On a 4×-variance dataset at G=8 with both capped, measured three times, the
dynamic queue holds shard imbalance to **1.16 ± 0.01** against the static stride's
**1.43 ± 0.04**, cleanly separated. The wall-clock version of that was ~6% at n=1 but
washed out to a tie at n=3 on a contended box, so the claim I'll make is about load
balance, which is what the queue actually acts on. Even getting that far needed a second
dataset: on the equal-cost set a static shard is already balanced, so any "win" would
have been noise.

---

## 🧭 The four lessons that cost the most time

1. **In a container, `nproc` lies.** The CFS quota here is 198, not the 256 `nproc`
   reports. cgroup throttling doesn't shrink the affinity mask — it lets you spawn 256
   threads and then freezes them all once the quota burns. There's no error, only
   slowness. `common/resources.py` is the one place this gets decided.
2. **`torchrun` silently sets `OMP_NUM_THREADS=1`; plain `python` doesn't.** One
   informational line, and then every CPU-side hot loop is single-threaded. It also has
   to be set before `import torch`, since torch fixes its pool size at extension load
   and a later `os.environ[...]` is a silent no-op.
3. **The thread U-curve has two walls, and oversubscription is the worse one.** The floor
   sits where `total_threads == physical cores`; 512 threads came out slower than 64.
   Naive Ray sits on the left wall, an uncapped `torchrun` on the right.
4. **Put the fleet's shared cache on a node-local disk.** A model-hub download lock under
   an NFS `$HOME` mounted in 12 containers turned a "per-machine" lock into a fleet-global
   one, and 96 ranks serialized to *not* re-download weights they already had. Load phase
   went from 9 m 06 s to about 44 s.

---

## 🚀 Reproduce it

### 🛠️ Configure for your hardware

Set the GPU count to the accelerators you want to use, and divide the CPU budget
available to the job across those GPU workers. The examples below use four GPUs
and 32 CPUs, so each worker receives eight CPU threads. Replace these values with
the resources allocated to your own machine or container.

```bash
# Inspect the CPU budget actually available to this job (respects cgroup limits).
python common/resources.py

# Sweep a chosen number of GPUs.  Use only values that exist on your machine.
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 \
  --gpus 1 2 4 --arms fork --limit 48 --max-frames 24 --tag my-hardware

# Run specific GPU IDs and explicitly set CPU threads per GPU worker.
# For 32 available CPUs and 4 GPU workers: THREADS=32 / 4 = 8.
THREADS=8 movebench/run_fork.sh --dataset data/eval81 --gpus 0,1,2,3
```

`--gpus` in `exp_gpu_sweep.py` is a list of **GPU counts** to test; `--gpus`
in `movebench/run_fork.sh` is a comma-separated list of **GPU IDs**. When
`THREADS` is not set, `run_fork.sh` automatically uses
`floor(available CPUs / selected GPUs)`, using the container CPU quota when one
is present. Set `THREADS` only when you deliberately want a smaller per-worker
CPU allocation. Avoid setting it to the full CPU count for every GPU, as that
oversubscribes the host.

> ⚠️ Start with one GPU, then test 2, 4, and so on. More GPUs only help while the
> CPU can feed every worker.

```bash
python common/resources.py      # this container's REAL cpu budget (198, not nproc's 256)
python common/make_dataset.py --source vipe_slow/test_video/recam1.mp4 --out data/eval81 --num 48

# the headline (capped scaling), the fix (metrics on GPU), and ViPE as production ships it
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --gpus 1 2 4 8 --arms fork --limit 48 --max-frames 24 --tag main
python exp_gpu_sweep.py --workload movebench --dataset data/eval81 --gpus 1 2 4 8 --arms fork --limit 48 --max-frames 24 --device-policy gpu --tag gpufix
python exp_gpu_sweep.py --workload vipe --dataset vipe_slow/test_video --gpus 1 2 4 8 --arms fork-unpinned,fork --limit 8 --stream-mode cached --tag prodcached
```

Each run writes a `summary.json` with per-sample CPU/GPU attribution, shard imbalance,
peak load and merged scores, which is where every table above is read from. The arm that
sets no thread limit is named `fork-unpinned` on the command line and
`results/*_unpinned/` on disk; that's the uncapped configuration described above. Warm
the weight cache before timing anything — a cold first run downloads several GB (CLIP,
RAFT, I3D, VGG16, DROID, UniDepth) and that lands inside whatever you're measuring.

## Install

Measured on Python 3.11.15. torch and torchvision need the index matching your driver's
CUDA, which `nvidia-smi` will tell you; it's cu128 here.

```bash
conda create -n vipe-new python=3.11 -y && conda activate vipe-new
pip install -U pip uv

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
cd vipe_slow && pip install -e . --no-build-isolation && cd ..
```

`requirements.txt` pins every dependency to the version these numbers were measured on.
torch and torchvision go through plain `pip` first, since the CUDA index isn't a normal
PyPI wheel, and then `uv` resolves the rest in seconds instead of minutes.

---

## Repo layout

```
speed_up_video_evaluation/           (push as: gpu-scaling-death-point)
├── README.md            the findings (this file)          ├── common/     resources.py (the ONE place 198 lives)
├── exp_gpu_sweep.py   ★ the experiment: sweep G, fix C    ├── movebench/  6-metric eval (core.py shared by ALL launchers)
└── requirements.txt     pinned to the measured versions   ├── vipe_slow/  vendored NVIDIA ViPE + the 4 launchers
                                                           ├── data/       datasets (gitignored, rebuild in ~2 min)
                                                           └── results/    every run's shards, logs, loadavg, summary.json
```

The evidence is committed, not just the tables. `results/` is 555 MB on disk, almost all
of it ViPE depth and rgb output that's excluded, but the ~1.2 MB the write-up rests on is
in the repo: every `summary.json` and `sweep.json`, every per-worker `shard_*.json` with
per-sample `cpu_s` and `gpu_busy_s`, and every 1 Hz `loadavg.txt`. Every number above
recomputes without an 8-GPU box.

**Method, briefly.** Parity is structural — one shared `score_one_video()` — and scores
are bit-identical across every scheduling change over 10 runs, with two measured
exceptions reported where they arise (`device_policy=gpu` LPIPS ≤6.7e-4 relative, and
ViPE pose ≤4.5e-4 abs across thread configs, from a nondeterministic bundle-adjustment
reduction). The straggler is the number, so wall equals the slowest shard, which also
lets wall decompose into serial and parallel cost. `--max-frames 24` keeps the metric
sweeps inside one session while ViPE uses all 81, so ratios compare across the two but
absolute times don't. Where results contradict each other, both are reported along with
the variable that changed — there are three such reversals, and they're the most useful
content here.

**Hardware.** 8× NVIDIA H100 80GB HBM3 (full NVLink mesh) · 2× AMD EPYC 9534 · Docker,
cgroup CFS quota = 198 CPUs. Three CPU numbers matter and only one of them is usable:
128 physical (AVX-512 saturation) < 198 quota (the enforced wall) < 256 host logical,
which is what naive auto-detection reports and what you should never size from.

*The ViPE author's Ray task design highlights a real tradeoff between elasticity and
warm models; the actor launcher in this repo makes that tradeoff explicit.*
