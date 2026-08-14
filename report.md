# Report — status, measurements, and the results that reversed

**Box:** 8× H100 80GB HBM3 (NVLink mesh) · 2× AMD EPYC 9534 · Docker, **cgroup CFS
quota = 198 CPUs** (`nproc` reports 256) · Python 3.11, torch 2.11.0+cu128, ray
2.55.1, env `vipe-new`.

**Dataset:** `data/eval81` — 48 raw/generated video pairs, **81 frames each**,
832×480, built deterministically from one source clip by
`common/make_dataset.py` (seeded degradation sweep, strength 0.15→0.85). Metric
runs use `--max-frames 24` to keep a 4-cell sweep inside a session; ViPE runs use
all 81.

Where a result contradicts an earlier one, both are reported with the variable that
changed — that reconciliation is §3 and §4, and it is the most useful part of the
document.

This is the detailed companion to the project [README](README.md). It keeps the
full measurements, assumptions, and caveats behind the concise recommendations
presented there.

**Measurement rounds.** §2–§7 were measured 2026-08-09. §9–§11 were added
2026-08-10 and close three of the five gaps §8 had listed as un-run: the
`device_policy=gpu` arm (§9), the ViPE `ray-auto` cliff (§10), and the scale-out
topology arithmetic (§11). §9 and §11 change what the repo's headline
recommendation is, so they are not appendices — read them.

## Navigation — strongest results first

The sections are in the order they were measured, so the newest and strongest are
near the bottom. If you are reading for the result rather than the chronology:

| Read | Section | The one number |
|---|---|---|
| **1st** | **§9 — placement fix + the 55.1× composition** | 180.1 → **34.2 s (5.26×)**; composed **55.10×** on fixed 8 GPUs |
| **2nd** | **§12 — ViPE 4 GPUs beats 8** (the 2×2 that finds the cause) | G=4 optimum in the shipped config; pinning → **2.29×** past it |
| **3rd** | **§11 — scale-out topology** (projection, see caveats) | 4×2 GPU = **2.05×** the throughput of 1×8 (upper bound) |
| **4th** | **§4 — the thread-storm** | unpinned G=8: **10.47×** penalty, load 502 vs 198 |
| then | §2 CPU:GPU balance · §3 pinned scaling · §5 Ray-vs-fork · §10 ViPE cliff | — |
| last | **§13 — nine engineering lessons** (the reusable part) | `nproc` lies: **198**, not 256 |

Every number in these sections is read from committed JSON under `results/`. §8 is the
honest list of what was *not* run.

[`README.md`](README.md) is the short version — the seven findings with a figure each,
and the four lessons that cost the most time. This document is the long one: every
measurement, every caveat, and the reversals.

> **One naming difference.** This report says **pinned** / **unpinned** threads; the
> README and the figures say **capped** / **uncapped**. They are the same two
> configurations — *was each worker told how many CPU threads it may use?* (`pinned`
> = `OMP_NUM_THREADS=quota/G`, `unpinned` = nothing set, so torch guesses 128 per
> worker). Neither term means `taskset`/CPU affinity; no run here binds a worker to
> specific cores. The on-disk arm and run names (`fork-unpinned`,
> `results/*_unpinned/`) keep the `unpinned` spelling.

---

## 1. What was built

| component | purpose |
|---|---|
| `common/resources.py` | cgroup-aware CPU budget. The single place `198` is derived; every launcher calls it. Handles cgroup v1/v2, falls back to the affinity mask. |
| `common/timing.py` | per-sample **CPU-seconds** and **GPU-busy-seconds**. GPU busy time comes from a background NVML sampler at 50 Hz, *not* from CUDA-event elapsed time — the distinction is §2. |
| `common/make_dataset.py` | builds N standard 81-frame raw/gen pairs from one clip, with `--jitter-frames` to create duration variance on demand. |
| `movebench/metrics/core.py` | the six metrics (CLIP-I, EPE/RAFT, LPIPS, SSIM, PSNR, FVD/I3D) and `score_one_video()` — called by **every** launcher, so parity is structural. |
| `movebench/{run_fork.sh, run_ray.py, worker.py, merge.py}` | bash-fork baseline · Ray actor pool · the shared worker · tree-reduce merge. |
| `vipe_slow/{run_infer.py, run_fork.sh, run_ray.py, run_ray_author.py, summarize.py}` | ViPE under four launchers, including the ViPE author's own Ray design. |
| `exp_gpu_sweep.py` | the experiment: sweep G, hold C fixed, predict `G*` from G=1 and check it. |

`vipe_slow/` was reduced from 1.9 GB of demo assets and prior outputs to the
checkout plus launchers; everything removed is in `vipe_slow/_archive/`, not
deleted.

---

## 2. The two workloads are on opposite sides of the CPU/GPU balance

Per video, measured at G=1 with the full 198-thread budget. Per-video absolutes
are not comparable across the two columns (eval uses 24 frames to fit the sweep
in a session, ViPE uses all 81) — **the ratios and utilizations are the point**:

| | movebench eval (24 frames) | ViPE pose+depth (81 frames) |
|---|---|---|
| wall | 14.6 s | 55.0 s |
| **CPU-seconds consumed** | **1469 s** | **1056 s** |
| **GPU-busy-seconds** | **1.06 s** | **12.38 s** |
| GPU utilization | **7.3%** | **23.3%** |
| CPU cores kept busy | **101** | **19.9** |
| CPU:GPU time ratio | **1390 : 1** | **85 : 1** |

Both are "GPU pipelines" by name. The evaluation suite spends **1390× more CPU
time than GPU time**, because LPIPS-VGG, SSIM and PSNR are handed CPU tensors and
never moved to the device — LPIPS (a VGG16 forward per frame) dominates. ViPE's
hot loop is SLAM + bundle adjustment, which genuinely lives on the GPU — but even
it spends 85× more time on the host than the device, because video decode, factor-
graph bookkeeping, and I/O are all CPU work. The two ratios differ by **16×**, and
that gap is what makes the two workloads scale differently in §3.

**Why `gpu_busy_s` and not CUDA-event elapsed time.** An event pair around the
sample reported `gpu_s ≈ wall_s` for the eval suite — the stream *is* resident for
the whole sample. Integrated NVML utilization says the GPU was actually working
**7%** of that time. Using event-elapsed would have made a host-bound workload
look GPU-bound; the two numbers are collected and reported separately for exactly
this reason.

---

## 3. GPU-count sweep, threads pinned to `198/G` — the two workloads scale differently

Both swept identically: bash-fork, `OMP_NUM_THREADS = 198/G`, one process per GPU.

**movebench eval** (48 videos, `--max-frames 24`):

| G | threads/worker | wall | speedup | efficiency | CPU-s/video | GPU-s/video | peak loadavg |
|---|---|---|---|---|---|---|---|
| 1 | 198 | 708.9 s | 1.00× | 100% | 1469 | 1.06 | 147 |
| 2 | 99 | 351.9 s | **2.01×** | 101% | 832 | 1.02 | 144 |
| 4 | 49 | 232.9 s | **3.04×** | 76% | 557 | 1.04 | 137 |
| 8 | 24 | 180.1 s | **3.93×** | **49%** | 431 | 1.04 | 141 |

**ViPE pose+depth** (8 videos, all 81 frames):

> ⚠️ **This is the pinned + lazy-stream arm, which is NOT how production runs ViPE.**
> The production launcher sets no `OMP_NUM_THREADS` and wraps every video in
> `.cache()`. In that configuration **G=4 beats G=8** — see **§12**, which
> reproduces the project's original observation. The table below is a *control*, not
> the headline ViPE result.

| G | threads/worker | wall | speedup | efficiency | CPU-s/video | GPU-s/video | GPU occupancy |
|---|---|---|---|---|---|---|---|
| 1 | 198 | 440.3 s | 1.00× | 100% | 1056 | 12.38 | 0.233 |
| 2 | 99 | 235.2 s | **1.87×** | 94% | 543 | 12.52 | 0.231 |
| 4 | 49 | 127.9 s | **3.44×** | 86% | 298 | 12.45 | 0.225 |
| 8 | 24 | 80.6 s | **5.45×** | **68%** | 182 | 12.52 | 0.217 |

**The two curves separate exactly as the CPU:GPU ratio predicts.** ViPE reaches
5.45× on 8 GPUs (68% efficiency) where the eval suite manages 3.93× (49%). Same
box, same launcher, same fixed 198-CPU budget; the difference is that ViPE spends
**12.5 GPU-seconds per video against the eval suite's 1.04**, so there is far more
device work to hide the host stage behind. `gpu_s/video` is flat across every cell
of both sweeps — the GPU does identical work at every G, which is what makes the
scaling deficit attributable to the host.

**All movebench cells produced identical scores** (clip=0.9421, epe=2.1085,
lpips=0.2567, ssim=0.9197, psnr=27.9234) — parity across the whole sweep.

### The contradiction, stated plainly

The model in `exp_gpu_sweep.py` predicts, from each workload's G=1 measurement,

```
movebench: G* = 198 × 1.057 / 1468.8 = 0.14   -> flat from G=1
ViPE:      G* = 198 × 12.38 / 1056.3 = 2.32   -> flat past G~2
```

Both predict saturation far earlier than observed: movebench climbed to G=8, and
ViPE climbed to G=8 with its *largest* marginal gain between 4 and 8. The ordering
`G*(ViPE) > G*(movebench)` is correct — 2.32 vs 0.14, a 17× gap, matching the
direction of the measured efficiency gap — but both absolute values are wrong.

The `CPU-s/video` column resolves it: that number **falls with G in both
workloads** (movebench 1469 → 431, ViPE 1056 → 182) as threads per worker fall
198 → 24. `W_cpu` is not a constant being divided among workers — it is *itself a
function of the thread count*, because a 198-thread convolution spends most of
those threads on synchronization and cache contention rather than arithmetic. The
`min(G/W_gpu, C/W_cpu)` model assumes a fixed `W_cpu` and therefore mispredicts
whenever the CPU stage is thread-inefficient, which is the common case.

So the defensible claims are narrower than "more GPUs is slower", and more useful:

1. **Scaling is real but sub-linear, and *how* sub-linear is set by the CPU:GPU
   ratio** — 68% efficiency at 85:1 (ViPE), 49% at 1390:1 (eval). The deficit is
   host contention, not GPU supply.
2. **The GPU count and the per-worker thread count are one coupled decision.** A
   launcher that sets only `--gpus` is tuning half a parameter. The two must be
   swept together, which is what this harness does.
3. **`G*` predicts the right ordering between workloads and the wrong absolute
   value.** Useful for ranking two pipelines, not for choosing G. Reported as a
   refuted prediction rather than quietly dropped.

The earlier "2 GPUs beats 8" result is not overturned — it was measured *without* a
thread pin, and §4 shows that the unpinned curve genuinely is non-monotonic.

---

## 4. The unpinned arm: the thread-storm, measured

Identical work, `THREADS=0`, so each worker takes torch's default (**128
threads**, half the host's 256 — note it does not even know about the 198 quota):

| G | total threads requested | wall, unpinned | wall, pinned | penalty | peak loadavg (quota 198) | GPU util |
|---|---|---|---|---|---|---|
| 1 | 128 | 712.9 s | 708.9 s | **1.01×** | 126 | 7.6% |
| 2 | 256 | 436.5 s | 351.9 s | **1.24×** | **202** | ~7% |
| 4 | 512 | 1574.2 s | 232.9 s | **6.76×** | **377** | 0.8% |
| 8 | 1024 | **1885.7 s** | 180.1 s | **10.47×** | **502** | **0.33%** |

Per-video steady state at G=8: **337 s/video unpinned vs 14.6 s pinned = 23×**.
Scores were identical in every cell, pinned or not — this costs throughput, not
correctness, which is exactly why it survives in production code.

**The mechanism, and the control that pins it down.** At G=1 unpinned is *free*
(713 vs 709 s) — 128 threads fit inside the 198 quota, so peak load sits at 126.
The penalty appears the instant `G × 128` crosses the quota: at **G=2 the peak load
is 202** — the first configuration that exceeds 198 — and it climbs to 377 (G=4)
and **502 (G=8)**, i.e. 2.5× oversubscribed. Once the cgroup has spent its quota in
a 100 ms period it freezes every thread until the next one. The control is
`gpu_busy_s`: it stays at **~1.02–1.24 s/video in every cell**, pinned or not.
**The GPU does exactly the same amount of work; only host time inflates.** That
rules out GPU contention and points squarely at CFS throttling — and the load
crossing 198 *exactly* at the G where threads first exceed the quota is the
signature.

Note the non-monotonicity the pinned sweep never showed: unpinned throughput
*peaks at G=2 and then collapses* (436.8 s → 1574.2 s → 1885.7 s). **This is the
"more GPUs made it slower" result, and its cause is the thread pin, not the GPU
count.** An earlier session observed exactly this shape without the pin and
attributed it to GPU count alone; the pinned arm above shows the same hardware
scaling monotonically to 8. Both measurements were right about what they measured.

This is the finding with the most practical value in the repo, because the bug is
invisible: no error, no warning, no log line. Just a pipeline that is 11× slower
than it should be, on hardware that looks busy.

---

## 5. Ray vs. bash-fork

### Parity

The fork launcher and the Ray actor pool produce **identical scores** on the same
input (clip=0.9559, epe=1.9083, lpips=0.2504, ssim=0.9113, psnr=28.6769 on the
3-pair smoke set). This is structural, not lucky: both call the same
`score_one_video()`, and only orchestration differs.

### Where each one actually wins

| | bash-fork | Ray actor pool |
|---|---|---|
| shard count | a bash variable — **can drift from the launch loop** | the pool size; cannot drift |
| dead worker | bare `wait` exits 0, shard silently missing | `RayActorError` propagates |
| load balance | static stride | dynamic queue |
| CPU detection | whatever you code (`nproc` is wrong) | reads the cgroup quota correctly (**198**) |
| thread default | torch's 128 → storms the quota at G≥2 | **1 per actor** → cliff on CPU-bound steady state |
| cold start | per-process, concurrent | ~11–12 s for 8 actors, flat in N |

Measured on ViPE, 4 videos on 2 GPUs, pose-only: fork 82.8 s wall vs Ray 82.5 s —
**a wash**, with Ray's 11.3 s pool build against fork's per-process cold start.
Shard imbalance 1.005 vs 1.046 (this dataset has equal-cost videos, so there is
nothing for a dynamic queue to rebalance — see §7).

### Ray's own default is the single largest penalty measured in this repo

`ray-auto` (Ray untouched: no `num_cpus`, no `set_num_threads`) vs `ray-tuned`
(`num_cpus=24` **and** an explicit `torch.set_num_threads(24)`), both G=8, same 48
videos:

| arm | OMP/actor | torch threads | cores busy/video | wall | per-video | GPU util |
|---|---|---|---|---|---|---|
| `ray-auto` | **1** | **1** | **1.0** | **1195.2 s** | 194 s | 0.49% |
| `ray-tuned` | 24 | 24 | ~20 | **186.3 s** | 14.6 s | 7.2% |

**6.4× slower from one default**, and the mechanism is visible in two numbers:
`cores_busy` is exactly **1.0** per actor, and the box-wide load average sat at
**~11 out of a 198-CPU budget** — 94% of the container's CPU capacity idle while
the job ran. Ray had correctly detected all 198 CPUs (`ray_detected_cpu: 198` vs
`nproc`'s 256) and then handed each actor one thread.

Both arms produced **identical scores**, and the dynamic queue balanced perfectly
in both (6 videos per actor, 8 actors). Cold start was 14.5 s vs 16.3 s — noise.
So this is purely the thread default, isolated.

**The full picture across all launchers at G=8**, same work, same box:

| launcher | thread config | wall | vs best |
|---|---|---|---|
| bash-fork, pinned | `OMP=24` per process | **180.1 s** | 1.00× |
| Ray actor pool, tuned | `num_cpus=24` + `set_num_threads(24)` | 186.3 s | 1.03× |
| Ray actor pool, **auto** | Ray's default (1 thread) | 1195.2 s | **6.64×** |
| bash-fork, **unpinned** | torch's default (128 threads) | 1885.7 s | **10.47×** |

The two correctly-configured launchers are within 3% of each other. **The two
misconfigured ones are 6.6× and 10.5× slower — and they fail in opposite
directions**: Ray starves the workload of threads (1 per actor, 94% of the box
idle), while unpinned bash-fork floods it (1024 threads against a 198 quota, load
average 502). Same 198-CPU ceiling, same workload, opposite ends of the U-curve
from §5 of the README. Neither emits a warning.

> **The fork-vs-Ray *winner* is inside the measurement boundary — do not name one.**
> The 3% gap between the two correct launchers is smaller than the interval the two
> timers disagree over. The bash launcher's internal clock brackets nearly its whole
> process; Ray's excludes ~12.4 s of `ray.init()` + driver startup:
>
> | launcher | sweep-driver wall | launcher-internal wall |
> |---|---|---|
> | bash-fork pinned | 180.3 s | 180.1 s |
> | Ray actor pool tuned | 186.3 s | 173.9 s |
>
> By **sweep-driver** wall fork wins by 3.3%; by **launcher-internal** wall Ray wins
> by 3.5%. The two are not measuring the same interval, so the *winner* is an
> accounting choice, not a result. What survives either way is the magnitude:
> |difference| ≤ 3.5%, a tie. The tables here prefer the sweep-driver wall (it
> includes startup for both, which is what a user waits for) and quote it consistently.

This is the cleanest statement of the repo's thesis: **the launcher choice is
worth 3%; the thread configuration is worth 10×.**

### The honest verdict

**"Ray vs. bash-fork" is not the variable that matters on this hardware; thread
pinning is.** The comparison reversed three times across sessions:

1. 8 GPUs, unpinned bash-fork → **Ray ~2× faster**
2. after pinning `OMP_NUM_THREADS=quota/G` in the bash launcher → **bash-fork
   faster at every N** (34.5 s vs 46.3 s at N=8)
3. both pinned → a wash, decided by fixed cold-start cost

Ray's win in (1) was real but not architectural: its `OMP_NUM_THREADS=1` default
accidentally avoided the very thread-storm §4 measures. **Unpinned thread counts
on a quota-limited container can flip a launcher comparison by 2–3× in either
direction**, independent of actors or scheduling. What survives as a genuine reason
to prefer Ray here is elasticity, fault propagation, and the structural
impossibility of the shard-count bug — not throughput.

### The one place the dynamic queue actually wins: unequal-cost videos

Every result above used the equal-cost dataset, where a static stride is already
perfectly balanced and a dynamic queue has nothing to do. To test the queue, I
built a **variance dataset** (`--jitter-frames 6 24`, frame counts 6–24, a 4×
spread) and ran both launchers at G=8, threads pinned to 24, `--max-frames 24` so
the jitter survives:

| | fork (static stride) | Ray (dynamic queue) |
|---|---|---|
| videos per worker | **6, 6, 6, 6, 6, 6, 6, 6** (fixed) | **6,6,6,6,5,7,5,7** (by availability) |
| shard compute spread | 78.7 – 103.7 s | 84.8 – 97.5 s |
| **shard imbalance** | **1.318×** | **1.15×** |
| straggler | 103.7 s | 97.5 s |
| wall clock | 119.7 s | 112.6 s |

Both produced **identical scores** (clip=0.9485, epe=2.1397, lpips=0.2502,
ssim=0.9197, psnr=27.9358).

The mechanism is visible in the first row: the static stride hands **every worker
exactly 6 videos** regardless of their length, so a worker that draws six long
videos finishes 32% behind one that draws six short ones. The dynamic queue hands
out **5 videos to the workers that drew slow ones and 7 to the workers that drew
fast ones**, compressing the imbalance from 1.318× to 1.15× and the wall clock by
6%. Scores are identical — the queue changes *who* scores each video, not the
arithmetic.

This is a modest 6% here because the variance is modest (4×) and the videos are
short; on a real annotation set with 10–100× duration spread the static stride's
straggler dominates and the gap widens. But it is the *only* configuration in this
repo where "Ray is faster" is true for a reason intrinsic to Ray (its scheduler)
rather than an accident of thread defaults — and it is reported here precisely
because the equal-cost runs could not show it.

> **Re-measured 3× (2026-08-11): report the imbalance, not the wall.** Both arms run
> three times on the variance set: **shard imbalance** is cleanly separated —
> **1.43 ± 0.04 (static stride) vs 1.16 ± 0.01 (dynamic queue)** — but the **wall-clock
> gap washed out to a tie** (128.8 ± 4.7 s vs 127.2 ± 1.4 s; 1.5 s difference against a
> 4.9 s pooled sd, on a box other users were loading). So the durable claim is the
> structural one — *the dynamic queue balances the shards*, which is what it is for and
> what it demonstrably does — not the 6% wall-clock win, which this hardware cannot
> resolve from noise at 4× variance. On a larger spread the imbalance advantage would
> translate to wall clock; at 4× it does not clear the noise floor. (This was flagged
> as a thin, noise-threatened claim during self-review, now disclosed rather than
> overstated.)

---

## 6. The ViPE author's Ray design, measured

The ViPE maintainer (`heiwang1997`, NVIDIA Toronto AI Lab) answered
[nv-tlabs/vipe#41][i41] by confirming Ray is how ViPE is scaled internally and
posting the code. `vipe_slow/run_ray_author.py` reproduces it faithfully —
`@ray.remote(num_gpus=1, num_cpus=4)` on a **task**, one task per video,
`ray.put()` for the stream list, `os.chdir(cwd)` in the worker, `make_pipeline()`
inside the task, a `ray.wait()` drain loop — adding only instrumentation.

> "internally we've already scaled up ViPE mainly using `ray`. It allows pretty
> robust elastic computing using multiple GPUs."

Measured, 4 videos, 2 GPUs, pose-only:

| | author's tasks | this repo's actor pool |
|---|---|---|
| worker processes | **4 for 4 tasks** | 2 actors |
| `make_pipeline()` | **6.9 s mean × 4 = 27.6 s** | 7.6 s **once per actor** |
| per-video wall | 37.7–40.9 s | 32.0–36.1 s |
| cores busy/video | 1.45 | 8.2 |
| total wall | 55.6 s | 82.5 s (12 videos of work, not 4)² |

² the two runs are not wall-comparable at this size; the comparable numbers are
per-video wall and the `make_pipeline()` column.

**The finding is a genuine tradeoff, not a defect.** Ray scheduled each of the 4
tasks into a *fresh worker process*, so model construction was re-paid every
time. At 4 videos that dominates; amortized over hundreds it is noise, and the
task design buys real elasticity — independent, requeueable units and per-video
fault isolation, versus an actor pool where one actor's death takes its queue with
it. The author's `num_cpus=4` is also a sensible conservative default for an
unknown cluster; it happens to leave 182 of this container's 198 CPUs unclaimed at
G=4, which matters here only because this workload has a CPU-heavy stage.

**A bug the task design hides (found, then fixed).** `RawMp4Stream.__init__`
probes frame count with `cv2` on the driver, but the task `os.chdir()`s in the
worker — and **`cv2.VideoCapture` returns 0 frames for an unreachable path instead
of raising**. A relative `--videos` that is valid on the driver silently yields
`len(stream) == 0`, and the failure surfaces much later as a bare
`assert index < len(self)` in `streams/base.py:360`, naming neither the path nor
the cause. Reproduced, then fixed by resolving to absolute before the config is
built plus an explicit in-worker check with a message that names the actual
problem.

[i41]: https://github.com/nv-tlabs/vipe/issues/41#issuecomment-3331003448

---

## 7. Method notes and caveats

- **Equal-cost dataset by default.** All 48 pairs are 81 frames, so a static
  stride is already perfectly balanced and **any dynamic-queue speedup measured
  here would be measuring nothing**. Shard imbalance stays at 1.00–1.09 for both
  launchers, as it should. The variance arm that actually exercises scheduling was
  built with `--jitter-frames 6 24` and is measured in §5.
- **Warmup separated, never averaged in.** Model load is timed on its own;
  steady-state means drop the first sample.
- **The straggler is the number.** Wall clock is set by the slowest shard, so
  `straggler_s` and `shard_imbalance` are reported rather than averaged away.
- **Thread requests are verified**, not assumed: `check_thread_mismatch()`
  compares the request against `torch.get_num_threads()` and prints loudly on
  disagreement. No mismatch fired in any run reported here.
- **Parity is scoped, not blanket.** movebench scores are bit-identical across every
  *scheduling* change (launcher, G, thread count). Two changes are **not**
  bit-identical and both are quantified: `device_policy=gpu` (§9, LPIPS ≤6.7e-4
  relative) and ViPE pose across thread configs (§10, ≤4.5e-4 absolute). The rule
  that emerged: changing *when* work is scheduled is exact; changing *where* it runs
  or *in what order* it reduces is not.
- **Load average traced at 1 Hz** into `loadavg.txt` for every run — a cgroup
  thread-storm is invisible in wall clock but obvious against the quota.
- **FVD is reduced over concatenated I3D embeddings**, never by averaging
  per-shard FVDs (a different, wrong statistic), and is refused below 8 samples
  where the covariance estimate is noise.
- **`--max-frames 24` for the metric sweep.** A full 81-frame × 48-video × 4-cell
  sweep is ~3.3× longer; the 24-frame runs preserve the CPU/GPU ratio (the metrics
  are per-frame) but absolute per-video times are not comparable to an 81-frame
  run.
- **Two bugs in my own harness, fixed and recorded**: `bc` does not exist in this
  container (replaced with `awk` float arithmetic), and `head -1` closing a pipe
  from `sort` raised SIGPIPE which, under `set -o pipefail`, failed a run *after
  every worker had succeeded* (`rc=141`). The second one destroyed a merge whose
  shards were all intact — recovered by re-running `merge.py` directly. Both are
  the same class of bug the repo is about: a default in a layer nobody was
  watching.

---

## 8. Status

**Done and measured**

- [x] container-aware CPU budget (`198`, not `nproc`'s 256), used by every launcher
- [x] per-sample CPU-s / GPU-busy-s attribution via NVML sampling
- [x] reproducible 81-frame dataset generator, equal-cost and jittered modes
- [x] six metrics ported to a shared core; **scores bit-identical across all
      launchers, all GPU counts, and all thread configs** (10 independent runs).
      Scoped by §9 and §10: this holds for movebench under *scheduling* changes;
      it does **not** hold for `device_policy=gpu` (LPIPS ≤6.7e-4 rel) or for ViPE
      pose across thread configs (≤4.5e-4 abs).
- [x] movebench GPU sweep G ∈ {1,2,4,8}, pinned — **3.93× at G=8, 49% efficiency**
- [x] **ViPE GPU sweep G ∈ {1,2,4,8}, pinned — 5.45× at G=8, 68% efficiency.** The
      predicted ordering (GPU-heavier workload scales further) confirmed.
- [x] movebench unpinned arm G ∈ {1,2,4,8} — up to **10.47× penalty**, peak load
      **502 against a 198 quota**, non-monotonic past G=2
- [x] **`ray-auto` vs `ray-tuned` at G=8 — 6.4× cliff** from Ray's 1-thread default
- [x] four-launcher comparison: correctly-configured launchers within 3%,
      misconfigured ones 6.6× and 10.5× slower in opposite directions
- [x] **load-balancing: static stride vs dynamic queue on a 4×-variance dataset —
      imbalance 1.318× → 1.15× (n=1), identical scores. Re-measured 3× (below):
      imbalance holds cleanly at 1.43 ± 0.04 → 1.16 ± 0.01; the wall-clock gap
      (119.7 → 112.6 s at n=1) washed out to a tie at n=3 — claim is about
      imbalance, not wall.**
- [x] ViPE under bash-fork and Ray actor pool, with timing
- [x] the ViPE author's task design reproduced and measured
- [x] `gdown` `fuzzy=` drift and the `cv2` 0-frame trap found and fixed
- [x] **`device_policy="gpu"` sweep — §9.** 5.26× at G=8, CPU-s/video down 57×,
      compute efficiency 53% → 93%, and an Amdahl model that *holds*.
- [x] **ViPE `ray-auto` vs `ray-tuned` — §10.** 1.39× cliff vs movebench's 6.4×;
      the prediction below was confirmed.
- [x] **scale-out vs scale-up topology arithmetic — §11.** 4 containers × 2 GPUs
      **project to** 2.05× the fleet throughput of 1 × 8 GPUs (arithmetic over measured
      per-container walls; upper bound, assumes own-quota + no interference). Reconciles
      the production launcher.
- [x] **ViPE unpinned arm — §12. The project's original "4 GPUs faster than 8" REPRODUCED,
      then STRENGTHENED under repetition.** Two independent unpinned sweeps both peak at
      G=4 (n=1: cached 184.8 s vs 194.3 s at G=8; lazy 177.2 s vs 213.9 s). Re-measured
      3× each: both arms converge on **G=8 ≈ 1.25× slower than G=4** (223.9 ± 14.0 vs
      179.0 ± 8.3 s cached; 220.0 ± 17.8 vs 176.6 ± 10.8 s lazy) — the n=1 cached G=8
      draw was a fast outlier. Root cause located: at G=8 each worker holds one video so
      all eight decode simultaneously — per-video decode 3.25 → 9.10 s, peak load 288 vs
      a 198 quota, `gpu_busy_s` flat.
- [x] **the production `.cache()` path is now a first-class arm** —
      `run_infer.py --stream-mode {lazy,cached}`. Found by reading
      `eval_cam/vipe/run_multinode.py`: it wraps every video in
      `ProcessedVideoStream(...).cache()`, which materializes all frames in host RAM
      (**30.5 MB/frame**, 2474 MB for one 81-frame 1280×1280 clip) before inference.

**Not yet run — named so the gaps are visible**

- [ ] **ViPE unpinned (`THREADS=0`) arm.** — **DONE, see §12.** Left here only as a
      pointer: the `ray-auto` half is §10 (1.39×), the unpinned half is §12, and it
      overturned §3's headline. Nothing further outstanding on this line.
- [ ] **jittered dataset for ViPE** — the eval suite's dynamic-queue win (§5)
      should carry over to ViPE, where real videos have larger duration spread.
- [ ] **induced-failure test** — kill one worker of 8 under each launcher and
      confirm bash-fork's `wait` reports success while Ray raises `RayActorError`.
      This is the last claim in the repo argued from code reading rather than
      measurement.
- [ ] **G=16+ / multi-node**, where the straggler and lock-contention effects from
      Tip 6 begin to dominate. §9's Amdahl fit predicts a **10.77× asymptote** for
      the fixed pipeline regardless of GPU count — that is the falsifiable claim a
      G=16 run would test.
- [ ] **`device_policy=gpu` under Ray**, and on the jittered set. §9 measured the
      fork launcher only.
- [ ] **cross-container interference test** — 4 concurrent 2-GPU containers at a
      49-CPU quota each (4×49 ≤ 198, so it fits this box). §11's 2.05× scale-out is a
      projection that assumes zero interference; this would measure it at smaller
      scale and turn the upper bound into a real number.

---

## 9. `device_policy="gpu"` — fixing the placement instead of scheduling around it

Everything in §3–§7 tunes *scheduling* around a host-bound pipeline. §2 says the
pipeline is host-bound because LPIPS/SSIM/PSNR are handed CPU tensors and never
moved to the accelerator. So the actual repair is to move them. Same sweep, same 48
videos, `--max-frames 24`, `--device-policy gpu`:

| G | threads | original | **gpu policy** | speedup | CPU-s/video | GPU-s/video | peak load |
|---|---|---|---|---|---|---|---|
| 1 | 198 | 708.9 s | **151.0 s** | 4.70× | 1469 → **25.7** | 1.06 → 2.03 | 147 → **20** |
| 2 | 99 | 351.9 s | **81.1 s** | 4.34× | 832 → **15.0** | 1.02 → 2.03 | 144 → **23** |
| 4 | 49 | 232.9 s | **49.0 s** | 4.75× | 557 → **9.8** | 1.04 → 2.02 | 137 → **25** |
| 8 | 24 | 180.1 s | **34.2 s** | **5.26×** | 431 → **7.5** | 1.04 → 2.01 | 141 → **22** |

**57× less CPU time per video, at every G** (57.1 / 55.6 / 56.9 / 57.4 — the
consistency is the check that this is a placement effect and not a scheduling
artifact). GPU-busy-seconds roughly *doubles*, as it must — the work moved onto the
device. Occupancy goes from 7.3% to ~34%, and the peak load average collapses from
~140 to ~22, i.e. the pipeline stops fighting its own CPU quota.

### The important part is the shape of the curve, not the constant factor

`straggler_s` (the slowest shard's compute) separates fixed serial cost from
parallel work. Decomposing:

| G | compute, orig | speedup | eff. | compute, **gpu** | speedup | eff. |
|---|---|---|---|---|---|---|
| 1 | 695.8 s | 1.00× | 100% | 138.1 s | 1.00× | 100% |
| 2 | 338.2 s | 2.06× | 103% | 67.6 s | 2.04× | 102% |
| 4 | 217.0 s | 3.21× | 80% | 34.9 s | 3.96× | **99%** |
| 8 | 162.6 s | 4.28× | 53% | 18.5 s | **7.46×** | **93%** |

Under the original placement compute efficiency decays to **53%** — workers
contending for the CPU quota. Under the fix it holds at **93%**. The residual
sublinearity is no longer contention: it is a **fixed ~14.0 s serial cost** (model
load 9.6–11.5 s, plus merge). A one-parameter Amdahl model, `wall(G) = S +
compute(1)/G` with `S = 14.0` s, predicts the measured curve:

| G | predicted | measured | error |
|---|---|---|---|
| 1 | 152.2 s | 151.0 s | +0.8% |
| 2 | 83.1 s | 81.1 s | +2.5% |
| 4 | 48.6 s | 49.0 s | −1.0% |
| 8 | 31.3 s | 34.2 s | −8.6% |

**This model holds where `G*` (§3) failed**, and the reason is instructive: `G*`
assumed a fixed `W_cpu` that gets divided, which was false because thread
efficiency varies with the thread count. Once the CPU stage is small enough not to
contend, the remaining structure is just serial-vs-parallel, and Amdahl is the right
model. The G=8 cell's −8.6% is the largest error and is in the expected direction:
at 18.5 s of compute per shard, a 6-video shard is only ~3 s per video, so
per-worker variance and merge cost are no longer negligible.

**The falsifiable prediction this makes:** with `S = 14.0` s fixed, the asymptotic
speedup ceiling is **10.77×** no matter how many GPUs are added — G=16 could reach
at most 6.66×, G=64 at most 9.33×. After this fix the thing worth optimizing is the
serial cost (model load), not the GPU count. A G=16 run would test this directly.

### ⚠️ The parity caveat — this is the one change that is *not* bit-identical

Every scheduling change in this repo preserves scores exactly. This placement change
does not. Per-video comparison against the original-placement G=8 run, all 48
videos:

| metric | max abs diff | max rel diff | mean rel diff |
|---|---|---|---|
| clip | **0** | **0** | **0** |
| epe | **0** | **0** | **0** |
| lpips | 1.51e-4 | **6.70e-4** | 3.36e-4 |
| ssim | 3.73e-8 | 4.25e-8 | 1.26e-8 |
| psnr | 1.27e-6 | 4.56e-8 | 2.46e-8 |

CLIP and EPE are unchanged because they already ran on the GPU. SSIM/PSNR agree to
float32 round-off. **LPIPS moves by up to 6.7e-4 relative**, because cuDNN selects
different convolution algorithms for the VGG16 forward pass than the CPU path uses,
and the reduction order differs. Rounded to 4 decimals every aggregate score is
identical (clip=0.9421, epe=2.1085, lpips=0.2567, ssim=0.9197, psnr=27.9234), which
is why the sweep tables above can still be called parity-clean at reporting
precision.

For a ranking metric 6.7e-4 is harmless. But "we moved it to the GPU and nothing
changed" would be false, and the distinction is exactly the kind that matters when
someone else reproduces a benchmark — so the number is reported rather than the
reassurance.

### The two host-side fixes compose: 55.1× on the same 8 GPUs

§4 (thread pin) and §9 (placement) fix independent bottlenecks, so they stack. Same
48 videos, same 8 H100s, same arithmetic, scores identical to 4 dp:

| configuration | wall at G=8 | per video | cumulative |
|---|---|---|---|
| `--gpus all`, thread defaults, metrics on CPU | 1885.7 s | 39.3 s | 1.00× |
| **+** threads pinned to `quota/G` (§4) | 180.1 s | 3.75 s | **10.47×** |
| **+** LPIPS/SSIM/PSNR on the accelerator (§9) | **34.2 s** | **0.71 s** | **55.10×** |

The factors multiply exactly — 10.473 × 5.261 = 55.10 against a measured
1885.7 / 34.2 = 55.10 — which is itself the evidence that the two are independent.
The first bottleneck is CFS throttling from `G × 128` oversubscribed threads; the
second is host-resident VGG16 convolution. Fixing either alone leaves the other.

**Against the scaling result, this is the whole argument in two numbers:** adding 7
GPUs bought **3.94×** (708.9 → 180.1 s at G=1→8, both pinned); fixing two host-side
defaults on a *fixed* 8 GPUs bought **55.10×**. The host fixes were **14× more
valuable than the accelerators.**

One caveat, stated so the ratio is not overclaimed: the 1885.7 s baseline is a **real**
configuration — torch's own thread default plus the placement
`movebench/utils/lpips.py` actually ships — not a strawman built to inflate a number.
But it is also the *worst* of the four launcher configurations in §5. So 55.1× is the
distance between the naive and tuned ends of a real spectrum, not a claim that every
deployment is leaving 55× on the table.

---

## 10. The ViPE `ray-auto` cliff — a prediction, then the measurement

§8 of the previous round recorded a prediction: because ViPE's SLAM steady state is
nearly thread-insensitive (an earlier probe measured **3.85 s at 1 thread vs 3.94 s
at 32**), Ray's 1-thread-per-actor default should cost *much* less here than
movebench's 6.4×. Measured, G=8, 8 videos, all 81 frames:

| arm | threads/actor | cold start | compute | wall | cpu-s/video | gpu-s/video | occupancy |
|---|---|---|---|---|---|---|---|
| `ray-auto` | Ray default (1) | **27.9 s** | 79.3 s | **123.2 s** | 82.6 | 12.31 | 0.160 |
| `ray-tuned` | 24 | 14.8 s | 62.4 s | **88.8 s** | 181.2 | 12.46 | 0.216 |

**Cliff = 1.39×** against movebench's **6.4×** — the prediction was right in
direction and roughly in magnitude. Side by side:

| workload | CPU:GPU ratio | `ray-auto` cliff |
|---|---|---|
| movebench eval | 1390 : 1 | **6.4×** |
| ViPE pose+depth | 85 : 1 | **1.39×** |

Two details worth keeping:

1. **Nearly half the ViPE cliff is cold start, not steady state** (27.9 s vs 14.8 s
   — a 13.1 s gap out of a 34.4 s total gap). Model construction is itself threaded,
   so starving threads taxes startup even for a workload whose steady state does not
   care. `compute_s` alone is 79.3 vs 62.4 s = 1.27×.
2. **`cpu_seconds_per_video` is *higher* in the tuned arm** (181.2 vs 82.6). That is
   not a regression — it is the point. The tuned arm is *allowed* to consume CPU in
   parallel and finishes sooner in wall-clock; the auto arm consumes less total CPU
   because it is single-threaded, and pays for it in wall time. CPU-seconds is a
   measure of resource *consumption*, not efficiency, and reading it as the latter
   inverts the conclusion.

**Verdict:** Ray's `num_cpus` default is a workload-dependent hazard, not a
universal one. Its cost scales with how much of the pipeline is host work — which
is precisely the quantity §2 measures and §9 fixes.

### ⚠️ A second parity caveat, found here

ViPE's outputs are **not bit-identical across thread configurations.** Comparing
`pose/*.npz` between the two arms, all 8 videos:

| | max abs diff |
|---|---|
| pose `data` (worst video, recam3) | **4.52e-4** |
| pose `data` (typical) | 2e-6 – 3e-5 |
| pose `inds` (frame indices) | **0** exactly |

The cause is expected: SLAM bundle adjustment is an iterative least-squares solve
whose floating-point reduction order depends on thread count, and small differences
compound across iterations. Frame indexing is exact, so nothing is misaligned.

This matters because it **bounds a claim made elsewhere in this repo.** "Identical
scores across all launchers and thread configs" is true and verified for the
movebench metric sweeps — those are bit-identical. It is *not* true for ViPE. The
blanket phrasing has been narrowed in the README accordingly.

---

## 11. Scale-out vs scale-up: the arithmetic that explains the production launcher

Every table so far measures **one container**. The production deployment this came
from is different: **multiple Docker nodes, each with its own 198-CPU quota.** That
changes the optimization, because `N` containers of `G` GPUs command `N × 198` CPUs
rather than 198 total.

Normalizing to a **fixed 8-GPU fleet**, computed from the *same measured
per-container wall times* in §3. **Every "fleet throughput" number below is a
projection, not a measurement** — each container's wall was measured *alone on an
otherwise idle box*, and the fleet figure is `N ×` that. Only the `1×8` row was
ever run as an actual fleet; `8×1`, `4×2`, `2×4` are arithmetic (this box cannot
host them — see the assumption block):

**movebench eval, original placement (48 videos):**

| topology | per-container wall | **projected** fleet throughput | vs 1×8 | CPU quota commanded |
|---|---|---|---|---|
| 8 × 1 GPU | 708.9 s | 0.5417 v/s | 2.03× | 1584 |
| **4 × 2 GPU** | 351.9 s | **0.5456 v/s** | **2.05×** | 792 |
| 2 × 4 GPU | 232.9 s | 0.4122 v/s | 1.55× | 396 |
| 1 × 8 GPU | 180.1 s | 0.2665 v/s | 1.00× | 198 |

**ViPE pose+depth (8 videos):**

| topology | per-container wall | **projected** fleet throughput | vs 1×8 |
|---|---|---|---|
| **8 × 1 GPU** | 440.3 s | **0.1454 v/s** | **1.46×** |
| 4 × 2 GPU | 235.2 s | 0.1361 v/s | 1.37× |
| 2 × 4 GPU | 127.9 s | 0.1251 v/s | 1.26× |
| 1 × 8 GPU | 80.6 s | 0.0993 v/s | 1.00× |

**movebench eval, `device_policy=gpu` (§9's fix):**

| topology | per-container wall | **projected** fleet throughput | vs 1×8 |
|---|---|---|---|
| **8 × 1 GPU** | 151.0 s | **2.5438 v/s** | **1.81×** |
| 4 × 2 GPU | 81.1 s | 2.3677 v/s | 1.69× |
| 2 × 4 GPU | 49.0 s | 1.9582 v/s | 1.40× |
| 1 × 8 GPU | 34.2 s | 1.4025 v/s | 1.00× |

### What this settles

**The production launcher's `for gpu in 0 1` was correct, and this is why.** An
earlier session recorded "2 GPUs beats 8" and attributed it to GPU count; §4 then
showed the pinned single-container curve is monotonic to G=8, which appeared to
refute it. Both observations were right about different systems. The reconciliation:

- **Within one container**, more GPUs is always faster (monotonic, §3) — just
  increasingly inefficiently (49% at G=8).
- **Across a fleet**, that lost efficiency is what decides the topology. At G=2 each
  worker still gets 99 threads and per-GPU throughput is undiminished (100.7% of the
  G=1 rate); at G=8 it is down to 49%. Splitting the same 8 GPUs into 4 containers
  recovers that — **a projected 2.05×** (upper bound; see the assumption block).

So the production finding was never about GPUs being harmful. It was about **2 GPUs
per 198 CPUs being the point where an added GPU is still worth its share of the CPU
budget** — the death-linear point, located.

### The assumption, stated so it can be attacked

This projection rests on **two** assumptions, and neither is tested here:

1. **Each container receives its own full 198-CPU quota.** True of the multi-node
   RunAI topology this came from; false if one 198-CPU box is subdivided into 4
   containers of 49 CPUs — in that case the fleet CPU total is unchanged and the
   scale-out gain largely disappears.
2. **Zero cross-container interference.** *Not* stated in the original draft, and
   the less defensible of the two: 4 concurrent containers would contend for memory
   bandwidth, shared L3, PCIe, and NVLink. Nothing here measures that, so **2.05× is
   an upper bound**, not a point estimate.

The honest framing: **scale-out converts an idle CPU quota you are already paying
for into throughput.** It is not free speedup; it is better utilization of a
resource that was being stranded — and the realized gain is at most the projected
one, likely less once interference is paid. A genuine partial test that *does* fit
this box: run 4 concurrent 2-GPU containers at a 49-CPU quota each (4×49 = 196 ≤ 198)
and check whether per-container throughput holds. That is un-run and named in §8.

Three further caveats:

1. **8×1 and 4×2 are statistically tied** for the eval workload (0.5417 vs 0.5456
   v/s, 0.7% apart) — this data cannot rank them. Prefer 4×2 for half the container
   count and half the per-container cold-start overhead.
2. **The optimum is workload-dependent, again.** ViPE's best is 8×1 and its curve is
   monotonically decreasing in `G`, because its death-linear point sits further
   right. There is no single topology recommendation, which is the whole thesis.
3. **The fix compresses the gain.** After §9, scale-out is worth 1.81× instead of
   2.05×, because a pipeline that no longer contends for CPU has less to gain from
   being handed more of it. **Fixing placement and scaling out are partially
   redundant — do the fix first**, since it is one flag and needs no extra hardware.

---

## 12. ViPE: 4 GPUs really is faster than 8 — reproducing the production result

§3 measured ViPE scaling monotonically to G=8, and I reported that as the answer.
That was the wrong arm. The project's original brief records the observation precisely:

> *"I copied `vipe_slow` from `eval_cam/vipe` (which is the version i found that run
> with workload with **4 gpus faster than run with 8 gpus with all 198 cpu
> cores**)"*

Reading the production launcher (`eval_cam/vipe/vipe_unicamo.sh` +
`run_multinode.py`) against my harness turned up **two differences, and my sweep had
reproduced neither**:

| | production (`run_multinode.py`) | my `run_infer.py` (§3) |
|---|---|---|
| `OMP_NUM_THREADS` | **never set** — fully unpinned | pinned to `198/G` |
| stream construction | `ProcessedVideoStream(RawMp4Stream(p), []).cache()` | `StreamList.make()` — lazy |

The second is the substantive miss. **`.cache()` decodes every frame into host RAM
before inference begins** (`streams/base.py:467` forces it by touching the last
element). Measured on one 81-frame 1280×1280 clip: **2474 MB resident, 30.5 MB per
frame, 1.35 s** to build. That is per-worker host work that multiplies by `G`, and
the lazy path never pays it.

### The measurement, in the production configuration

Unpinned, `--stream-mode cached`, 8 videos, all 81 frames. `run_infer.py` now takes
`--stream-mode {lazy,cached}` so the production path is a first-class arm rather
than a description:

| G | threads requested | **cached (production)** | lazy | pinned (§3) |
|---|---|---|---|---|
| 1 | 128 | 448.4 s | 436.0 s | 440.3 s |
| 2 | 256 | 245.6 s | 237.0 s | 235.3 s |
| 4 | 512 | **184.8 s** ← best | **177.2 s** ← best | 128.1 s |
| 8 | 1024 | **194.3 s** | 213.9 s | 80.6 s |

**G=4 beats G=8 by 1.05× (cached) and 1.21× (lazy).** Two independent unpinned
sweeps, run hours apart, both peak at **G=4** — the value the original brief recorded.
The production result reproduces.

> **Re-measured 3× (2026-08-11): the effect is larger and firmer than these single
> shots.** The n=1 cells above are honest but the G=8 cached draw (194.3 s) was a
> lucky-fast outlier. With G=4 and G=8 each run three times:
>
> | arm | G=4 (mean ± sd) | G=8 (mean ± sd) | G8/G4 | pooled sd | verdict |
> |---|---|---|---|---|---|
> | unpinned + cached (production) | 179.0 ± 8.3 s | 223.9 ± 14.0 s | **1.25×** | 16.2 s | 45 s gap ≫ noise |
> | unpinned + lazy | 176.6 ± 10.8 s | 220.0 ± 17.8 s | **1.25×** | 20.8 s | 43 s gap ≫ noise |
>
> Both arms converge on **~1.25× slower at G=8**, and the gap is roughly 2–3× the
> pooled run-to-run spread — so "4 GPUs beats 8" is not a marginal n=1 artifact. This
> was the finding the user cares most about, flagged during self-review as thin at
> 5.2%; the repeat retires that concern by making the effect *bigger*, not smaller.

### Where the regression comes from

Ideal scaling would put G=8 at 92.3 s (half of G=4's 184.6 s). It measured 194.2 s —
**102 s worse than ideal.** Decomposed:

| | G=4 | G=8 | change |
|---|---|---|---|
| videos per worker | 2 | **1** | nothing left to pipeline |
| **cache decode per video** | 3.25 s | **9.10 s** | **2.8× slower** |
| CPU-seconds per video | 1804 | 3383 | 1.9× |
| peak load average (quota **198**) | 150 | **288** | 1.9× |
| shard imbalance | 1.032 | **1.276** | stragglers appear |
| **`gpu_busy_s` per video** | 12.94 s | 13.74 s | **flat — the control** |

`gpu_busy_s/video` is flat, so **the GPU does identical work at every G; only host
time inflates.** The mechanism: at G=8 every worker holds exactly one video, so all
eight decode *simultaneously* — per-video decode goes 3.25 s → 9.10 s while peak
load hits 288 against a 198-CPU quota. At G=4 each worker has two videos, so one
worker's decode overlaps another's GPU phase. **Doubling the GPUs removes the
pipelining that was hiding the host stage, and the CFS quota charges for it.**

That also explains why the effect is *stronger* in the lazy arm (1.21×) than the
cached arm (1.05×): `.cache()` front-loads decode into a burst that the OS can at
least schedule contiguously, whereas lazy decode interleaves with inference and
contends throughout.

### The 2×2 that isolates the cause

Running all four combinations of {pinned, unpinned} × {lazy, cached} turns this from
an anecdote into a controlled experiment. 8 videos, 81 frames, same box:

| G | unpinned+cached **(production)** | unpinned+lazy | pinned+cached | pinned+lazy |
|---|---|---|---|---|
| 1 | 448.4 s | 436.0 s | 444.6 s | 440.3 s |
| 2 | 245.6 s | 237.0 s | 234.7 s | 235.3 s |
| **4** | **184.8 s** ← best | **177.2 s** ← best | 133.6 s | 128.1 s |
| 8 | 194.3 s | 213.9 s | **79.3 s** ← best | **80.8 s** ← best |

| arm | optimum | `G8/G4` | verdict |
|---|---|---|---|
| unpinned + cached (production) | **G=4** | 1.052 | **4 beats 8** |
| unpinned + lazy | **G=4** | 1.207 | **4 beats 8** |
| pinned + cached | G=8 | 0.594 | 8 fastest |
| pinned + lazy | G=8 | 0.631 | 8 fastest |

**The thread pin is the whole story.** Both unpinned arms invert at G=4; neither
pinned arm does — including the one that still pays the full `.cache()` cost. So
`.cache()` is *not* the cause of the inversion; it only modulates its size (1.05× vs
1.21×, and it is the *smaller* effect because front-loading decode into one burst
schedules better than interleaving it with inference). The cause is
`G × 128` unpinned threads against a 198-CPU quota.

This is worth stating as a factorial result because it is the difference between
"4 GPUs was faster for us" and **"4 GPUs is faster iff threads are unpinned, and the
fix is to pin them — which then makes 8 GPUs 2.29× faster than the 4-GPU optimum
ever was."**

### What this changes, and what it does not

1. **The original brief's claim is correct as written** — with the production launcher
   (unpinned, `.cache()`), on this box, **4 GPUs beat 8**. Quotable.
2. **It is a host-contention result, not a GPU result.** The `gpu_busy_s` control
   proves the accelerators do the same work; the cost is decode + CFS throttling.
3. **The cause is the thread pin, not `.cache()`** — established by the 2×2 above.
   `.cache()` changes the magnitude (1.05× vs 1.21×), not the sign.
4. **Pinning threads is the bigger lever by far.** Pinned+lazy at G=8 is **80.8 s —
   2.29× faster than the production 4-GPU optimum (184.8 s)**. The strongest honest
   statement: *under the shipped configuration the optimum was 4 GPUs; pinning the
   threads moved the optimum to 8 and beat the old optimum by 2.29×.*
5. **§3 was not wrong, it was incomplete.** It measured pinned+lazy, which genuinely
   is monotonic. Reporting it as *the* ViPE answer when the production code was
   neither pinned nor lazy is the error — the same class of mistake this repo is
   about: **benchmarking a configuration nobody runs.**

The methodological lesson: **read the launcher you claim to be modelling.** A
two-line difference between `StreamList.make()` and `ProcessedVideoStream(...).cache()`
inverted the headline conclusion, and no amount of care *inside* the harness would
have surfaced it.


---

## 13. Nine engineering lessons

Nine hard-won lessons from building the benchmarks above — the part of this repo an
engineer running the same workloads will actually reuse. In rough order of how much
damage each one does. Every number here traces to the same committed evidence as the
findings: `results/**/*.json`, `common/resources.py`, and the launcher logs.

### Lesson 1 — `torchrun` silently sets `OMP_NUM_THREADS=1`; `python` does not

```bash
python   worker.py   # torch takes the box default: nproc/2 = 128 threads
torchrun worker.py   # torchrun forces OMP_NUM_THREADS=1 unless you set it
```

`torchrun` prints one line — *"Setting OMP_NUM_THREADS environment variable for each
process to be 1"* — and moves on. For a GPU-bound job you will never notice. For
anything with a CPU-side hot loop (every perceptual metric, every decode, every
host-side preprocess) you have just single-threaded it.

```bash
export OMP_NUM_THREADS=24     # (cgroup quota) / (workers)
torchrun --nproc_per_node=8 worker.py
```

**Always set it explicitly and record it next to the timing.** An unset environment
variable is an unmeasured variable.

### Lesson 2 — `OMP_NUM_THREADS` must be set *before* `import torch`

```python
import torch
os.environ["OMP_NUM_THREADS"] = "24"   # <- silent no-op; torch already read it
```

torch fixes its intra-op pool size when the extension loads. Export it in the shell
before the process starts, or set it at the very top of the entry point before any
torch import — and use `torch.set_num_threads(n)` if you must change it later. Every
worker here does env-first ordering, and `check_thread_mismatch()` verifies the request
actually took effect. A silent mismatch voids every timing in the run.

### Lesson 3 — in a container, `nproc` lies

```bash
nproc                                       # 256  <- the HOST, not you
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us     # 19800000
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us    # 100000
# 19800000 / 100000 = 198  <- the enforced ceiling
```

cgroup CPU throttling **does not shrink the affinity mask**. It lets you spawn 256
threads, then freezes every one once the cgroup burns its quota inside each 100 ms
period. **There is no error — only slowness.** Anything sized off `nproc`
oversubscribes by 1.29× here and by arbitrary factors elsewhere.
`common/resources.py` is the single place this is decided (cgroup v1 + v2, falling back
to the affinity mask on bare metal).

### Lesson 4 — Ray reads the quota correctly and then hands each actor one thread

Two separate behaviours, worth separating because one is a point in Ray's favour:

```python
ray.available_resources()["CPU"]    # 198.0  -- correct! reads the cgroup quota
```

But:

```python
@ray.remote(num_gpus=1)
class Probe:
    def info(self):
        return os.environ["OMP_NUM_THREADS"], torch.get_num_threads()
# -> ('1', 1)   for every actor
```

Ray maps `num_cpus` (default 1) onto `OMP_NUM_THREADS`. Whether that is a bug or a
feature depends entirely on your workload — measured, same default:

- **CPU-bound steady state** (this eval suite): a cliff — **6.4×**.
- **GPU-bound steady state** (ViPE): **1.39×**, mostly cold start.
- **Concurrent cold start** (either): *protective* — 8 unpinned processes at 128
  threads each storm the cgroup, and Ray's low default accidentally avoids it.

```python
@ray.remote(num_gpus=1, num_cpus=24)
class Worker:
    def __init__(self):
        torch.set_num_threads(24)   # explicit; num_cpus alone is import-order dependent
```

### Lesson 5 — the thread U-curve has two walls, and the right one is worse

Holding workers and workload fixed, sweeping only threads-per-worker (16 workers,
CPU-only box, 128 physical cores):

| total threads | regime | time |
|---|---|---|
| 16 | 8× under | 33.0 s |
| 64 | under | 13.6 s |
| **128** | **= physical cores** | **7.4 s** <- floor |
| 256 | = logical (SMT) | 25.7 s |
| 512 | 4× over | 48.0 s |

The floor sits at `total_threads == physical cores`, and **over-subscription is worse
than under-subscription** (512 threads is slower than 64). SMT siblings do not help a
VGG-convolution workload — they share one AVX-512 FPU. Naive Ray sits on the left wall
(`num_cpus=1`); an unpinned `torchrun` with 8 workers sits on the right one. *(This
curve is from the sibling CPU-only study `ray_learning/cpu_bench`, not this repo's
8-GPU runs — its JSON is committed at `results/cpu_bench_ucurve.json`.)*

### Lesson 6 — put the fleet's shared cache on a *node-local* disk

*Extracted from diagnosing UniCaMo's 96-GPU (12×8) video-diffusion fleet, where every
rank stalled 5–10 minutes at startup.* The model hub placed its download lock under
`$HOME`; `$HOME` was one NFS export mounted in all 12 containers; so a lock the library
believed was per-machine was in fact **fleet-global** — 96 ranks serialized to *not*
download 107 GB of weights they already had locally.

| | before | after |
|---|---|---|
| load phase | 9 m 06 s | **~44 s** (~12×) |
| intra-node GPU desync | 267 s | **~11 s** (~24×) |
| denoise throughput | 8.74 s/it | 8.77 s/it (unchanged) |

Three transferable lessons:

1. **Lock hold time degrades under contention** (2.7 s → ~17 s at 96-way), so cost
   grows *super-linearly* in fleet size — doubling the fleet more than doubles the
   queue. This is anti-scaling, not merely slowness.
2. **The observable symptom named the wrong subsystem.** It surfaced as an NCCL
   watchdog timeout; NCCL was innocent. Cross-layer failures live in the *seam* between
   layers, where no single layer can see them.
3. **Straggler variance is what multi-node jobs pay for**, not mean throughput — fleet
   wall-clock is set by the slowest rank, so an unbounded wait converts a tail latency
   into the mean cost of every launch. The fix bought *determinism* (267 s → 11 s
   spread), and denoise throughput never moved.

*Diagnosed at 96 ranks; fix measured on one node. Full write-up:
`../../multinode_management_unicamo.md`.*

### Lesson 7 — `torch.compile`'s win evaporates as input shapes multiply

Guard misses force recompilation; past Dynamo's recompile ceiling the function stops
specializing and falls back to roughly uncompiled speed. Observed qualitatively: the
speedup holds for **~16** distinct shapes and is largely gone by **~100**. Video work
generates shape variety naturally (resolutions, frame counts, aspect ratios), so a
compiled path benchmarked on one shape can be a pessimization in production. Bucket or
pad shapes, and watch `TORCH_LOGS=recompiles`. *(Thresholds are remembered from
production work, not logged here — flagged as approximate rather than quoted as
measured.)*

### Lesson 8 — a dependency that returns 0 instead of raising will cost you an hour

`cv2.VideoCapture` on an unreachable path reports **0 frames** and does not raise. In
ViPE that makes `len(stream) == 0`, and the failure surfaces hundreds of lines later as
a bare `assert index < len(self)` naming neither the file nor the cause. This bites
specifically in the Ray-*task* design, where the stream list is built on the driver and
the worker then `os.chdir()`s elsewhere — a relative path valid on the driver silently
becomes empty in the worker. **Resolve paths to absolute before they cross a process
boundary, and assert on frame count at construction.**

### Lesson 9 — two harness bugs, because the tooling is part of the experiment

- **`bc` does not exist in this container.** Float arithmetic in the launchers uses
  `$EPOCHREALTIME` + `awk`.
- **`sort | head -1` under `set -o pipefail` fails the script.** `head` closing the pipe
  SIGPIPEs `sort` (rc=141), which once failed a run *after every worker had succeeded* —
  the shards were all on disk. Both launchers now end that pipeline with `|| true`, and
  a dead merge is recovered by re-running `merge.py` rather than re-running the sweep.

Both are the same class of bug as the rest of the repo: **a default in a layer nobody
was watching.**
