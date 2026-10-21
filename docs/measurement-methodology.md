# Measurement methodology

This document combines two complementary source notes. Historical measurements are source-reported examples from separate experiments, not a single reproduced run.

## Investigation workflow

The order matters more than any single trick.

## 1. Build a reproducible baseline

- Fix the things that change the workload: model, data mix (e.g. image/text ratio), sequence length,
  packing, token budget per step, sample order, seed.
- Measure steady state, not the first steps: skip warm-up (JIT, autotuning, allocator growth), then
  average over a window (e.g. 100 steps), and repeat the run (e.g. 3×). Report the coefficient of
  variation; above ~2%, the number is not stable enough to compare.
- Use one metric whose numerator cannot move: logical-FLOPs MFU ([Logical MFU derivation](https://github.com/vInThePrism/logical-mfu/blob/main/docs/mfu-from-first-principles.md)),
  plus tokens/s/GPU.

## 2. Decompose the step before touching anything

From an Nsight Systems trace (or Holistic Trace Analysis over many ranks), split each step into:

| Bucket | What it means | Typical causes |
|---|---|---|
| compute | GPU kernels doing math | small/fragmented kernels, poor tiles, non-matmul work |
| exposed communication | a communication kernel runs, no compute does | missing overlap, barriers, wrong placement on links |
| host wait | GPU idle, waiting for the CPU | launch-bound loops, `.item()` / `.tolist()` syncs, data loading |

Then optimize the biggest bucket. Example from one MoE job: compute 48%, exposed communication 34%,
host wait 18%: three different problems with three different fixes. Optimizing an isolated kernel first
would have missed two of them.

## 3. Use control experiments to rule hypotheses out

A control changes one thing that would remove a suspected cause, and nothing else.

- *"Experts are imbalanced."* → force uniform routing. <3% faster → not the cause.
- *"The new kernel isn't faster."* → is the kernel's own GPU time lower? If yes, but the step is not
  faster, the problem is around the kernel (launches, syncs, overlap), not in it.
- *"Communication is slow."* → measure achieved link bandwidth. If it is already >85% of the link,
  the problem is that communication is *exposed*, not slow.

## 4. Change one variable per A/B run

Single-variable A/B runs give each optimization its own number: e.g. swapping only the RMSNorm kernel
(382.8 → 169.5 ms/step), then only the loss (full logits → Cut Cross-Entropy, 476.9 → 18.0 ms/step).
Stacked changes give one number you cannot attribute.

## 5. Separate lossless from lossy

Label every optimization:

- **Lossless:** same math, same semantics (a faster attention kernel, grouped GEMM, recompute policy,
  overlap, better stage balance).
- **Lossy:** changes numerics or semantics (BF16 optimizer states, FP8 GEMMs, a token-dropping MoE
  capacity factor, larger micro-batch that changes the schedule).

Report them as two separate steps. Example from one 8-GPU MoE continued-pretraining job: MFU
7.2% baseline → 8.6% lossless → 13.9% after lossy changes. A reviewer must be able to take the
lossless number without accepting the lossy risk.

Also record the negative results: e.g. activation offload and optimizer CPU offload were net losses in
that setup, and MoE overlap ran out of memory at EP=8 on 8 GPUs. Those save the next person a week.

## 6. Prove correctness

- Kernel level: fp32 reference, normalized L2 ≤ 1e-2 on outputs and *all* gradients, max error, NaN/Inf
  as hard failure, shapes chosen by bug probability ([attention checks](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/tools/attn_check)).
- Model level: matched runs with the same data order; compare loss curves (e.g. final-50-step mean
  within 0.5%) and gradient norms.

## 7. Prove it is real

- Microbenchmarks: warm-up, L2-cache rotation, locked clocks, 3 repeated runs, geometric mean over shapes.
- Then end to end, inside a fixed multi-GPU job. Microbenchmark gains often shrink there.
- Scaling: weak-scaling efficiency (e.g. 91.1% from 8 to 32 GPUs) shows the speedup survives scale.

## 8. Check the meter

The MFU ratio must equal the throughput ratio. If they disagree, the numerator moved, and every number
above it is suspect.

## Measurement contracts and reporting

Rules I now follow because each one caught a wrong number. These are measurement requirements; tool support must be checked separately.

## Aggregation

* **Time-weighted, never mean-of-ratios.** Steady-state MFU = ΣFᵢ / (ΣTᵢ · N · P).
  With variable-length packed data, a median of per-step values was −4.25 % off the
  time-weighted value in a representative case. Median/p25/p75 are jitter diagnostics.
* **No outlier rejection.** Dropping >3σ steps hides GC, eval and save spikes that
  belong to the workload, and silently flatters the result. Keep every raw step;
  exclude warm-up only from the steady window, flagged.
* Steady window: drop compile/first steps and epoch-boundary short steps explicitly,
  and say which.

## The step window

* Opens **before the first micro-batch is requested**. HF Trainer prefetches the
  whole gradient-accumulation group before `on_step_begin`; anchoring there puts
  data wait outside the window (measured: first step 0.467 s vs steady 0.0046 s, a 100×
  stall that would vanish).
* Closes after optimizer + scheduler, before eval/logging/checkpoint.
* One monotonic host clock with a `cuda.synchronize()` before the end timestamp.
  CUDA events measure stream time and miss host stalls.
* No device→host reductions (`.item()`, `.tolist()`) inside the window. Capture
  references, reduce after the window closes.

## Single-variable discipline

* One change per A/B. A config diff of exactly the intended lines (I diff YAMLs before
  every comparison). When two things changed — e.g. packing *and* a new dispatcher —
  say the result cannot be attributed.
* Verify the variable from runtime evidence: log markers, kernel names in the trace,
  launch counts. Config intent is not evidence (see the silent-failure catalogue).
* Additive attribution ("A gave +x, B gave +y") needs per-change isolation runs;
  otherwise report only the combined gain.

## Machine state

* Before a run: every target GPU `memory.used` < 2 GB; `/proc/loadavg` sane. One run
  had a neighbouring job's preprocessing workers push load to 216 on 80 cores; step
  time jumped between 7 s and 41 s and throughput read 4.2× low.
* GPUs share an NVSwitch fabric: a neighbour on *another* GPU can slow your
  collectives. Two otherwise identical runs differed 8.0 %; the FLOP sums were
  bit-identical, both runs were internally stable (≤3.3 % range), and one GPU in the
  node was running someone else's 589 W job. Same-period, idle-node runs only.
* Repeat the baseline in reversed order once; if the two baselines disagree by more
  than your claimed gain, you have no result.

## Instrumentation overhead

* Never take step time from a run that also counts FLOPs or captures a full trace.
  FLOP counting inflated step time ×2.1–2.4, and differently per config.
* For nsys, check that Δt inside the trace window matches Δt from a clean run (in one
  case: −6.76 % vs −6.75 %); then the profiler's ~1.4 % overhead cancels in the delta.

## Reporting

* Include flat or negative results when they bear on the claim.
* Say which inputs were measured and which were borrowed or estimated.
* A ceiling ("perfect overlap would save ≤ 18 %") is not an expected gain; label it.
* State the scope: GPU, precision, framework versions, shapes. Most conclusions here
  are version-bound.
