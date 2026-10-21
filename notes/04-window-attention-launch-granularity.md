# 1,006,980 attention launches: when "FA3 is slower than FA2" is a dispatch bug

**Setting.** Qwen2.5-VL-7B LoRA SFT on H800, DeepSpeed. Switching the attention
backend from FA2 to FA3 made the training window **2.1× slower** (92 s → 198 s for
10 steps), even though FA3's kernel is faster.

## Diagnosis

Nsight Systems, same 10-step window:

| | FA2 baseline | FA3 (transformers 4.57.1) | FA3 varlen (5.6.1) |
|---|---:|---:|---:|
| Training window | 92.4 s | 197.6 s | 93.2 s |
| GPU kernel launches | 384,918 | **1,398,822** | 387,855 |
| Total GPU kernel time | 35.0 s | 37.0 s | 31.6 s |
| Flash-attention launches | 6,000 | **1,006,980** | 9,520 |
| Flash-attention GPU time | 7.29 s | 8.79 s | 3.79 s |

Total GPU work barely changed (35 s → 37 s) while wall time doubled. That is the
signature of a **launch-bound** regression, not a slow kernel. The three-part test
I use for "launch-bound":

1. total kernel GPU time roughly unchanged between the two runs;
2. median kernel duration is at the scale of launch overhead (a few µs);
3. the GPU timeline is full of small gaps, and cumulative host time in
   `cudaLaunchKernel` fills the step wall-clock.

"SM activity is low" alone is not evidence; it is equally consistent with a
memory-bound kernel.

## Root cause

The vision tower uses **window attention**: 112×112-pixel windows → 8×8 = 64 patch
tokens per window, with full attention only in a few layers. The old adapter path
called the attention kernel **once per window**, so a batch of high-resolution
images produced hundreds of thousands of tiny calls. (It is easy to blame the
LLM-side image placeholder tokens; the count does not fit that explanation.)

The fix was not a new kernel. The upstream model code in transformers 5.6.1
batches all windows into **one varlen call with `cu_seqlens`** (5.6.0 had a
`s_aux=None` bug in the flash path; 5.6.1 fixed it).

Why varlen is better than a block-diagonal mask, precisely: with varlen, the KV-loop
bounds come from `cu_seqlens`, so tiles that cross a sequence boundary are **never
loaded or computed**. A block-diagonal mask computes those tiles and then discards them.

## Result

30-step throughput A/B, same GPU, same data order and seed, same 376,997 non-pad
tokens:

| | runtime | tokens/s |
|---|---:|---:|
| FA2 (run 1) | 277.7 s | 1357.5 |
| FA2 (run 2, reversed order) | 276.8 s | 1362.0 |
| FA3 varlen | 159.4 s | **2365.4 (1.74×)** |

The two FA2 runs agree within 0.33 %, ruling out page-cache/order effects.

Numerical check: 313 windows × 64 tokens, 16 heads, head_dim 80, BF16 — output and
`dq/dk/dv` cosine 1.0 vs SDPA, max-abs ≤ 3.9e-3. A dispatch unit test confirmed 3
variable-length sequences → exactly 1 kernel call with correct `cu_seqlens`.

## Honest caveats

* The A/B changed more than the attention library (cu126 → cu128, transformers
  4.57 → 5.6.1). 1.74× is a *stack* gain; it is not all attributable to the FA3 kernel.
* 30-step loss drifted by ~0.03; operator-level numerics passed, but convergence
  needs a longer run before production use.
* The fine-tuning framework pinned `transformers<=4.57.1`; running 5.6.1 is an
  override that needs a maintained compatibility patch.

## Takeaway

Before concluding "kernel X is slower", count launches. When kernel count explodes
and total GPU time does not, the problem is call granularity in the framework, and
the fix is usually batching into varlen — not a faster kernel.
