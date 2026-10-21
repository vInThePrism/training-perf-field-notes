# Same kernel swap: +3.6 MFU points on 1 GPU, +1.1 on 8 GPUs. Why?

**Swap under test** (both scales, strict single-variable): loss Liger
fused-linear-cross-entropy → Cut-Cross-Entropy (CCE); RMSNorm Liger (Triton) →
QuACK (CuTe-DSL). RoPE/SwiGLU stayed on Liger, FA3 on both sides. Verified from log
markers, not config.

| | before | after | ΔMFU |
|---|---:|---:|---:|
| 1×H200, Qwen3-8B LoRA, 4k cutoff, no GC | 49.43 % | 53.01 % | **+3.58 pt** |
| 8×H800, Qwen3-32B full SFT, 32k cutoff, GC, ZeRO-3 | 43.23 % | 44.30 % | **+1.07 pt** |

The per-step numerator was bit-identical on 1 GPU (`total_model_flops` equal) and
within 0.05 % on 8 GPUs (packing randomness), so ΔMFU is entirely Δtime.

## The answer is a ratio, not a constant

`dMFU = MFU_old · s / (T − s)` where `s` is time saved per step and `T` the step time.
Normalised per token on one device:

| | 1-GPU 8B | 8-GPU 32B | ratio |
|---|---:|---:|---:|
| saved (µs/token) | 5.25 | 10.89 | ×2.07 |
| step (µs/token) | 77.8 | 458.5 | ×5.89 |
| share | 6.75 % | 2.38 % | ÷2.84 |

The kernels did **more** work on 32B (4× the absolute seconds saved per step) — the
step just grew faster (ZeRO-3 gathers, recompute, 32k attention). `tools/.../amdahl.py`
reproduces the table.

My first explanation — "CCE saves a constant µs/token because it scales with vocab
only" — was wrong, and the kernel-level trace showed why.

## What CCE actually removes: an entire GEMM chain, not just a memory round-trip

Kernel-by-kernel diff of two steady steps on 1 GPU:

* Two cuBLAS GEMMs each lost **512 launches** (= 256 per step) — one vanished
  entirely. 256 per step is exactly the number of **chunks** Liger FLCE uses
  (131,072 tokens/step ÷ 512 tokens/chunk). Liger materialises a logits chunk and
  runs a projection GEMM + CE kernel *per chunk*.
* CCE streams the projection inside its own kernel and launches **16** times.
* Every other GEMM matched to within 0.1 ms with identical launch counts.

Sanity check: logits forward = 2 × 131,072 × 151,936 × 4,096 = 163 TFLOP/step;
the two vanished GEMMs took 466 ms/step for fwd+bwd (~3× FLOPs) = 35 % of peak.
Plausible.

So CCE's benefit must be counted as *loss kernel + removed per-chunk logits GEMMs*,
and it depends on hidden size and chunking — not just vocab.

## Closing the books

Σ(all kernel deltas) = Δbusy = 673.5 ms/step exactly (summed over every kernel,
not sampled). Predicting the new step time from kernel deltas alone:
10.1968 − 0.6691 = 9.5277 s → MFU 52.90 % vs measured 53.01 %. **0.11 pt error,
no fitted parameters.** Per-item: CCE +2.33 pt, QuACK +1.06 pt.

## Why 1-GPU ratios do not transfer to 8 GPUs

Applying the 1-GPU replacement ratios (CCE 26.5×, QuACK 2.26×) to the 8-GPU residual
kernels predicts 6.39 s saved; measured 2.755 s. CCE's ratio is dominated by the
256 chunk GEMMs, and with a 32k cutoff and micro-batch 1 the chunking is different
(8 CCE launches/step instead of 16). QuACK's ratio does transfer (fused norm vs fused
norm). Transferring only QuACK's ratio and solving for CCE gives a split of
33 % CCE / 67 % QuACK on 8 GPUs — the reverse of 1 GPU (68 / 32). The norm target
grows with layers × hidden (36×4096 → 64×5120) and gradient checkpointing runs the
forward norm twice. *(The 8-GPU split rests on the transfer assumption; the total is
measured.)*

## Takeaways

1. Report kernel gains as a share of step time, with the step-time breakdown.
2. Verify the mechanism in the trace (launch counts are the cleanest signal) before
   writing the explanation.
3. A kernel-level budget that closes against measured step time is the strongest
   form of attribution there is.
