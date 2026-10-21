# FP8 training on H800 fine-tuning shapes: the GEMM wins, the step loses

A negative result, measured to the bottom. Worth publishing because most FP8
numbers in circulation are GEMM-only microbenchmarks or large-scale pre-training.

**Shapes:** Qwen3-32B projections (hidden 5120, intermediate 25600, q_out 8192,
kv_out 1024), `M` = 1632 tokens (the measured median packed length, aligned to 16).
H800, torch 2.8, `torch._scaled_mm` e4m3 vs `torch.matmul` bf16. A runnable probe is
in `tools/probes/fp8_breakeven_probe.py`.

## Four rounds

**1 — GEMM only (operands pre-quantised, not timed): 1.77×** (per-layer 1.69–1.91×;
median over all shapes 1.75×; the only loser was `kv_proj` at M=512, N=1024 too narrow
to fill the GPU). FP8's ceiling is real.

**2 — add dynamic quantisation, eager** (amax → scale → cast as separate kernels):

| variant | speed-up vs bf16 |
|---|---:|
| GEMM only | 1.77× |
| LoRA-like: weight cached, activation quantised every step | **0.69×** |
| Full SFT: activation + weight quantised every step | **0.28×** |

Caching the frozen weight's FP8 copy helps (0.28 → 0.69) but activation
quantisation alone eats the whole 1.77×.

**3 — `torch.compile` the quantise+GEMM region** (Inductor fuses amax/scale/cast,
but cannot fuse into `_scaled_mm`; still two kernels):

| | eager | compiled |
|---|---:|---:|
| LoRA-like | 0.74× | **1.09×** |
| Full SFT | 0.41× | **0.76×** |

Very uneven per layer: MLP gate/up 1.6×, down 1.1–1.3×, but `kv_proj` 0.25–0.53× and
`o_proj` 0.57–1.18×.

**4 — real training, 8×H800, Qwen3-32B full SFT, ZeRO-3, 32k, eager, 10 steps:**

| | s/step | vs bf16 |
|---|---:|---:|
| bf16 | 13.25 | — |
| TE FP8 DelayedScaling | 14.20 | **−7.2 %** |
| TE FP8 Float8BlockScaling | 14.10 | **−6.4 %** |

Loss curves identical (numerics fine). The two recipes are within 0.7 % even though
block scaling recomputes per-block amax every step — so amax bookkeeping is not the
bottleneck; casts and non-GEMM work are. Earlier runs agreed: torchao FP8 eager on the
same 32B job −32 %; FSDP2 + torchao FP8 on 2×14B −37 %.

## Why compile, the only lever, was unavailable

With packed variable-length batches, `M` changes nearly every step; dynamo
recompiled until `recompile_limit`, then OOM. Getting compile would have required
moving to FSDP2, disabling packing, and patching an accelerate FP8 all-gather bug —
for a best case of +9 % on LoRA.

## Conclusion and its scope

On H800 + fine-tuning shapes + an eager stack, FP8 is net negative; the only
positive configuration (+9 %, LoRA, compiled) is within what real-training overhead
erases. This says nothing about Blackwell MXFP8/NVFP4 or large pre-training runs,
where reported gains are real — the regime is different and needs re-measuring.

**The method transfers:** measure GEMM-only, then GEMM + quantisation in the
execution mode you will actually run, then end-to-end. Stop at the first round that
fails.
