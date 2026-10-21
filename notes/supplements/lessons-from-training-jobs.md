# Lessons from tuning real training jobs

Before learning to write kernels, I spent an internship making LLM fine-tuning and continued-pretraining jobs faster on multi-GPU H-class and B-class nodes. These are the lessons that generalize, stated without internal details: no model names, configurations, absolute numbers or paths. Each one connects to a CUDA-level idea in the other notes.

## 1. Use a kernel-invariant logical MFU numerator

MFU divides useful logical model FLOPs by wall time and the stated dense hardware peak. A blanket `6 · parameters · tokens` misses LoRA/frozen gradients, sequence-dependent attention, active MoE experts, and separate vision-token geometry. Derive forward, input-gradient, and weight-gradient work from the modeled training graph and actual attention segments. See the canonical [MFU derivation](https://github.com/vInThePrism/logical-mfu/blob/main/docs/mfu-from-first-principles.md).

Use an operator counter only as a coverage-aware cross-check on a small unfused batch with activation checkpointing disabled. Fused/custom kernels may bypass its formulas. Recompute adds physical work without increasing the logical numerator. Keep the peak convention fixed and report precision changes separately.

## 2. GPU timeline gaps often come from one `.item()`

The most common bottleneck I saw on Nsight Systems / HTA timelines was not a slow kernel but **idle GPU time** ("host wait"): the GPU finished its queue and waited for the CPU to enqueue more.

A frequent cause with variable-length (packed) attention: computing `max_seqlen` from the `cu_seqlens` tensor with `.item()` inside every attention layer. Each `.item()` is a device→host copy plus a synchronization, so the CPU stops enqueuing until the GPU catches up, then the GPU idles while the CPU enqueues again — once per layer.

Fix: compute sequence metadata **once per batch on the CPU** (in the data collator, where it's already known) and pass it down. In general: grep the training step for `.item()`, `.cpu()`, `.tolist()`, `print(tensor)`, and Python control flow on tensor values. ([Note 05](https://github.com/vInThePrism/gpu-kernel-lab/blob/main/notes/05-async-is-removing-false-dependencies.md) — each of these is an implicit sync point.)

## 3. JIT-compiled kernels: put the cache on local disk

Newer attention kernels written in Python DSLs (CuTe DSL, Triton) compile on first use and cache the binaries. With the cache on a **shared network filesystem**, every rank on every node compiles or reads through NFS at startup and on every new shape, which showed up as long, variable stalls before the first steps. Pointing the cache to **node-local disk** removed them. Also warm the cache with the shapes you'll see, since varlen workloads keep producing new ones.

## 4. Sequence packing and the attention mask path

When adding a new attention backend to a training framework that supports sequence packing, check **which mask path** packed batches take. If the framework still builds a dense `[batch, 1, seq, seq]` mask for packed sequences instead of routing to the varlen path (`cu_seqlens`), memory grows as `seq²` and long-context runs OOM even though the kernel itself is memory-efficient. The bug lives in the collator / dispatch code, not in the kernel.

## 5. When a fix moves the OOM, the profile changed

Fusing the LM head with cross-entropy (never materializing `[tokens, vocab]` logits in FP32) is often the single largest activation-memory saving. After enabling it, an OOM that used to happen at the loss can reappear somewhere else (e.g. in MoE dispatch buffers). That isn't the fused kernel failing; the memory peak moved to the next-largest consumer. Re-read the memory timeline after every memory optimization.

## 6. Offloading only pays if PCIe time can hide

CPU offload of activations or optimizer state trades GPU memory for PCIe transfers. It was a net loss whenever transfer time exceeded the compute it could overlap with, which is easy to estimate beforehand: bytes moved per step / PCIe bandwidth vs. step compute time. Same arithmetic as `copy + max(compute, IO)` in note 05.

## 7. Check the data before the kernels

A NaN eval loss in a run with brand-new kernels looks like a numerics bug. In my case it was one malformed sample in the eval split. Bisect over data before bisecting over kernels.

## 8. A/B everything with one variable changed

Attention backend comparisons are only meaningful with identical data order, packing, sequence lengths, batch size, and warm caches. Log every run (W&B or equivalent), keep the baseline run alive in the same dashboard, and report throughput per GPU along with MFU, since MFU alone hides changes in the FLOP count.
