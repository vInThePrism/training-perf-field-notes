# Field notes: bugs that looked like something else

Short write-ups of real problems from GPU training jobs. Each has the symptom, what it really was,
and the fix. Generalized; no proprietary code or paths.

---

### The faster attention kernel that changed nothing
**Symptom:** switched a vision tower to FlashAttention-3; end-to-end step time barely moved.
**Cause:** window attention with 64-token windows, and the adapter launched one kernel per window
(~1M launches in 10 steps). Launch-bound: the GPU was waiting for the CPU.
**Fix:** one varlen call with `cu_seqlens` for all windows. ~1M → 10K launches, attention time
8.8 s → 4.0 s. See [`kernels/window_attention_varlen`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/demos/window_attention_varlen).
**Lesson:** check call granularity before kernel speed.

### Host wait from `.item()` inside the model
**Symptom:** large `host_wait` share in Holistic Trace Analysis, spread across every layer.
**Cause:** the varlen attention path computed `max_seqlen` with `.item()` (a device → host sync) in
every layer, so the CPU could never run ahead of the GPU.
**Fix:** compute `cu_seqlens` and `max_seqlen` once per batch (in the collator, on the CPU) and pass
them down. In general: grep the hot path for `.item()`, `.tolist()`, `.cpu()`, `nonzero`, and
Python `if` on tensor values.

### OOM only when sequence packing is on
**Symptom:** a new attention backend trains fine on padded batches, runs out of memory with packing.
**Cause:** the packed batch was routed to a code path that builds a dense 4D attention mask
(`[B, 1, S, S]`, O(S²) memory) instead of passing segment boundaries to the varlen kernel.
**Fix:** make packed batches take the `cu_seqlens` path; never materialize a 4D mask for a kernel
that does not read it.

### JIT compile that took minutes, but only on the cluster
**Symptom:** a CuTe DSL kernel compiled quickly on a dev box and very slowly in multi-node jobs.
**Cause:** the JIT cache lived on NFS; every rank compiled and read/wrote cache files over the
network file system at the same time.
**Fix:** put the JIT cache on node-local disk (optionally warm it once per node).

### Tiles that worked on H100 and crashed on H800
**Symptom:** Triton `OutOfResources: shared memory` for a fused LoRA GEMM on H800.
**Cause:** tiles tuned on H100 (BLOCK_N=256, 4 stages) needed 245,760 B of shared memory; the limit
was 232,448 B. The library's fallback halved BLOCK_N for *every* shape, which fixed the crash and
slowed everything.
**Fix:** co-search BLOCK_N and num_stages per projection shape under the real limit. A representative
M=8192, N=3584, K=18944 kernel went 2.089 ms → 1.721 ms (1.21×).
See [`tools/triton_smem_budget`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/tools/triton_smem_budget).

### An OOM whose stack trace kept moving
**Symptom:** after enabling a fused cross-entropy, a MoE model ran out of memory, at a different line
in the MoE forward than before.
**Cause:** the fusion changed *when* memory was freed, so the allocation that finally failed moved.
The line in the trace was the victim, not the cause.
**Fix:** look at the memory timeline (PyTorch memory snapshot) for the true peak, not at the line
that failed.

### Prefetching that never prefetched
**Symptom:** GPU data-wait share of 26% even though image prefetching was "enabled".
**Cause:** the download thread pool was created in the main process; DataLoader workers are forked,
and threads do not survive `fork`. Prefetch hit rate was zero.
**Fix:** create pools inside each worker (`worker_init_fn`), use pinned memory with non-blocking
H2D copies, and prefetch a fixed number of batches per worker. Data-wait 26% → 7%, effective-token
throughput +26%.

### Eval loss = NaN, train loss fine
**Cause:** one eval sample had every label masked (`-100`), so the token-mean loss was 0/0.
**Fix:** count loss tokens per batch; skip or assert on zero.

### Two kernel libraries, one DSL
**Symptom:** installing a MoE kernel library broke a norm kernel library.
**Cause:** both depended on `nvidia-cutlass-dsl`, pinned to different versions.
**Fix:** find a version both accept, or vendor/isolate one; pin it in the image and test the image on
every target GPU (consumer and data-center cards behave differently).

### Training hangs after a crash and restart
**Cause:** stale processes from the previous run still held GPUs and the rendezvous port, so the new
NCCL group waited forever.
**Fix:** kill leftovers before relaunch (check `nvidia-smi` on every node), set NCCL timeouts, and
run with `NCCL_DEBUG=INFO` when a hang reproduces.

### "Capacity factor" in an MoE config
**Question:** what capacity factor do public deployments of a popular MoE VLM use?
**Answer:** none: they are dropless (`capacity_factor=None`). Setting one drops tokens, which is a
*lossy* change to the experiment. If you use it for speed, report it as lossy.
