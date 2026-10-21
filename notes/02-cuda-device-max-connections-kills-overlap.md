# "overlap_grad_reduce=True" and still 0.03 % overlap: a falsification walkthrough

> Interpretation scope: this is a historical case record. The four causes can coexist. An environment setting, mixed-stream count, or grid-size ratio alone cannot prove absent overlap or SM saturation; inspect the actual scheduling dependencies and device/framework behavior. The bundled tool reports these as diagnostic observations.

**Setting.** Megatron-Core 0.15.3 driven by ms-swift 3.12.0, a 30B-A3B MoE
vision-language model, 8×H800, TP=1, EP=8, distributed optimizer,
`overlap_grad_reduce=True`, `overlap_param_gather=True`. Nsight Systems showed
communication overlapped with compute for **0.02–0.04 %** of NCCL time. NCCL was
58 s of a 158.5 s window.

**Result.** Communication and compute were on the same CUDA stream, because
`CUDA_DEVICE_MAX_CONNECTIONS=1` is set by the launcher with no TP to justify it.
The flags were working as designed; the environment variable made overlap
structurally impossible. The tool in `tools/src/perfdoctor/nsys_overlap.py`
automates the method below.

## Method: four diagnostic hypotheses, each with a measurable signal

| | Hypothesis | Measured with |
|---|---|---|
| H1 | NCCL and compute share a stream → serialised by construction | per-stream NCCL vs compute time |
| H2 | compute was *ready* but blocked by an event/dependency | strict "ready-but-not-started" ∩ NCCL busy |
| H3 | host stuck in a synchronize call → nothing submitted | sync-API intervals on the launching thread ∩ NCCL busy |
| H4 | NCCL saturates the SMs | NCCL grid size / SM count |

### What the trace said

* **H1 holds.** On every rank, 99.9 % of NCCL time sat on stream 7, which also ran
  19–27 s of compute. Stream 7 was absent from the trace's created-streams table —
  it was the default stream. Event sequence on that stream switched NCCL↔compute
  49,186 times, evenly across the window.
* **H2 rejected.** The *loose* metric ("launched but not started" ∩ NCCL) was 94 %
  — tempting and wrong: every kernel queued behind NCCL on the same stream
  satisfies it, so it is a consequence of H1, not a cause. The strict metric
  (launched **and** stream predecessor finished) was 20 %, and only **2.5 %** of the
  71.6 s GPU-idle time had ready work waiting. Event blocking would make that large.
* **H3 minor.** Host sync calls overlapped 16 % of NCCL time.
* **H4 rejected, and informative.** `AllGather_RING_LL` used **1** block,
  `SendRecv` 32 blocks, on a 132-SM GPU. NCCL occupied the stream while leaving
  75–99 % of SMs idle.

The single most telling number: **time-weighted kernel concurrency 0.558**; for
99.1 % of the window the GPU ran zero or one kernel. Not only did communication not
overlap compute — almost no two kernels ever overlapped.

## Root cause, from source

1. Megatron creates a separate communication stream only when
   `num_distributed_optimizer_instances > 1`
   (`distributed_data_parallel.py`); single-node runs have 1, so
   `param_and_grad_buffer.py` uses `nullcontext()` — the current stream.
2. **That is by design.** With one instance, collectives are issued with
   `async_op=True` and are expected to run on NCCL's *internal* stream.
3. `CUDA_DEVICE_MAX_CONNECTIONS=1` limits the GPU to one hardware work queue, so
   every stream — including NCCL's internal one — is issued in submission order.
   Megatron's own comment in the same file says CDMC=1 prevents overlap of
   communication kernels with computation.
4. ms-swift sets it unconditionally:
   `os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')` in the megatron SFT
   entry point and again in argument post-init. It is not printed in Megatron's
   argument dump, which is why "all flags are True but nothing overlaps" is hard to
   spot.

CDMC=1 exists for **tensor parallelism**, where TP collectives must be issued in
order relative to the GEMMs they feed. With TP=1 it has only the side effect. A
common config comment — "TP=1: CDMC has limited effect but is harmless" — is wrong.

Because the launcher uses `setdefault`, an explicit
`export CUDA_DEVICE_MAX_CONNECTIONS=8` overrides it without code changes. (Revert to
1 if TP>1 is ever enabled.)

## What CDMC does *not* fix

* **MoE all-to-all (52 % of NCCL time).** The token dispatcher calls
  `torch.distributed.all_to_all_single(...)` with no `async_op` — a synchronous
  point on the critical path. Freeing hardware queues cannot overlap a collective
  whose output the next kernel needs. That needs a different dispatcher (e.g.
  DeepEP with a flex dispatcher) or a schedule that interleaves micro-batches.
* **GPU-idle time (45 %).** A separate problem: 197k sub-millisecond gaps (39 s,
  launch-bound host) plus 60 gaps >10 ms (19.5 s) evenly spaced — the signature of
  periodic host-side data preparation.

Upper bound from perfect overlap was `min(NCCL, compute) = 28.8 s` (−18 %) — a
ceiling, not an expected gain.

## The tempting wrong fix: `num_distributed_optimizer_instances=2`

It does create the torch communication stream, but in this configuration:

1. **It cannot be set.** Megatron asserts
   `expert_data_parallel_size % instances == 0`. With world=8, EP=8 →
   expert-DP = 1 → assertion fails. Verified on 8 GPUs: EP=4 and EP=2 pass, EP=8 fails.
2. **It would not reach the big cost.** The communication stream is used only on the
   grad-reduce path. `start_param_sync()` (the AllGather, 48 % of NCCL time) has no
   stream context at all — it issues on the current stream with `async_op`. And the
   all-to-all is unrelated to DDP. The knob reaches ~0.2 % of NCCL time.
3. **It is not an overlap switch.** `instances > 1` means partial distributed
   optimizer (replicated shards, HSDP-like) and it flips `async_op` to `False`.

## Checklist for anyone hitting "flags on, no overlap"

1. Per-stream table: is NCCL on a stream that also runs compute? (H1)
2. `echo $CUDA_DEVICE_MAX_CONNECTIONS` *inside* the training process, not your shell.
3. Is the dominant collective issued with `async_op=True` at all? Grep the call site.
4. Look at NCCL grid sizes before blaming SM contention.
5. Report the overlap upper bound as a ceiling, and separate idle time from
   exposed communication — they have different causes.

Versions matter: this is verified for megatron-core 0.15.3 + ms-swift 3.12.0.
