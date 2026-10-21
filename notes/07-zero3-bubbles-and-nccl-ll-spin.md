# ZeRO-3 on two GPUs: the bigger loss is not exposed communication

> Hardware metadata note: the source record labels this setup H800 while reporting capacities inconsistent with a standard H800 per-device memory figure. The original hardware logs are not bundled, so the exact SKU and whether these figures are per device or aggregate remain unverified. Do not use these capacity numbers for hardware sizing.

**Setting.** Qwen3-14B full-parameter SFT, 2×H800 (143 GB), DeepSpeed ZeRO-3 with
`overlap_comm`. Steady-state window of 8.17 s from an Nsight Systems trace.

## Three findings

### 1. Communication is half-exposed, but idle time is bigger

| | Rank 0 | Rank 1 |
|---|---:|---:|
| exposed comm (not under compute) | 1.21 s (14.8 % of wall) | 2.53 s (30.9 %) |
| GPU fully idle (no kernel at all) | **3.86 s (47 %)** | 1.97 s |

Even 100 % overlap could not fill Rank 0's idle half. They are separate diseases.

### 2. The idle time is launch-bound, made of tens of thousands of small gaps

Rank 0: 30,891 idle gaps, median 28.9 µs; the 0.2–1 ms bucket holds 74 % of idle time;
only 66 gaps exceed 1 ms. ~41,000 `cudaLaunchKernel` per iteration per GPU. NVTX shows
where the host time goes: ZeRO-3's per-module Python hooks — `fetch_sub_module`
216k calls / 30 s, `release_sub_module` 211k / 12 s, pre-backward hooks 72k / 9 s
over the run. The GPU finishes a kernel and waits for Python to issue the next.

### 3. The asymmetry between ranks is NCCL LL spin-waiting, not more data

Rank 1's AllGather kernels took 4.40 s vs Rank 0's 1.47 s in the same window (35 s vs
15 s over the run), though data-parallel ranks move the same bytes. The kernels are
`*_RING_LL`: the **low-latency protocol busy-polls a flag on the GPU**, so the rank
that reaches the collective first spins *inside the kernel*, and the wait is billed as
kernel duration.

Consequences for analysis:

* Per-rank "communication kernel time" overstates transfer volume; the true transfer
  is closer to the shorter rank's number.
* The spin still occupies the stream and does no compute, so counting it as
  critical-path stall is correct.
* Asymmetry is itself a signal: something desynchronises the ranks every step.

Here, the desync came from **allocator cache flushes**: memory was at the ceiling,
and PyTorch reported 1–2 `cudaFree`-everything flushes per step, at different times on
each rank.

## The bucket-size trade-off this exposes

`overlap_comm` hides all-gather/reduce-scatter behind backward compute, but larger
buckets keep more communication buffers resident:

| ZeRO-3 comm config | median step | tokens/s | peak mem/GPU | flushes/run |
|---|---:|---:|---:|---:|
| default (bucket ≈ 26M) | 6.742 s | 4847 | 134.4 GiB | 0 |
| bucket 5e8 + max-live 5e8 | **6.706 s** | **4887** | 130.1 GiB | 2 (warm-up) |
| bucket 2e9 + max-live 2e9 | 7.983 s | 4100 | 139.8 GiB | 19 |

2e9 pushed peak memory to the ceiling; each flush forces synchronisation and
breaks stream queues, costing more than the overlap gains (−15 %). 5e8 kept memory
*below* the default (max-live caps resident params) and the overlap gain survived.

## What to do, in order

1. Treat idle (launch-bound) and exposed communication as separate budgets; measure
   both from the trace (`perfdoctor-overlap` prints both).
2. Get off the memory ceiling first — flushes amplify both problems.
3. Size buckets against memory headroom, not "bigger = more overlap".
4. For the launch-bound part on two GPUs, structural options beat tuning:
   ZeRO-2 with a larger batch if it fits, or fewer, larger modules per hook.
