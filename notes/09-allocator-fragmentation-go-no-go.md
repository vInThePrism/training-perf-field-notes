# Measure fragmentation before adopting a custom GPU allocator

> Hardware metadata note: the source record labels this setup H800 while reporting capacities inconsistent with a standard H800 per-device memory figure. The original hardware logs are not bundled, so the exact SKU and whether these figures are per device or aggregate remain unverified. Do not use these capacity numbers for hardware sizing.

Static/planned allocators (e.g. STAlloc-style approaches) promise lower peak memory
by eliminating caching-allocator fragmentation. Before spending days integrating
one, measure whether there is recoverable fragmentation in *your* configuration.

**Case.** Qwen3-14B full SFT, ZeRO-3, packing, batch 1, 8k cutoff, 2×H800 (140 GB).
Batch 2 OOMed; the hypothesis was fragmentation.

## Probe

Per step: `torch.cuda.reset_peak_memory_stats()`, then record allocated peak,
reserved peak, and fragmentation = (reserved − allocated) / reserved. At train end,
dump `torch.cuda.memory_summary()` and a snapshot. Compare
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` vs `:True`.

| | caching, non-expandable | `expandable_segments:True` |
|---|---:|---:|
| allocated peak | 129.7 GB | 129.7 GB |
| reserved peak | 136.8 GB | 136.8 GB |
| steady fragmentation | 5.2 % (7 GB) | 5.2 % |
| **non-releasable (true fragmentation) peak** | **7,120 MiB** | **0** |
| reserved segments | 32 | 0 |

## Reading it

1. The 7 GB is real fragmentation (scattered across 32 segments) — but expandable
   segments already removes it, at zero cost, and the framework enables it by default.
2. Reserved peak is identical either way: at the peak moment the live tensors
   themselves need ~130 GB. Removing fragmentation helps off-peak, not at the peak.
3. The batch-2 OOM is **capacity**, not fragmentation: allocated peak is 93 % of the
   card. No allocator fixes that.

**Verdict: no-go** for this configuration. A planned allocator would remove
fragmentation that is already zero.

## Where such allocators can pay off

Fragmentation is high when lifetimes interleave irregularly: dynamic padding with
variable-length batches (which conflicts with the fixed-shape requirement such
allocators usually have), pipeline parallelism with virtual stages, interleaved
recomputation, dynamic MoE layers. To evaluate one, pick a configuration that has
fragmentation to begin with — testing on a packed, fixed-length setup is guaranteed to
show nothing.

The general rule: **write down the go/no-go metric before integrating** (here:
non-releasable memory at peak under the default allocator settings), measure it,
then decide.
