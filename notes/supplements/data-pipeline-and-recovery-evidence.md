# Data pipelines and checkpoint recovery at scale

Two places where a training job loses time without any kernel being slow.

## Data loading for multimodal SFT

**Measure the wait first.** Trace remote image reads, CPU preprocessing and host-to-device copies
separately, and report "GPU data-wait share" (time the GPU waits for the next batch / step time).

What fixed a 26% data-wait share (→ 7%, +26% effective tokens/s):

- **Thread pools must be created inside the DataLoader worker.** Workers are forked; threads started
  in the parent do not survive `fork`. A download pool created at import time silently did nothing:
  prefetch hit rate was zero.
- **A fixed amount of look-ahead:** 4 worker processes per rank × 2 prefetched batches per worker.
- **Pinned memory + `non_blocking=True` together.** Pinned (page-locked) host memory lets the copy
  engine DMA directly; `non_blocking` lets the CPU continue while the copy runs. Either one alone
  gives a synchronous copy. Too much pinned memory starves the OS, so pin buffers, not datasets.

**Cold start is its own problem.** Short jobs spent most of their first minutes parsing:

- parse Parquet shards in parallel (32 shards → 32 parallel readers),
- cache preprocessing results keyed by a **content hash** (of the data and the preprocessing config),
  so a re-run with the same data skips the work and a changed config invalidates it,
- stream tokenization for short jobs instead of tokenizing everything up front,
- measure sequence packing separately so its cost is not hidden in "loading".

Result: cold-cache load + parse 12.4 min → 2.1 min.

## Asynchronous, sharded checkpoints

A synchronous checkpoint stops every GPU while hundreds of GB go to storage. Split it in two:

1. **Snapshot (foreground):** copy GPU state to staged host memory. Must be short.
2. **Persist (background):** write the staged shards to storage while training continues.

With this split (on top of DLRover Flash Checkpoint), foreground pauses went 96 s → 3.2 s.

**Capacity-plan the staging memory.** A ~420 GB checkpoint across 4 nodes is ~105 GB of staged state per
node; it must fit in the host-memory budget (256 GB) next to the data loader and pinned buffers.

**Avoid backpressure.** If the background write takes longer than the checkpoint interval, snapshots
pile up. Track write latency at p95 (78 s) against the interval (600 s), not the mean.

**Make recovery exact.** A checkpoint is only useful if resuming reproduces the same training:

- bind model state, optimizer state, RNG states, the **data cursor** and the **packing buffers** into
  one manifest,
- commit the manifest **atomically** (write to a temp name, then rename) and only after every shard is
  durable; a crash mid-write leaves the previous manifest valid,
- test it: 50 fault-injection trials across process, shard and node failures, and check that the
  100 batch IDs after resume match the uninterrupted run exactly.

Why the packing buffer matters: a packer holds partially filled rows. Without saving it, resume skips or
repeats documents, and the loss curve after a restart is not the same experiment.
