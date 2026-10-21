# A hand-written prefetch thread pool vs DataLoader workers: 3× and why

**Setting.** Multimodal SFT (Qwen2.5-VL-3B, LoRA r=8, 1×H800) on a dataset that
stores images as URLs, downloaded lazily (~14 ms per image). A fork of the
fine-tuning framework had a custom `BackgroundPrefetchService`: a
`ThreadPoolExecutor` in `Dataset.__getitem__` that downloads upcoming images into a
cache.

## The conflict: fork does not copy threads

With `dataloader_num_workers > 0`, PyTorch forks worker processes and runs
`__getitem__` there. The thread pool and its queue were created in the parent.
**A forked child has only the thread that called `fork()`**, so in every worker the
prefetch pool had zero threads: hit rate exactly 0, every image fetched
synchronously. The custom prefetcher only works with `num_workers=0` — so this parent-created
pool cannot be reused unchanged with forked workers. Either use worker-based loading
or construct a separate thread pool within each worker; workers and threads are not
inherently mutually exclusive.

## The A/B (120 steps, identical everything else, caches cleared before each run)

| | threads (`num_workers=0`, 8 prefetch threads) | DataLoader (4 workers, `prefetch_factor=2`, persistent) |
|---|---:|---:|
| runtime | 330.2 s | **110.3 s (3.0×)** |
| mean step | 2752 ms | 919 ms |
| GPU utilisation | 23 % | 59 % |
| images downloaded | 2819 | 1513 |
| final train loss | 0.1316 | 0.1348 |

Memory behaviour was the same (mmap/munmap counts nearly identical, same GPU peak);
the difference is entirely *time*.

## Why: the GIL

One process, one GIL. Network waits release it (so downloads *look* parallel), but
image decode/resize/to-tensor is CPU-bound Python/C that holds it. With 8 decode
threads and the training loop competing for one lock, the main thread repeatedly
fails to acquire the GIL, kernel issue stalls, and the GPU waits. DataLoader workers
are separate interpreters with separate GILs: true parallel decode, and the main
process only pops ready tensors.

The thread version also downloaded 86 % more images: speculative prefetch plus LRU
eviction threw away images before they were used.

## Recommendation

Delete the custom prefetcher; set `dataloader_num_workers = min(cores per GPU, 4–8)`,
`prefetch_factor` 2–4 and `persistent_workers=True`. The framework default of
`num_workers=0` means data loading runs serially in the training process — for
anything with decode work, that default is a performance bug.

## Attribution limit

The recorded A/B compares the complete loading strategies, including different speculative downloads. It does not isolate GIL contention from every other cost. Image-decoding libraries may release the GIL; profile the actual decoder and process/thread behavior before attributing a new workload to the same cause.
