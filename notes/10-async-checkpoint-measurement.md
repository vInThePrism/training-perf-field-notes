# Async checkpointing: what to measure, and what "done" means

Async checkpointing (here: DLRover Flash Checkpoint) splits a save into two phases:

1. **stage** — copy GPU state into pinned/shared CPU memory. Training is blocked.
2. **persist** — a separate saver process writes shared memory to storage.
   Training continues.

## Measure the blocked time, per save, not the total save time

14B full SFT, full checkpoint with optimizer state:

| | blocked per save |
|---|---:|
| synchronous (DeepSpeed save, serial `torch.save`) | ~135 s |
| async, first save | 13.4 s (one-time shared-memory allocation + pinning 22 GB) |
| async, steady state | **1.42–1.52 s** |

The synchronous figure was cross-checked two independent ways (timestamp delta
between consecutive saves minus a step; total runtime minus pure training time):
133 s and 138 s.

Steady-state blocking equals pure copy time **only if persist finishes before the
next save starts**. Here the save interval (~75 s) exceeded persist time (15–25 s),
so there was no back-pressure. Shrink the interval below persist time and the next
stage waits on the previous persist — report the interval alongside the number.

A second, smaller configuration (4B full SFT, 4 GPUs, ZeRO-2, 32k, 10 saves):
blocked time 370 s → 180 s (−51 %), end-to-end 731.7 s → 558.3 s (−23.7 %), with
**identical** token count and loss to 1e-4 — evidence that async saving did not alter
training. Blocking there stayed at 11–16 s per save after the first: that is the
GPU → shared-memory copy itself, which async mode cannot remove (double-buffered
staging would be the next lever).

## Correct A/B rules

* The only variable is sync vs async. Align save frequency first (one arm originally
  saved half as often).
* Both arms must save **equivalent payloads** that actually persist and restore.
  Comparing a full-state sync save with a weights-only async save is meaningless.
* In E2E accounting, async contributes its exposed stage time plus the **final
  drain** (waiting for the last persist before exit) — never the total background
  write time.

## Integration traps (the ones that cost the most time)

* **"Directory exists" ≠ "checkpoint complete."** A directory with a partially
  persisted state is the normal state while the saver works. Resume must read a
  completion tracker written after persist (`*_latest.txt`-style), not test for the
  directory.
* The official trainer integration pinned an old transformers; reusing only the
  shared-memory + async engine via a mixin kept the modern trainer lifecycle.
* `save_pretrained` calls `safe_save_file` directly, and PEFT imports its own
  reference — patch both, or LoRA runs capture an empty state.
* The saver is a forked process; install patches **before** the fork so the child
  inherits them.
* ZeRO-3's consolidated 16-bit state dict is a collective: every rank must call it.
* Non-saving ranks returning `False` is not a failure; synchronise the success flag
  with a MAX all-reduce.
