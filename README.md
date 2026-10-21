# Training Performance Field Notes

**LLM training performance case studies, negative results, and small diagnostic tools.**

[Find your symptom](#find-your-symptom) · [Use the tools](#use-the-tools) · [Read the evidence](#read-the-evidence) · [Contribute a case](#contribute-a-case)

A faster kernel does not always produce a faster training job. An enabled configuration flag does not prove that an optimization ran. These notes examine the gap between an intended optimization and what the system actually did.

Each case follows the same path: **symptom → hypothesis → evidence → intervention → limitations**.

## Find your symptom

| What you see | Where to start |
|---|---|
| Overlap flags are enabled, but communication remains exposed | [An overlap investigation](notes/02-cuda-device-max-connections-kills-overlap.md) |
| Training finishes, but the optimization may not be active | [Silent failures catalog](notes/03-silent-failures-catalog.md) |
| A newer attention backend makes the job slower | [Window-attention launch granularity](notes/04-window-attention-launch-granularity.md) |
| A kernel improvement shrinks at a larger scale | [Why kernel wins shrink](notes/05-why-kernel-wins-shrink-at-scale.md) |
| FP8 GEMM is faster, but the full step is slower | [Quantization break-even](notes/06-fp8-eager-quantization-breakeven.md) |
| NCCL time differs sharply between ranks | [ZeRO-3 bubbles and LL spin](notes/07-zero3-bubbles-and-nccl-ll-spin.md) |
| Prefetching appears enabled but GPUs wait for data | [Workers versus thread prefetch](notes/08-dataloader-workers-vs-thread-prefetch.md) |
| You suspect allocator fragmentation | [An allocator go/no-go investigation](notes/09-allocator-fragmentation-go-no-go.md) |
| Checkpoint saving appears fast—until the job exits | [Async checkpoint measurement](notes/10-async-checkpoint-measurement.md) |
| Good step MFU does not explain total runtime | [Estimating end-to-end MFU](notes/11-e2e-mfu-estimation.md) |

Additional cases on host sync, JIT caches, packing, offload, and recovery are in [the supplements](notes/supplements/README.md).

## Use the tools

[`perfdoctor`](tools/README.md) contains small Python utilities for trace analysis, end-to-end estimates, Amdahl-style attribution, and checks for inactive optimizations.

```bash
git clone https://github.com/vInThePrism/training-perf-field-notes.git
cd training-perf-field-notes
python -m pip install -e ./tools
python -m perfdoctor.e2e_mfu
```

The last command runs the documented estimation example on CPU. It prints an **estimated** end-to-end MFU and a wall-clock breakdown; it does not measure a live training job.

To inspect an existing Nsight Systems SQLite export:

```bash
perfdoctor-overlap trace.sqlite --device 0 --num-sms 132 --json report.json
```

Set the SM count for the actual device. The report summarizes recorded compute/NCCL kernel overlap, per-stream activity, concurrency, and gaps. Its verdicts are diagnostic hypotheses. A gap in recorded kernels is not proof that every GPU engine was idle.

| Tool | Requirement |
|---|---|
| Interval analysis, E2E estimates, attribution, log/environment checks | Python; no GPU required |
| Model/optimizer identity checks | The relevant model and optimizer objects; typically PyTorch |
| FP8 break-even probe | Compatible CUDA GPU, PyTorch, and FP8 support |

## Read the evidence

Results belong to the hardware, software versions, shapes, and measurement boundaries stated in each case. A negative result for one eager fine-tuning stack is not a universal statement about FP8.

Separate measured values from estimates and borrowed inputs. A teaching probe is not automatically a reproduction of a historical experiment. The supplied CPU tests include synthetic traces; they do not certify every Nsight export format or reproduce the cluster experiments.

The [measurement methodology](docs/measurement-methodology.md) explains timing windows, baseline comparisons, instrumentation overhead, and attribution. [Logical MFU](https://github.com/vInThePrism/logical-mfu) provides the companion FLOPs-accounting reference.

## Contribute a case

Include the environment, a precise symptom, the alternative explanations you considered, a minimal reproduction or shareable evidence, and the conditions under which the conclusion holds. Failed optimizations are welcome when they help another reader make a better decision.

## License

The original field notes and perfdoctor code are MIT-licensed. Imported material from the companion notebooks retains its original license and attribution; consult the per-file notices and repository license map.

## Continue learning and contributing

See [where new work belongs](docs/content-routing.md), [contribution guidance](CONTRIBUTING.md), [release validation](VALIDATION.md), and the [license and source map](LICENSES.md).
