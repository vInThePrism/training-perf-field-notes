# perfdoctor

Small, dependency-light diagnostics for LLM training performance work. Pure
Python standard library for everything except the torch-based guards.

```bash
pip install -e ".[dev]"
pytest            # 13 tests; the nsys tests run on a synthetic SQLite trace
```

## `perfdoctor-overlap` — why isn't communication overlapping?

```bash
nsys export --type sqlite -o trace.sqlite trace.nsys-rep
perfdoctor-overlap trace.sqlite --device 0 --num-sms 132 --json report.json
```

Reports four diagnostic observations (same-stream
serialisation, event blocking, host sync, SM saturation), and reports time-weighted
kernel concurrency, the recorded-kernel gap histogram and the overlap upper bound.
Background: [`../notes/02`](../notes/02-cuda-device-max-connections-kills-overlap.md).

H2 uses the strict definition of "ready" (launched **and** stream predecessor
finished). The loose one is satisfied by every kernel queued behind NCCL on the same
stream, so it would just restate H1.

## `perfdoctor.e2e_mfu` — E2E MFU from per-step MFU

```python
from perfdoctor.e2e_mfu import E2EInputs, estimate, steps_for_packed_epochs
full, partial = steps_for_packed_epochs(n_bins=952, global_batch=64, epochs=3)
print(estimate(E2EInputs(mfu_step=0.4431, t_step=113.194, n_full_steps=full,
                         partial_step_fractions=partial, t_init=260.7,
                         eval_times=[26.15] * 55, t_ckpt_exposed=19.5,
                         t_final=181.1)).summary())
```

Background: [`../notes/11`](../notes/11-e2e-mfu-estimation.md).

## `perfdoctor.amdahl` — why a kernel's MFU gain shrinks at scale

Background: [`../notes/05`](../notes/05-why-kernel-wins-shrink-at-scale.md).

## `perfdoctor.silent_checks` — guards for optimisations that silently don't apply

| function | catches |
|---|---|
| `assert_backend(log, "fa3")` | attention backend fell back (e.g. FA3 → SDPA) with only a warning |
| `optimizer_coverage(model, opt)` | parameters replaced after optimizer creation (orphaned) |
| `muon_eligible(named_params)` | ZeRO-3 shards defeating `ndim`-based Muon routing |
| `overlap_env_warnings(tp, overlap)` | `CUDA_DEVICE_MAX_CONNECTIONS=1` with TP=1 |
| `KernelSlots` | releasing a fused-kernel slot before the replacement is eligible |

Background: [`../notes/03`](../notes/03-silent-failures-catalog.md).

## `probes/fp8_breakeven_probe.py`

GEMM-only vs GEMM + dynamic quantisation (eager or compiled) on your model's
projection shapes. Needs an FP8 GPU; not run in CI.
Background: [`../notes/06`](../notes/06-fp8-eager-quantization-breakeven.md).

## Trace interpretation limits

The kernel table does not include copy or memset engine activity: idle_s means time without recorded kernels inside the first-to-last kernel window. Non-NCCL kernels are grouped as compute. H2 is a queue-wait proxy and does not resolve cross-stream event dependencies. Grid blocks divided by SM count is not occupancy. Interpret verdicts as leads and inspect real traces; synthetic tests do not certify every Nsight schema. Use one process/context per input trace; runtime joins use correlationId and stream IDs without a multi-process context key.
