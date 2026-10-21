# Estimating end-to-end MFU without running the whole job

Per-step MFU tells you how efficient a training step is. Users pay for the whole
job: initialisation, eval, checkpoints and teardown included. Running a multi-hour
job per configuration just to measure that is often not possible. This method
reuses the measured per-step numbers and measures only the *extra* time.
Implementation: `tools/src/perfdoctor/e2e_mfu.py`.

## Derivation

Per-step and E2E MFU share a definition:
`MFU = F / (T · N_gpu · P_peak)`. Having measured `MFU_step` and `T_step` in the same
steady window, the step's logical FLOPs follow without re-deriving the model:

```
F_step  = MFU_step · T_step · N_gpu · P_peak
N_equiv = N_full + Σ_p (F_partial,p / F_full)       # tail steps of each epoch
T_e2e   = T_init + N_equiv·T_step + T_eval + T_ckpt_exposed + T_final

Training-E2E MFU = MFU_step · T_step · N_equiv / T_e2e
                 = MFU_step × (training share of wall clock)

Workload-E2E MFU = (MFU_step·T_step·N_equiv + Σ_j MFU_eval,j·T_eval,j) / T_e2e
```

Peak and GPU count cancel. Report both, labelled **Estimated**. Workload-E2E can
exceed per-step MFU when eval is efficient; that does not mean training got faster.

## The double-counting rules (where most estimates go wrong)

| Item | Where it already is | Rule |
|---|---|---|
| DataLoader wait during training | inside `T_step` (the step window opens before the batch is fetched) | no separate `T_dataload` |
| Tokenisation / packing at start-up | inside `T_init` | show as a sub-item only |
| Final eval | an eval | in `T_eval`, not `T_final` |
| Sync checkpoint | pauses training | add pause × count, unless `T_step` already contains it |
| Async checkpoint | mostly in the background | add exposed stage time + final drain only |
| Recompute, comm, optimizer, logging | physical work | never in the numerator |

Hold eval and checkpoint *counts* fixed across configurations being compared.
Packing reduces the number of optimizer steps; recomputing "eval every N steps"
would quietly reduce monitoring and book it as a speed-up.

## Worked example

Qwen3-32B full SFT, 8×H800, FA3 + packing + fused CE + fused RMSNorm:

| input | value |
|---|---:|
| `MFU_step`, `T_step` | 44.31 %, 113.194 s |
| 952 packed bins, GBS 64, 3 epochs | 42 full + 3 × 0.875 → `N_equiv` = 44.625 |
| `T_init` (launch → first step) | 260.7 s |
| eval: 55 × 26.15 s (mean of 4 measured, incl. final) | 1438.0 s |
| checkpoints: 13 × 1.5 s exposed (async) | 19.5 s |
| `T_final` (terminal drain envelope) | 181.1 s |

→ `T_e2e` ≈ 6950.6 s, **Estimated Training-E2E MFU = 32.20 %** (per-step 44.31 %).
Eval alone is 20.7 % of the wall clock — the biggest lever left is evaluation policy,
not kernels.

Caveat stated in the original write-up: the async exposure (1.5 s) and the drain
(181 s) were borrowed from other measurements, not measured on this exact job. An
estimate is only as good as its least-measured input; say which ones are borrowed.
