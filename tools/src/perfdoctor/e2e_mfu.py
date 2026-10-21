"""Estimate end-to-end (E2E) MFU from a measured per-step MFU.

A full training job is often too long to run just to measure E2E MFU. If the
steady-state per-step MFU and step time are already measured (same GPU type and
count, precision, model and training shapes), the logical FLOPs of a step can be
recovered without re-deriving the model formula:

    F_step = MFU_step * T_step * N_gpu * P_peak

and E2E only needs the *extra* wall time the full job spends outside steady
steps (init, eval, exposed checkpoint time, final drain/teardown).

Two estimates, always reported separately and labelled "Estimated":

  Training-E2E  numerator = training FLOPs only; eval is pure denominator cost.
                = MFU_step * T_train / T_e2e    (per-step MFU x share of wall clock)
  Workload-E2E  numerator also includes eval forward FLOPs. Can exceed per-step
                MFU when eval itself is efficient -- that does not mean the
                training step got faster.

Double-counting rules this module enforces by construction:
  * DataLoader wait inside a step is already in T_step -- there is no T_dataload.
  * Data preprocessing at start-up is inside T_init.
  * The final eval is an eval, not part of T_final.
  * Async checkpoints contribute only their critical-path exposure plus the final
    drain, never the total background write time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class E2EInputs:
    mfu_step: float              # steady-state per-step MFU, ratio in [0, 1]
    t_step: float                # steady-state optimizer-step seconds (same window)
    n_full_steps: int            # full optimizer steps in the whole job
    partial_step_fractions: Sequence[float] = ()  # F_partial / F_full for tail steps
    t_init: float = 0.0          # launcher start -> first step begins
    eval_times: Sequence[float] = ()        # wall time of every eval (incl. final)
    eval_mfus: Sequence[float] = ()         # optional, same length: eval-pass MFU
    t_ckpt_exposed: float = 0.0  # sum of checkpoint time ON the critical path
    t_final: float = 0.0         # last step end -> exit, excluding final eval


@dataclass
class E2EResult:
    n_equiv: float
    t_train: float
    t_e2e: float
    training_e2e_mfu: float
    workload_e2e_mfu: float | None
    breakdown: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"N_equiv steps           {self.n_equiv:.3f}",
            f"T_train (est)           {self.t_train:.1f} s",
            f"T_E2E   (est)           {self.t_e2e:.1f} s",
            f"Estimated Training-E2E MFU  {self.training_e2e_mfu:.2%}",
        ]
        if self.workload_e2e_mfu is not None:
            lines.append(f"Estimated Workload-E2E MFU  {self.workload_e2e_mfu:.2%}")
        lines.append("wall-clock shares: " + ", ".join(
            f"{k} {v / self.t_e2e:.1%}" for k, v in self.breakdown.items()))
        return "\n".join(lines)


def estimate(x: E2EInputs) -> E2EResult:
    if not 0 < x.mfu_step <= 1:
        raise ValueError("mfu_step must be a ratio in (0, 1]")
    for p in x.partial_step_fractions:
        if not 0 <= p <= 1:
            raise ValueError("partial step fractions must be in [0, 1]")
    if x.eval_mfus and len(x.eval_mfus) != len(x.eval_times):
        raise ValueError("eval_mfus must match eval_times one-to-one")

    n_equiv = x.n_full_steps + sum(x.partial_step_fractions)
    t_train = n_equiv * x.t_step
    t_eval = sum(x.eval_times)
    t_e2e = x.t_init + t_train + t_eval + x.t_ckpt_exposed + x.t_final

    # MFU*T is "peak-seconds of logical work"; dividing by T_e2e gives E2E MFU.
    train_work = x.mfu_step * x.t_step * n_equiv
    training = train_work / t_e2e
    workload = None
    if x.eval_mfus:
        eval_work = sum(m * t for m, t in zip(x.eval_mfus, x.eval_times))
        workload = (train_work + eval_work) / t_e2e

    return E2EResult(
        n_equiv=n_equiv, t_train=t_train, t_e2e=t_e2e,
        training_e2e_mfu=training, workload_e2e_mfu=workload,
        breakdown={"init": x.t_init, "train": t_train, "eval": t_eval,
                   "ckpt": x.t_ckpt_exposed, "final": x.t_final},
    )


def steps_for_packed_epochs(n_bins: int, global_batch: int, epochs: int) -> tuple:
    """(n_full_steps, partial_fractions) for `epochs` passes over `n_bins` packed bins."""
    full_per_epoch, rem = divmod(n_bins, global_batch)
    partial: List[float] = [rem / global_batch] * epochs if rem else []
    return full_per_epoch * epochs, partial


if __name__ == "__main__":  # worked example (numbers from the accompanying note)
    full, part = steps_for_packed_epochs(n_bins=952, global_batch=64, epochs=3)
    r = estimate(E2EInputs(
        mfu_step=0.4431, t_step=113.194, n_full_steps=full,
        partial_step_fractions=part, t_init=260.714,
        eval_times=[26.146] * 55, t_ckpt_exposed=13 * 1.5, t_final=181.123))
    print(r.summary())
