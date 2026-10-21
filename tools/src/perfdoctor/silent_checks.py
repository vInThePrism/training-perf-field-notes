"""Guards against training optimisations that "work" without doing anything.

Every check here comes from a real case where a run finished, loss went down,
and the optimisation under test was silently not active:

  * attention backend  -- config said fa3; a failed availability probe logged a
    WARNING and fell back to SDPA. A whole table of "FA3" results was SDPA.
  * optimizer coverage -- a float8 conversion replaced nn.Parameter objects after
    the optimizer was built; the optimizer kept stepping orphaned tensors, the
    model never trained, and loss still moved (data-order noise).
  * Muon routing under ZeRO-3 -- routing by `p.ndim >= 2` sees 1-D shards, so 0
    parameters went to Muon: the "Muon" run was pure AdamW.
  * CUDA_DEVICE_MAX_CONNECTIONS=1 with TP=1 -- may restrict useful stream
    concurrency; inspect the actual timeline and framework behavior.
  * kernel slot arbitration -- disabling the incumbent kernel (e.g. a fused CE)
    before confirming the replacement is eligible leaves *neither* active.

Torch is imported lazily; the log/env checks have no dependencies.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# ----------------------------------------------------------------- attention

BACKEND_PATTERNS = {
    "sdpa": re.compile(r"Using torch SDPA", re.I),
    "fa2": re.compile(r"Using FlashAttention-2", re.I),
    "fa3": re.compile(r"Using FlashAttention-3", re.I),
    "eager": re.compile(r"Using vanilla attention", re.I),
    "probe_failed": re.compile(r"FlashAttention-\d is not installed", re.I),
}


def backend_from_log(text: str) -> Dict[str, int]:
    """Count backend markers in a training log. Trust this, not the yaml."""
    return {k: len(p.findall(text)) for k, p in BACKEND_PATTERNS.items()}


def assert_backend(text: str, expected: str) -> None:
    counts = backend_from_log(text)
    others = {k: v for k, v in counts.items() if k not in (expected,) and v}
    if counts.get(expected, 0) == 0 or others:
        raise AssertionError(
            f"expected attention backend {expected!r}, log markers say {counts}")


def backend_from_model(model) -> Optional[str]:
    """HF models record the resolved implementation on the config."""
    cfg = getattr(model, "config", None)
    return getattr(cfg, "_attn_implementation", None)


# ----------------------------------------------------------------- optimizer

@dataclass
class CoverageReport:
    trainable_in_model: int
    in_optimizer: int
    missing_from_optimizer: List[str] = field(default_factory=list)
    orphaned_in_optimizer: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_from_optimizer and self.orphaned_in_optimizer == 0


def optimizer_coverage(model, optimizer) -> CoverageReport:
    """Compare parameter *identity* between model and optimizer.

    Call it right before the first optimizer.step(). Identity (not name or
    shape) is what matters: a module swap that rebuilds Parameters leaves the
    optimizer holding tensors no longer reachable from the model.
    """
    model_params = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    return CoverageReport(
        trainable_in_model=len(model_params),
        in_optimizer=len(opt_ids),
        missing_from_optimizer=[n for i, n in model_params.items() if i not in opt_ids],
        orphaned_in_optimizer=len(opt_ids - set(model_params)),
    )


# ----------------------------------------------------------------- Muon / ZeRO-3

def logical_shape(p) -> tuple:
    """Shape before ZeRO-3 partitioning. DeepSpeed stores it as `ds_shape`."""
    return tuple(getattr(p, "ds_shape", p.shape))


def muon_eligible(named_params: Iterable, exclude_suffixes=(".gate.weight",),
                  exclude_substrings=("embed", "lm_head", "norm")) -> Dict[str, List[str]]:
    """Split params into Muon (2-D hidden weights) vs AdamW, by *logical* shape.

    Exclusions use an exact suffix for the MoE router (`.gate.weight`) -- a
    substring match on "gate" would also kick SwiGLU's `gate_proj` out of Muon.
    Returns both the correct split and what an `ndim`-based router would do, so
    a mismatch is visible.
    """
    out = {"muon": [], "adamw": [], "ndim_router_muon": []}
    for name, p in named_params:
        excluded = name.endswith(exclude_suffixes) or any(s in name for s in exclude_substrings)
        if len(logical_shape(p)) >= 2 and not excluded:
            out["muon"].append(name)
        else:
            out["adamw"].append(name)
        if getattr(p, "ndim", len(getattr(p, "shape", ()))) >= 2 and not excluded:
            out["ndim_router_muon"].append(name)
    return out


# ----------------------------------------------------------------- env

def overlap_env_warnings(tp_size: int, overlap_requested: bool,
                         env: Optional[dict] = None) -> List[str]:
    env = os.environ if env is None else env
    w = []
    cdmc = env.get("CUDA_DEVICE_MAX_CONNECTIONS")
    if overlap_requested and tp_size == 1 and cdmc == "1":
        w.append(
            "CUDA_DEVICE_MAX_CONNECTIONS=1 with TP=1 may limit useful stream "
            "concurrency. Inspect the timeline and framework behavior before changing "
            "it; this environment value alone does not prove that overlap is absent. "
            "Some launchers use os.environ.setdefault, so an explicit export overrides "
            "their default. Preserve ordering requirements when tensor parallelism is used.")
    return w


# ----------------------------------------------------------------- kernel slots

class KernelSlots:
    """One owner per fusion slot (loss, rmsnorm, swiglu, attention, ...).

    `claim()` must be called only after the candidate has passed its own
    eligibility check; the incumbent is released atomically. This ordering is
    the whole point: releasing first and then discovering the candidate is
    ineligible leaves the slot on the unfused stock path -- slower than either.
    """

    def __init__(self) -> None:
        self.owner: Dict[str, str] = {}
        self.rejected: List[tuple] = []

    def claim(self, slot: str, candidate: str, eligible: bool, reason: str = "") -> str:
        if not eligible:
            self.rejected.append((slot, candidate, reason))
            return self.owner.get(slot, "stock")  # keep incumbent untouched
        self.owner[slot] = candidate
        return candidate

    def table(self) -> Dict[str, str]:
        return dict(self.owner)
