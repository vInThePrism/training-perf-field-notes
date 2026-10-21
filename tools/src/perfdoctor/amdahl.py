"""Why does the same kernel optimisation give +3.6 MFU points on 1 GPU and +1.1 on 8?

Per-step MFU is F / (T * N * P). If an optimisation leaves the numerator F
unchanged (it must -- see the numerator-invariant note) and saves `s` seconds
per step out of `T`, then

    MFU_new / MFU_old = T / (T - s)          and
    dMFU = MFU_old * s / (T - s)

so the gain is governed by the *share* s/T, not by s. Normalise both to
per-token time and compare two configurations: the saved time can grow (bigger
hidden size -> bigger fused GEMM removed) while the share still collapses,
because the step time grows faster (communication, ZeRO-3 gathers, recompute).

`compare()` produces the decomposition table used in the note.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    name: str
    step_s_before: float
    step_s_after: float
    tokens_per_step_per_gpu: float
    mfu_before: float

    @property
    def saved_us_per_token(self) -> float:
        return (self.step_s_before - self.step_s_after) / self.tokens_per_step_per_gpu * 1e6

    @property
    def step_us_per_token(self) -> float:
        return self.step_s_before / self.tokens_per_step_per_gpu * 1e6

    @property
    def share(self) -> float:
        return (self.step_s_before - self.step_s_after) / self.step_s_before

    @property
    def predicted_mfu_after(self) -> float:
        return self.mfu_before * self.step_s_before / self.step_s_after

    @property
    def delta_mfu_pts(self) -> float:
        return (self.predicted_mfu_after - self.mfu_before) * 100


def compare(a: Config, b: Config) -> str:
    rows = [
        ("saved  us/token", a.saved_us_per_token, b.saved_us_per_token),
        ("step   us/token", a.step_us_per_token, b.step_us_per_token),
        ("share  saved/step", a.share * 100, b.share * 100),
        ("dMFU   (pts)", a.delta_mfu_pts, b.delta_mfu_pts),
    ]
    out = [f"{'':20}{a.name:>14}{b.name:>14}{'ratio b/a':>12}"]
    for label, x, y in rows:
        out.append(f"{label:20}{x:14.3f}{y:14.3f}{(y / x if x else float('nan')):12.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    # Same kernel swap (chunked fused-linear-CE -> streaming CE, Triton RMSNorm ->
    # CuTe RMSNorm) measured on two scales; numbers from the accompanying note.
    one = Config("1-GPU 8B", 10.1968, 9.5082, 131072, 0.4943)
    eight = Config("8-GPU 32B", 115.949, 113.194, 252_900, 0.4323)  # tokens/GPU/step
    print(compare(one, eight))
