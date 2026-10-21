"""FP8 break-even probe: GEMM-only vs GEMM + dynamic quantisation (eager / compiled).

Answers "will FP8 speed up *my* training step?" in three rounds, stopping at the
first that fails (see notes/06-fp8-eager-quantization-breakeven.md):

  V1  scaled_mm with pre-quantised operands         -> FP8 ceiling
  V2  weight cached in FP8, activation quantised     -> LoRA / frozen-base best case
  V3  activation + weight quantised every call       -> full fine-tuning

Requires an FP8-capable GPU (SM89/SM90+) and torch>=2.4. Not run in CI.

    python fp8_breakeven_probe.py --hidden 5120 --inter 25600 --q-out 8192 \
        --kv-out 1024 --m 512 1632 4096 [--compile]
"""
from __future__ import annotations

import argparse

import torch

E4M3 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(E4M3).max


def quant(x: torch.Tensor):
    """Per-tensor dynamic quantisation: amax -> scale -> cast (three eager kernels)."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = FP8_MAX / amax
    return (x.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(E4M3), scale.reciprocal()


def scaled_mm(a8, a_inv, b8_t, b_inv):
    return torch._scaled_mm(a8, b8_t, scale_a=a_inv, scale_b=b_inv, out_dtype=torch.bfloat16)


def bench(fn, iters=50, warmup=10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def run_shape(m: int, k: int, n: int, use_compile: bool) -> dict:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)  # nn.Linear layout
    a8, a_inv = quant(a)
    w8, w_inv = quant(w)
    w8_t = w8.t()  # column-major view required by _scaled_mm

    def v2():
        x8, x_inv = quant(a)
        return scaled_mm(x8, x_inv, w8_t, w_inv)

    def v3():
        x8, x_inv = quant(a)
        q8, q_inv = quant(w)
        return scaled_mm(x8, x_inv, q8.t(), q_inv)

    if use_compile:
        v2 = torch.compile(v2, mode="max-autotune-no-cudagraphs")
        v3 = torch.compile(v3, mode="max-autotune-no-cudagraphs")

    t_bf16 = bench(lambda: a @ w.t())
    t_v1 = bench(lambda: scaled_mm(a8, a_inv, w8_t, w_inv))
    t_v2, t_v3 = bench(v2), bench(v3)
    return {"M": m, "K": k, "N": n, "bf16_ms": t_bf16,
            "V1_gemm_only": t_bf16 / t_v1, "V2_act_quant": t_bf16 / t_v2,
            "V3_full_quant": t_bf16 / t_v3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=25600)
    ap.add_argument("--q-out", type=int, default=8192)
    ap.add_argument("--kv-out", type=int, default=1024)
    ap.add_argument("--m", type=int, nargs="+", default=[512, 1632, 4096],
                    help="token counts; must be multiples of 16")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    h = a.hidden
    layers = {"q_proj": (h, a.q_out), "kv_proj": (h, a.kv_out), "o_proj": (a.q_out, h),
              "gate/up": (h, a.inter), "down": (a.inter, h)}
    print(f"{'layer':8} {'M':>6} {'bf16 ms':>8} {'V1':>6} {'V2':>6} {'V3':>6}"
          f"   ({'compiled' if a.compile else 'eager'})")
    for m in a.m:
        assert m % 16 == 0, "_scaled_mm needs dims divisible by 16"
        for name, (k, n) in layers.items():
            r = run_shape(m, k, n, a.compile)
            print(f"{name:8} {m:6d} {r['bf16_ms']:8.3f} {r['V1_gemm_only']:6.2f}"
                  f" {r['V2_act_quant']:6.2f} {r['V3_full_quant']:6.2f}")


if __name__ == "__main__":
    main()
