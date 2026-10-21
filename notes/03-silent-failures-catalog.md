# Optimisations that "worked" while doing nothing — a catalogue

Every entry is a case where a training run **finished, loss went down, and the
optimisation under test was not active (or was wrong)**. None of them raised an
exception. Most were found only because the before/after numbers looked too good,
too bad, or too similar. Guards for several of them live in
`tools/src/perfdoctor/silent_checks.py`.

The general lesson: **verify the mechanism from runtime evidence (log markers,
kernel names in a trace, parameter identity), never from the config file.**

---

### 1. Attention backend: "fa3" in the config, SDPA in reality

An availability probe for FlashAttention-3 failed, logged a `WARNING`
("FlashAttention-3 is not installed."), and `return`ed without setting the
implementation; transformers then defaulted to SDPA. A full table of "FA3" results
was SDPA. In a later checkout the same config value raised `NotImplementedError`
because that branch had no FA3 dispatch at all.

Extra trap: the warning's reported line number was the line inside the *logging
helper*, not the attention module, because the logger did not set `stacklevel`.

**Guard:** after every run,
`grep -oE "Using torch SDPA|Using FlashAttention-[23]|Using vanilla attention" run.log`
and fail if it does not match the intent (`assert_backend()`).

### 2. FP8 conversion orphaned the optimizer's parameters

A float8 conversion (`Float8Linear` swap) replaces `nn.Parameter` objects. The
trainer had already built the optimizer, so it kept stepping the *old* tensors. The
model did not train; loss still moved (data order). Fix: convert before the
optimizer is constructed and make the accelerator's own conversion a no-op.

**Guard:** compare parameter *identity* between `model.named_parameters()` and
`optimizer.param_groups` right before the first step (`optimizer_coverage()`).

### 3. Muon under ZeRO-3 routed 0 parameters to Muon

DeepSpeed's flagging routes by `p.ndim >= 2`. Under ZeRO-3 every parameter is a
1-D shard, so nothing qualified: the "Muon" experiment was pure AdamW. The flagging
runs inside `deepspeed.initialize`, overwriting earlier marks, so the fix had to
replace that function and route by `p.ds_shape`. Second trap in the same code: the
MoE router must be excluded by the exact suffix `.gate.weight` — a substring match
on `gate` also excludes SwiGLU's `gate_proj`.

**Guard:** `muon_eligible()` reports both the logical-shape split and what an
`ndim` router would do.

### 4. `CUDA_DEVICE_MAX_CONNECTIONS=1` with TP=1

Overlap flags on, overlap 0.03 %. See [note 02](02-cuda-device-max-connections-kills-overlap.md).

### 5. Disabling the incumbent kernel before the replacement is confirmed

Fused kernels compete for "slots": loss (cut-cross-entropy vs Liger fused-linear-CE),
RMSNorm (CuTe-DSL vs Liger), SwiGLU (GEMM-epilogue fusion vs Liger), attention.
Liger decides its kernel set from the config alone, early. If the integration turns
Liger's loss kernel off *and then* finds CCE ineligible for this model (e.g. VL +
ZeRO-3 corner case), the run falls back to the unfused stock loss — slower than
either. Eligibility must be decided **before** the incumbent is released; I kept a
regression test for exactly this.

**Guard:** `KernelSlots.claim(slot, candidate, eligible)` never releases the
incumbent for an ineligible candidate.

### 6. Fused SwiGLU with the wrong weight layout

A GEMM-epilogue SwiGLU kernel reads `gate = pre_act[2i]`, `up = pre_act[2i+1]`.
The fused weight must therefore be **row-interleaved** `[g0, u0, g1, u1, …]`, not the
obvious concat `[gate; up]`. Concat runs, produces finite loss, and is numerically
wrong. Building it with `stack(...).reshape(...)` keeps autograd, so `dW_fused`
splits back into gate/up gradients without a hand-written backward.

### 7. The FLOP counter that went blind

Fused LoRA (base GEMM + adapter in one Triton kernel) bypasses the dispatcher:
operator-level FLOPs dropped ~370× and "MFU" read 0.08 %. Opposite failure in the
same family: the counting pass itself slows training ~2×, so its step time poisons
any MFU denominator — and the inflation differs per config, so it does *not* cancel
in an A/B. Two published speedups were wrong because of this (+14 % was really
+135 %). Count in one pass, time in another.

### 8. Autotuner IPC broken by a log line

A CuTe-DSL kernel library's autotuner uses a subprocess's **stdout as a pickle
channel**. Any start-up code that prints to stdout corrupts it:
`unpickling stack underflow`. All diagnostics in the integration had to go to stderr.

### 9. Shape-keyed autotuning × packing

Two different fused kernels autotuned per exact shape and did not persist the cache.
With packed variable-length batches, `M` changes nearly every step, so each step
re-tuned (one case: 249 s stalls; another: `torch.compile` hitting
`recompile_limit` then OOM). Verdict for both: NO-GO with packing, unless the kernel
keys autotuning on a bucketed size.

### 10. Hardware/toolchain gates that silently no-op

A Transformer-Engine block-scaling FP8 path requires CUDA ≥ 12.9 and SM ≥ 9.0;
on a cu126 environment it silently did nothing. A torchao FP8 tensor subclass
returned `data_ptr() == 0`, which broke the accelerator's FSDP2 shard remapping, so
the advertised FP8 all-gather was unavailable.

### 11. Async checkpoint that saved nothing (for LoRA)

Patching transformers' `safe_save_file` was not enough: `save_pretrained` calls it
directly, and PEFT holds its *own* imported reference, so LoRA runs captured an empty
state. Patches also had to be installed **before** the saver process forks (the child
inherits them). And "the checkpoint directory exists" ≠ "the checkpoint is complete":
resume must read a completion marker written after persistence, not test for the
directory.

### 12. Hand-written prefetch threads in a forked DataLoader

A thread-pool image prefetcher lived in the main process. With
`num_workers > 0`, `__getitem__` runs in forked workers — and **fork does not copy
threads**. Hit rate went to exactly 0. See [note 08](08-dataloader-workers-vs-thread-prefetch.md).

---

## A checklist distilled from the above

- [ ] Grep the log for the *active* backend/kernel markers; compare to intent.
- [ ] Assert optimizer ⊇ trainable params by identity before step 1.
- [ ] For ZeRO-3, never use `p.ndim` / `p.shape` for routing decisions.
- [ ] Print effective env vars from inside the training process.
- [ ] Release a fused-kernel slot only after the replacement passes eligibility.
- [ ] Check numerical equivalence of every fused kernel (forward + all grads, cosine
      and max-abs vs reference) *before* measuring speed.
- [ ] Never take timing from a run that also counts FLOPs.
- [ ] Treat any too-good or too-identical result as a bug report.
