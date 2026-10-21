# When an operator counter loses fused work

An operator counter can miss fused kernels or custom launches. With step time and peak fixed, under-counting FLOPs makes reported MFU **fall**. A genuine kernel speedup can change time, but must not change the logical work assigned to the same model, training state, and batch.

The source investigation reported about a 370× gap between a fused-LoRA operator count and a logical reference. Its original traces are not bundled here. Treat that number as a historical observation, not a reproduction supplied by these tools.

Use the canonical [MFU derivation](https://github.com/vInThePrism/logical-mfu/blob/main/docs/mfu-from-first-principles.md) and the companion [counter-coverage validation note](https://github.com/vInThePrism/logical-mfu/blob/main/docs/operator-counter-validation.md). Compare an unfused small case with checkpointing off before trusting counter coverage. Then verify that toggling a semantically equivalent fused implementation leaves the logical numerator unchanged.
