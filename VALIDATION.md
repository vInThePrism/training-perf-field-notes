# Release validation

Validated on 2026-10-21 on macOS arm64 with Python 3.14.0. The isolated validation environment used pytest 9.1.1, NumPy 2.5.4, and PyTorch 2.14.1 where needed. These are the versions exercised for this release, not a compatibility matrix.

Original private datasets, raw cluster traces, and historical run logs are not bundled. Historical measurements in notes were not independently reproduced in this release.

## Checks completed

- Editable perfdoctor installation succeeded.
- `python -m pytest tools/tests -o pythonpath=tools/src -q`: **13 passed**, including both PyTorch parameter checks.
- CLI help and the E2E estimation example execute.
- A mixed-stream counterexample confirmed that the report permits full cross-stream overlap even when NCCL shares another stream with compute.

## Boundaries

SQLite tests use a synthetic single-context export. No new production trace was supplied. The parser does not resolve multi-process correlation keys or every Nsight schema. `idle_s` means no recorded kernels, not absence of copy/memset activity. Queue-wait and grid-size observations are hypotheses rather than dependency or occupancy proofs.

The FP8 probe requires suitable NVIDIA hardware and was not run. Historical GPU-capacity labels in two case records are explicitly marked unresolved rather than guessed.
