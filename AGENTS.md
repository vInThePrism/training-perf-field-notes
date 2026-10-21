# Working in training-perf-field-notes

Purpose: Training diagnosis and experimental evidence.

notes/ holds cases; docs/ holds methodology; tools/src/perfdoctor contains diagnostics. Existing tests include synthetic SQLite traces. FP8 probes require matching hardware and are not covered by CPU tests.

## Commands

Run from the repository root unless a command says otherwise.

```bash
python -m pip install -e "./tools[dev]"
python -m pytest tools/tests -o pythonpath=tools/src
```

## Editing rules

- Keep changes focused on the request. Do not add test files, test dependencies, or scaffolding unless requested or required by an existing build. Run relevant existing checks.
- Keep README commands and relative links aligned with the actual files.
- Report measured, estimated, source-reported, and unverified results accurately. Include environment and assumptions with performance claims.
- Preserve source copyright and license provenance; consult LICENSES.md before moving or adapting material.
- Never commit credentials, private paths, datasets, raw internal logs, or generated caches.
- Use docs/content-routing.md to keep the four companion repositories distinct.
