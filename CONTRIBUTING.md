# Contributing

notes/ holds cases; docs/ holds methodology; tools/src/perfdoctor contains diagnostics. Existing tests include synthetic SQLite traces. FP8 probes require matching hardware and are not covered by CPU tests.

Start with the question your change answers. Explain the mechanism and its limits, then link to the relevant implementation or primary source. Update an existing explanation when appropriate. See [content routing](docs/content-routing.md) for companion-repository boundaries.

For numerical or performance work, include input shapes, precision, reference, tolerance, device/toolchain, warm-up and timing boundaries. Separate kernel measurements from complete-job outcomes. A configuration flag alone does not establish that the intended implementation ran.

Use the existing validation path:

```bash
python -m pip install -e "./tools[dev]"
python -m pytest tools/tests -o pythonpath=tools/src
```

State checks you could not run. GPU-only paths must retain that status until exercised on suitable hardware. Keep raw traces and private data out of contributions. No extra test suite or dependency is needed for a documentation correction.

Before sharing adapted material, retain the applicable source notice from [LICENSES.md](LICENSES.md). New prose in this release uses CC BY 4.0; new source edits retain the MIT terms of the code they modify.
