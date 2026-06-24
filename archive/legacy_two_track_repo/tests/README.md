# Tests

This folder contains lightweight stability tests for the reusable AI decision-intelligence framework.

The current tests focus on:
- component import and execution,
- expected output columns,
- basic validation errors,
- smoke tests for starter models and workflows,
- synthetic sample-data compatibility.

They intentionally do **not** test model accuracy yet. At this stage, the goal is to keep the framework stable as the shared core and deployment tracks evolve.

Run the suite from the repository root:

```bash
python -m unittest discover -s tests
```
