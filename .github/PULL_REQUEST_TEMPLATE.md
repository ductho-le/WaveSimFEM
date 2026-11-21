## Summary

<!-- Describe what this PR changes and why. Link related issues if applicable (e.g., Fixes #12). -->

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature changing existing schema, CLI, or API)
- [ ] Documentation / example configuration update
- [ ] CI / packaging / maintenance

## Verification

- [ ] `ruff check src tests tools` and `ruff format --check src tests tools` pass
- [ ] `python -m pytest` passes locally (or `python -m pytest -m "not slow"` for non-physics edits)
- [ ] MPI / acoustic verification run if solver or parallel code was modified
- [ ] Updated `CHANGELOG.md` under `## [Unreleased]` (if user-visible)
- [ ] Updated documentation / `configs/template.yaml` (if schema or CLI changed)
