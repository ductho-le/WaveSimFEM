# Contributing to WaveSimFEM

Thank you for your interest in contributing to WaveSimFEM! This guide covers
development setup, code quality standards, verification procedures, and the
release workflow.

## Ways to Contribute

- **Bug Reports**: Open an issue with OS, Python/DOLFINx/PETSc/CuPy versions,
  and a minimal YAML configuration to reproduce the problem.
- **Feature Requests**: Propose new physics models, boundary conditions,
  post-processing diagnostics, or backend optimizations.
- **Code & Verification**: Submit pull requests with unit/physics tests.
- **Documentation & Examples**: Improve guides, docstrings, or benchmark
  configurations in [`configs/`](../configs/).

## Development Setup

DOLFINx, PETSc, `petsc4py`, and `mpi4py` are installed from `conda-forge` via
[`environment.yml`](../environment.yml):

```bash
# Clone the repository
git clone https://github.com/ductho-le/WaveSimFEM.git
cd WaveSimFEM

# Create and activate the Conda environment (Linux & macOS)
conda env create -f environment.yml
conda activate wavesim

# Install WaveSimFEM in editable mode
pip install -e .

# Optional: install dev & release tooling (twine, packaging, pre-commit)
pip install -e ".[dev]"

# Optional: NVIDIA CUDA GPU support
pip install -e ".[cuda]"

# Install pre-commit hooks (recommended for all contributors)
pre-commit install
```

### Verify Installation

```bash
# Check package version and CLI entry point
wavesim --version

# Inspect a configuration without assembling the mesh
wavesim -c configs/01_plate_aluminum_lamb.yaml --dry-run

# Run fast local tests
python -m pytest -m "not slow"

# Run linting and formatting checks
ruff check src tests tools
ruff format --check src tests tools
```

## Code Quality

WaveSimFEM uses **Ruff** (`0.16.9`) for linting and formatting, enforced by
both pre-commit hooks and CI.

### Automatic Checks (on every commit)

Once `pre-commit install` is run, commits automatically check:

| Check | Purpose |
|-------|---------|
| **`forbid-junk`** | Blocks committing caches, build artifacts, `.DS_Store`, or coverage files |
| **Ruff linter & formatter** | Enforces style, import order, bugbear/simplify rules, and 100-char formatting |
| **File & syntax hygiene** | YAML/TOML/JSON/AST validation, LF line endings, private-key detection, file size cap |
| **`clean-junk`** | Removes regenerable caches from the working tree without touching tracked files or `results_*` |

Run all hooks manually before pushing:

```bash
pre-commit run --all-files
```

### Code Style Guidelines

- Follow PEP 8 (enforced by Ruff, `line-length = 100`, target Python 3.11+).
- Add type annotations on public functions and classes (`py.typed` is shipped).
- Preserve lazy top-level imports in `src/wavesimfem/__init__.py` so lightweight
  modules (`config`, `units`, `sources`, `analysis`, `lamb`, `characterisation`)
  never require DOLFINx, PETSc, or MPI at import time.
- Keep `ruff==0.16.9` aligned across `pyproject.toml`, `environment.yml`,
  `.pre-commit-config.yaml`, and `.github/workflows/tests.yml`.

## Running Tests & Verification

Run the full local suite before opening a pull request:

```bash
# Full local test suite (serial, including physics verification)
python -m pytest

# Fast local subset (excludes slow end-to-end physics verification)
python -m pytest -m "not slow"

# Distributed checks on 2 MPI ranks
mpiexec -n 2 python -m pytest -q -p no:cacheprovider \
  tests/test_numerics.py tests/test_acoustic_time.py \
  -k "exact_point or gather or distributed or rank_local"

# Continuum acoustic verification suite
python tools/verify_acoustics.py --case all
```

Validate packaging and isolated wheel/sdist installation:

```bash
python -m build
python -m twine check --strict dist/*
python tools/smoke_installed.py
```

### Continuous Integration

- **Pushes to `main` & Pull Requests** run lightweight CI in
  [`tests.yml`](workflows/tests.yml): Ruff lint & format checks, pure-Python &
  array backend tests, wheel/sdist build, `twine check --strict`, and isolated
  `smoke_installed.py` verification from outside the checkout.
- **Full DOLFINx / MPI / LaTeX Matrix** runs automatically every week on
  schedule, on every release tag (`v*.*.*`), and on demand via:

```bash
gh workflow run tests.yml -f full_tests=true
```

## Pull Request Process

1. Fork the repository and create a branch (`git checkout -b feat/your-feature`).
2. Keep changes focused and add/update tests in [`tests/`](../tests/).
3. Update [`CHANGELOG.md`](../CHANGELOG.md) under `## [Unreleased]` for
   user-visible changes.
4. Run `pre-commit run --all-files` and `python -m pytest`.
5. Open a pull request describing the motivation, numerical/physical impact, and
   verification performed.

## Adding Example Configurations

When contributing a new configuration under [`configs/`](../configs/):

- Follow the header, unit, and explicit-mesh (`mesh.nx`, `mesh.ny_per_region`)
  conventions documented in [`configs/README.md`](../configs/README.md).
- Verify dry-run sizing with `wavesim -c configs/<your_file>.yaml --dry-run`.
- Ensure the configuration passes `tests/test_examples.py` and uses
  `output.usetex: false` unless testing external LaTeX output explicitly.

## Releasing (Maintainers)

Releases are automated via [`.github/workflows/release.yml`](workflows/release.yml).
Pushing a semantic version tag (`vX.Y.Z`) validates metadata, runs lint and the
full DOLFINx/MPI test matrix, verifies wheel/sdist installations, publishes to
PyPI via OIDC Trusted Publishing, and creates a GitHub Release with notes
extracted from `CHANGELOG.md`.

### Release Checklist

1. **Update [`CHANGELOG.md`](../CHANGELOG.md)**: Move items from
   `## [Unreleased]` into a dated `## [X.Y.Z] — YYYY-MM-DD` section.
2. **Bump version** in [`pyproject.toml`](../pyproject.toml) (`version = "X.Y.Z"`)
   and [`CITATION.cff`](../CITATION.cff) (`version: X.Y.Z`, `date-released`).
3. **Validate the tag locally** before tagging:

   ```bash
   python tools/check_release.py --tag vX.Y.Z --notes-file /tmp/wavesimfem-notes.md
   ```

4. **Commit, tag, and push**:

   ```bash
   git add CHANGELOG.md pyproject.toml CITATION.cff
   git commit -m "Release vX.Y.Z"
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

5. **GitHub Actions** runs the full release gate before publishing to PyPI
   (requires PyPI Trusted Publishing configured for the `pypi` environment) and
   creating the GitHub Release.

### Versioning

WaveSimFEM follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (`1.0.0` → `2.0.0`): Incompatible schema or public API changes.
- **MINOR** (`0.4.0` → `0.5.0`): Backward-compatible features and physics models.
- **PATCH** (`0.4.0` → `0.4.1`): Backward-compatible bug fixes.

## Questions?

Open an [issue](https://github.com/ductho-le/WaveSimFEM/issues) or start a
[GitHub Discussion](https://github.com/ductho-le/WaveSimFEM/discussions).
