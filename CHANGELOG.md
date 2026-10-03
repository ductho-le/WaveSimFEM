# Changelog

## 0.3.0 — production hardening

### Breaking changes
- Explicit device requests are strict: `cuda`, `cuda:N` and `mps` raise
  `DeviceUnavailableError` (CLI exit code 3) instead of silently falling back to
  the CPU; invalid device strings raise `ValueError`. `auto` still falls back.
- Configuration schema is strict: unknown keys (top-level, section, or region),
  duplicated parameters, non-boolean switches, and constitutive fields ignored by
  a region's `material_model` (e.g. `Cv` without `viscoelastic`) are rejected.
- `1_layer` mode rejects multiple layers; `absorb_depth` is only accepted on halfspaces.
- Removed `solve_explicit_rk4_torch/cupy/scipy` and the `HAS_CUPY`/`HAS_TORCH`
  globals (use `integrate_explicit_rk4` or the retained `solve_explicit_rk4_device`).
  Private helpers `_normalize_structure`/`_canonicalize_mode` are now public as
  `normalize_structure`/`structure_from_config`/`canonicalize_mode`.
- Physical units are SI throughout the public API: `dispersion_spectrum` returns
  frequencies in Hz and wavenumbers in rad/m; `plot_full_analysis` takes
  `f_min/f_max/k_min/k_max` (SI numbers or unit strings such as `"2 MHz"`,
  `"3 rad/mm"`) instead of `f_min_disp_MHz`/`k_max_disp`; the CLI flags
  `--f-min/--f-max/--k-min/--k-max` take SI numbers or unit strings.
- Postprocess keys renamed to `postprocess.f_min/f_max/k_min/k_max` (SI or unit
  strings, `null` = automatic). The legacy `f_*_disp_MHz`/`k_*_disp` keys and
  archive fields are still read and converted; giving both forms is an error.
- Defaults now scale with the problem: sponge `alpha/beta` (`null`) scale with
  `f_center` (`alpha` unchanged at 1 MHz), the default duration adds the pulse length, and
  the default `source_x` keeps the load one `excitation_width` (≥ 5 mm) from the sponge.
- Default stiffness-proportional sponge damping lowered from 1e-8 s to 1e-9 s at
  1 MHz: it contributed < 4 % of the damping at `f_center` but made the explicit
  step damping-limited (4-33× more steps in the examples). The solver now warns
  when damping, rather than wave speed, sets the explicit step.
- Example configurations replaced (see *Examples* below).

### Fixed
- `solver.device` and `solver.precision` in sectioned configs were ignored.
- A shipped example silently dropped the bone viscosity tensor (now rejected by
  the strict material schema).
- Figure titles rendered literal `\textbf{...}` when LaTeX was unavailable;
  plotting no longer mutates global `rcParams`, the Matplotlib backend, or `PATH`.
- Preflight estimates now share the solver's mesh-layout and time-step code.
- `--dry-run` without `-c` now errors instead of crashing; postprocess-only runs
  honor the plot limits stored in the archive.
- Invalid or missing configuration files and missing sensor archives now produce a
  one-line `guided-waves: error:` message with exit code 2 instead of a traceback
  (`-v` re-raises for debugging).

### Added
- Hann-windowed N-cycle tone burst excitation (`excitation.source_type:
  tone_burst`, `excitation.n_cycles`) next to the Ricker wavelet; the source type
  is recorded in the archive and shown in the preflight and quick-look figure.
- Scale awareness (`guidedwavefem.units`): logs, the preflight box, and figures
  pick units from the data (s/ms/μs, m/mm, Hz/kHz/MHz, rad/m or rad/mm), so the
  same tool serves seismic and ultrasonic problems; `parse_quantity` parses
  `"40 Hz"`-style inputs. The band-pass filter no longer has a fixed 100 Hz floor.
- `clean-junk` and `forbid-junk` pre-commit hooks backed by `tools/clean_junk.py`
  (safe removal of caches, build artifacts, and OS/editor files; staged junk is
  rejected). Ruff is pinned to the same version in pre-commit and CI.
- Unified, backend-agnostic single-process RK4 engine (SciPy/CuPy/PyTorch): stacked
  `[K|C]` operator, allocation-free stages, chunked device-resident sensor
  history, periodic non-finite detection; verified against the PETSc engine.
- Config keys `output.compress_output`, `output.figure_formats`, `output.usetex`;
  `substrate`/`superstrate` aliases; "did you mean" suggestions.
- `logging`-based console output (`guidedwavefem` logger), `-q/--quiet`,
  `-v/--verbose`, `--version`, `set_verbosity()`.
- Runtime provenance records package version, engine, device, and precision.
- Tests: configuration schema, figures, engine-vs-reference RK4, MPI `Gatherv`,
  CLI paths, and an end-to-end physics check (S0 plate velocity, ≈0.1 % error).

### Changed
- Lazy package exports: `import guidedwavefem` no longer imports DOLFINx/MPI.
- Vectorized `operator_bound` (≈100× faster on large meshes); sensor histories
  are collected with `Gatherv` instead of pickled object gathers.
- Packaging: SPDX license metadata, classifiers, URLs, `dev` extra, `py.typed`,
  version single-sourced from package metadata; stricter Ruff rule set + formatter.
- CI: least-privilege token, SHA-pinned actions, lint/test/build jobs, coverage,
  MPI and CLI smoke tests. Removed unused conda dependencies.
- Documentation: corrected GPU claims, documented device semantics, regularized
  fluid limitations, conservative explicit time step, and degree-independent
  mesh sizing.

### Examples
- `configs/` now holds exactly one documented example per structure mode, each
  with real materials and references (enforced by a test):
  `01_plate_aluminum_lamb` (`1_layer`, Lamb-wave NDE),
  `02_layers_bonded_joint` (`N_layers`, Al/viscoelastic epoxy/Al, tone burst),
  `03_halfspace_steel_rayleigh` (`halfspace`, rail steel Rayleigh waves),
  `04_plate_on_halfspace_soil_bedrock` (`plate_on_halfspace`, seismic MASW at 15 Hz),
  `05_plate_in_halfspaces_bone_qus` (`plate_in_halfspaces`, cortical bone between
  soft tissue and marrow).
- Removed the previous examples 01-11 (including the TOML/JSON copies of 01 and the
  BoneDC benchmark cases). TOML and JSON remain supported formats.

## 0.2.0 — numerical robustness update

- Replace first-order damped Verlet with fourth-order explicit RK4.
- Replace random fixed-count spectral iteration with conservative assembled
  operator bounds; preserve requested damping and fit the exact final time.
- Evaluate sensors at exact FE points; store only owned sensor histories per rank.
- Correct FFT plotting coordinates and add Hann tapering/twofold padding.
- Correct B-scan spatio-temporal pixel extents with half-cell boundaries, pulse peak offset, and adaptive axis ticks.
- Add preflight resource estimator and `--dry-run` CLI mode to preview mesh, DOF counts, time steps, and RAM before allocation.
- Add runtime numerical CLI overrides: `--max-frequency`, `--duration`, `--dt`, and `--cfl`.
- Eliminate cumulative time drift in simulation loop using integer step multiplication.
- Validate material energy, geometry, sampling and integration parameters.
- Record actual damping, mesh, step, dependency versions and source hashes via module-relative paths.
- Check PETSc convergence and fail fast on MPI runtime exceptions with detailed process failure diagnostics.
- Make `--no-postprocess` disable all plotting and honor archived plot limits.
- Add installable CLI, pinned FEM API versions, regression tests and CI.
- Reorganize codebase into a modular Python package (`src/guidedwavefem/`) with dedicated modules for configuration, materials, numerics, solvers, analysis, and CLI.

Numerical results intentionally differ from 0.1: integrator, sensor interpolation,
automatic mesh-sizing frequency, default implicit sponge damping and FFT windows
have changed. Existing output files are not migrated or overwritten by tests.

The regularized elastic fluid model remains an approximation. Acoustic-interface
validation and publication-scale convergence/scaling studies remain separate
scientific verification work; successful software regressions are not a substitute.
