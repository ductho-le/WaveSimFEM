# Changelog

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **CLI & units**: `--generate-config` requires `--force` to overwrite; `--min-frequency` / `--max-frequency` accept unit strings; ambiguous `"mhz"`/`"mHZ"` spellings are rejected; `--postprocess` combined with `--dry-run` exits with a usage error.
- **Fail-fast validation**: incompatible `lamb_modes` / `wave_characterisation` options, missing LaTeX when `usetex: true` (warn-only on `--dry-run`), and acoustic source/boundary/interface/shared-corner conflicts fail during config resolution before mesh assembly. Explicit `sensor_sample_rate` below Nyquist is raised automatically.
- **Outputs & examples**: elastic archives write `schema_version: 1`; atomic outputs preserve existing file modes or honour the umask; report READMEs reflect the active `usetex` typography mode; Examples 03–08 standardize on broadband Ricker excitation (`1 MHz` for 06–08) and consistent receiver apertures; CI expands the lightweight pure-Python/array-backend test job.

### Performance

- **On-device acoustic recording**: serial explicit runs (`CoupledArrayRK4`) evaluate sensors (`pressure`, `displacement`, `velocity`, including prescribed boundary lifting) and quadratic fluid energy on-device in a chunked ring buffer with zero per-step host syncs; `particle_velocity` interpolates only over probe cells.
- **Array & GPU kernels**: fused CuPy `ElementwiseKernel` updates (38 → 17 vector launches per RK4 step), FP32-preserving scalar casts, zero-forcing fast paths, and Stage-4 in-place accumulation across `solvers_gpu`, `CoupledArrayRK4`, and `PetscRK4`.
- **Assembly & sampling**: cached sparse `point_evaluation_matrix` (O(`n_sensors`) DOF clearing) replaces per-step `Function.eval`; `fem.Constant` form scalars allow JIT kernel reuse across sweeps; coupled Newmark factorises the monolithic operator once (MUMPS direct, field-split GMRES / CG+BJacobi/ICC fallback).

### Fixed

- **Solvers & numerics**: set `atol=0.0` in iterative PETSc KSP solvers for scale invariance on small right-hand sides; perform acoustic GLL mass lumping before Dirichlet elimination (omitting `Mp_fD` lifting for explicit prescribed pressure); build layered mesh interfaces directly from `y_bot`/`y_top`; check velocity finiteness in PETSc explicit runs.
- **Resource & MPI safety**: release PETSc temporaries and `HistoryStore` scratch directories on exception paths; avoid reporting `"single rank"` in `describe_thread_state` under unrecognized multi-rank launchers; report the first validation error in `PointSensors.gather`; enforce spec-compliant, linear-time YAML 1.1 merge precedence in `_UniqueSafeLoader` (working around PyYAML 6.0.4's `flatten_mapping` regression).
- **Post-processing & timing**: compute scalar pressure colourbar limits without stacking placeholder zeros; read sectioned `mesh` frequency bands, handle `max_frequency < f_center / 5`, and tolerate missing metadata in Lamb reports; tolerate ULP rounding on characterisation `time_window` bounds; align elastic snapshot timestamps to completed steps and share mesh-based interface tolerances.

## [0.4.0] — 2026-10-10 — acoustic fluids

### Breaking changes

- **Explicit meshes & CFL**: `mesh.nx` and `mesh.ny_per_region` (bottom-to-top) are required; removed `elems_per_wavelength`, `min_elems_x`, and `min_elems_y`. Explicit runs require `solver.CFL` in `(0, 1]`; preflight reports actual cell sizes (`h_x`, `h_y_per_region`).
- **Acoustic fluid default**: `material_model: fluid` defaults to inviscid scalar-pressure acoustics (`fluid_formulation: acoustic`, normal coupling with tangential slip) and rejects `fluid_shear_reg` (use `fluid_formulation: regularized_elastic` for legacy runs). Field-aware archives use `schema_version: 2` with explicit `sensor_observable`, `sensor_units`, and observable-specific keys.

### Added

- Acoustic sources (`acoustic_pressure`, `acoustic_normal_acceleration`, `acoustic_volume`), `source_boundary`/`source_y`/`source_region`, `acoustic_boundaries` (`rigid`, `pressure_release`, `radiation`), and `sensors.observable`/`sensors.region`.
- Coupled PETSc RK4/Newmark integrators (serial & MPI), serial SciPy/CuPy `CoupledArrayRK4`, `SeparableForcing`, coupling-inclusive explicit stability bounds, continuum verification suite (`tools/verify_acoustics.py`), and Examples 07–08.
- Structure-aware multi-panel reports (`wave_physics/`) with JSON/CSV/NPZ exports and synthetic preview tools (`tools/preview_structure_reports.py`, `tools/preview_postprocessing.py`).

### Performance & Fixed

- Coupled RK4 reuses end-of-step acceleration (4 instead of 5 evaluations/step), downloads host mirrors lazily, and evaluates fluid energy via preassembled quadratic forms.
- Aligned field-aware Gaussian source footprint to `exp(-(x - x0)² / (2σ²))` (`σ = excitation_width / 3`), matching the elastic solver; rejected mismatched `source_y` on exterior faces.
- Replaced Example 02 with an 8-ply T300/914 CFRP laminate; renamed CLI executable to `wavesim`; standardized 190 mm / 10 pt figures with `output.usetex: false` default; hardened Lamb, dispersion, and packet-characterisation reports (JSON schema v2, atomic exports).

## [0.3.0] — 2026-10-03 — production hardening

### Breaking changes

- **Strict devices & schema**: explicit `cuda`/`cuda:N` raises `DeviceUnavailableError` (exit code 3) instead of falling back; unknown/duplicate/ignored config keys and multi-layer `1_layer` definitions are rejected; `absorb_depth` applies only to halfspaces; legacy backend wrappers removed.
- **SI units & adaptive defaults**: public API, CLI flags (`--f-min`/`--f-max`/`--k-min`/`--k-max`), and `postprocess` keys use SI or unit strings; sponge damping, source/sensor layout, and duration default to wavelength/frequency-scaled values; implicit Newmark uses `solver.steps_per_period` (default 40); sensor histories are decimated (`sensors.sample_rate`); default `absorb_beta` lowered to `1e-9 s` at 1 MHz.

### Added

- Hann-windowed `tone_burst` excitation; scale-adaptive units (`wavesimfem.units`); unified SciPy/CuPy explicit RK4 engine with chunked device-resident sensor ring buffer; `logging` output (`-q`, `-v`, `--version`); `output.compress_output`/`figure_formats`/`usetex`; `substrate`/`superstrate` aliases; `tools/clean_junk.py` pre-commit hooks; curated Examples 01–06.

### Changed & Fixed

- **Solvers**: RK4 stability radius `R = 2.6` (2.6× larger rigorous step) with matching element-level preflight bounds; once-assembled implicit Newmark operator (2.1× faster, GAMG rigid-body near-nullspace); allocation-free `PetscRK4` with zero-dropped damping; vectorized `operator_bound` (~100× faster); MPI `Gatherv` sensor collection; lazy top-level imports.
- **Runtime & plotting**: automatic BLAS/OpenMP thread capping per MPI rank (`wavesimfem.parallel`); relative fluid-layer threshold (`c_T < 0.01 c_L`); isolated Matplotlib `rcParams` and clean no-TeX fallbacks; clean exit-code-2 CLI errors; hardened CI and SPDX packaging.

## [0.2.0] — numerical robustness update

- Replaced damped Verlet with fourth-order explicit RK4 using conservative assembled stability bounds and drift-free integer time stepping.
- Added exact FE point-sensor evaluation with rank-owned histories, Hann-tapered/padded dispersion FFTs, and corrected B-scan extents.
- Added preflight resource estimation (`--dry-run`), CLI numerical overrides (`--max-frequency`, `--duration`, `--dt`, `--cfl`, `--no-postprocess`), strict input validation, PETSc convergence checks, and MPI fail-fast diagnostics.
- Reorganized codebase into the `src/wavesimfem/` package with an installable CLI, regression tests, and CI.
