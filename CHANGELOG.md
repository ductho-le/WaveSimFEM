# Changelog

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking changes

- Mesh resolution is supplied explicitly as `mesh.nx` and
  `mesh.ny_per_region`, with one vertical count per region in bottom-to-top
  order. Material properties and frequency settings no longer refine the mesh.
  Removed `elems_per_wavelength`, `min_elems_x`, and `min_elems_y`, including
  regional overrides; legacy inputs raise migration guidance.
- Explicit simulations require a user-supplied `solver.CFL` in `(0, 1]`.
  The time step is calculated from the assembled stability bound, then rounded
  down to fit the recording grid. The validated `solver.dt` override remains
  available. Implicit Newmark retains its accuracy-based time-step control.
- Preflight reports actual cell dimensions (`h_x`, `h_y_per_region`) in place
  of automatically selected `h_target` and `lambda_min`. `mesh.max_frequency`
  continues to control the analysis/sampling band and implicit accuracy.

### Changed

- Rename the CLI launcher executable to `gwfem` for consistent and concise
  simulation invocation.
- Standardize all figure exports at 190 mm width with 10-point text,
  including ticks, legends and colourbars. Compact panel layouts replace figure
  headers and explanatory annotations; accompanying README files retain the
  interpretation and provenance. `output.usetex` defaults to `false`, using
  Matplotlib's bundled serif fonts and mathtext without an external TeX install.
  Set it to `true` to require LaTeX/dvipng/Ghostscript, or `auto` to use LaTeX
  when the toolchain is available and built-in rendering otherwise.
- Add default structure-aware multi-panel reports in `wave_physics/`, selected
  from archived geometry and material provenance. Free plates, halfspaces,
  viscoelastic stacks, layered ground and solid–fluid systems receive applicable
  references and distinct diagnostics, with JSON/CSV/NPZ measurement exports.
- Phase-velocity imaging screens weak, aliased and ambiguous peaks; polarization
  requires coherent components. Spatial decay uses untapered Fourier amplitudes
  to avoid arrival-dependent window bias. Six explicitly synthetic structure
  previews are available through `tools/preview_structure_reports.py`.
- Post-processing figures share consistent typography, physical units and
  annotated amplitude scales. Dispersion handles arrays left of the source,
  honours recorded geometry, caps display bounds at sampling limits and retains
  slow modes when choosing automatic wavenumber bounds.
- Packet reports screen incomplete or ambiguous arrivals, resolve peak times
  below sample spacing, qualify velocity/decay fits and export fit residuals,
  preprocessing and per-receiver exclusion reasons (JSON schema v2).
- Lamb reports show both propagation directions, sampling/resolution margins,
  usable bands, shared-scale input/mode/residual maps and reconstruction
  diagnostics. Report exports are atomic; analytical previews can be generated
  with `tools/preview_postprocessing.py`.
- Examples and numerical tests use explicit counts preserving their previous
  resolved meshes. The resolution comparison tool now refines counts with
  `--refinement-factor` and time controls with `--step-factor`; resumed runs
  must match the requested model inputs.

## [0.3.0] — 2026-10-03 — production hardening

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
  `f_center` (`alpha` unchanged at 1 MHz) and the default duration adds the pulse
  length.
- Source and sensor geometry defaults (`excitation_width`, `source_x`,
  `sensor_offset`, `sensor_pitch`, `n_sensors`) are now `null` and derived from the
  reference wavelength `λ_ref = c_slow / f_center` (width `λ/2`, sponge gap `λ`,
  offset `2λ`, pitch `λ/8`, sensors filling the sensing region up to 1001), each
  capped by a fraction of `sensing_length`. The former fixed millimetre defaults
  (2 mm width, 5 mm gap, 50 mm
  offset, 301 sensors at 0.5 mm) were meaningless at seismic scale.
  Configurations that relied on them should set the values explicitly.
- Implicit Newmark step is accuracy-driven: `Δt = 1 / (steps_per_period ·
  max_frequency)` with the new `solver.steps_per_period` (default 40). It replaces
  the element-size/CFL rule, which made thin layers force needlessly small steps
  (example 02: 2,160 instead of about 11,200 steps). `CFL` now applies to the explicit
  solver only. `materials.implicit_time_step` changed signature accordingly and
  `explicit_time_step_heuristic` was removed.
- Sensor archives are decimated: `times` and `sensor_data_*` hold every
  `runtime.sample_stride`-th step (default 20 samples per period of
  `max_frequency`, Nyquist kept above 1.25× the plotted band, ≥ 32 samples),
  instead of every solver step. `sensors.sample_rate` sets the rate explicitly.
- Default stiffness-proportional sponge damping lowered from 1e-8 s to 1e-9 s at
  1 MHz: it contributed < 4 % of the damping at `f_center` but made the explicit
  step damping-limited (4-33× more steps in the examples). The solver now warns
  when damping, rather than wave speed, sets the explicit step.
- Example configurations replaced (see *Examples* below).

### Fixed
- Explicit RK4 step limit: `Δt_crit` now uses the RK4 half-disk stability radius
  `R = 2.6` (largest left half-disk in the RK4 stability region, 2.6156) instead
  of 1, so the step is 2.6× larger while remaining rigorous. Example 03 runs
  4,512 instead of 11,727 steps with an unchanged Rayleigh speed (2982 m/s vs
  `c_R` = 2981 m/s).
- Preflight explicit step now evaluates the same Gershgorin bound per element
  before meshing (`materials.element_operator_bounds`), so it is a guaranteed
  lower estimate of the solver step instead of a wave-speed heuristic.
- BLAS/OpenMP oversubscription under MPI: `guided-waves` and `python -m
  guidedwavefem` cap threads per rank at `cpus // ranks_per_node` unless the user
  set them (`guidedwavefem.parallel`); the setting is shown in the run banner.
- White strip in dispersion maps at the plot-limit edges (frequency/wavenumber
  masks now include the boundary bins).
- Fluid-layer detection in the analysis used an absolute `c_T < 50 m/s`
  threshold; it is now relative (`c_T < 0.01 c_L`), so slow solids such as soft
  soils are no longer misclassified.
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
- Implicit solver: `K + c₁M + c₁c₅C` is assembled once and the right-hand side
  is formed from two SpMVs with preassembled `M` and `C` (no per-step form
  assembly; 2.1× faster per step in example 02, bit-identical results). GAMG
  receives the rigid-body near-nullspace.
- Distributed explicit solver: allocation-free `PetscRK4` integrator with fused
  vector kernels; sensors are interpolated and gathered only at sample steps;
  explicitly stored zeros are dropped from the damping matrix, so its SpMV
  scales with the sponge volume.
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
- `configs/` holds documented benchmark examples covering every structure mode,
  each with real materials and references (enforced by a test):
  `01_plate_aluminum_lamb` (`1_layer`, Lamb-wave NDE),
  `02_layers_bonded_joint` (`N_layers`, Al/viscoelastic epoxy/Al, tone burst),
  `03_halfspace_steel_rayleigh` (`halfspace`, rail steel Rayleigh waves),
  `04_plate_on_halfspace_soil_bedrock` (`plate_on_halfspace`, seismic MASW at 15 Hz),
  `05_layers_bone_qus` (`N_layers`, stratified solid-fluid plate: bone marrow / cortical bone / soft tissue, Le et al. 2025),
  `06_plate_in_halfspaces_immersed_water` (`plate_in_halfspaces`, aluminium plate immersed in water).
- Removed the previous examples 01-11 (including the TOML/JSON copies of 01 and the
  BoneDC benchmark cases). TOML and JSON remain supported formats.

## [0.2.0] — numerical robustness update

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
