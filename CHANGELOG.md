# Changelog

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

Numerical results intentionally differ from 0.1: integrator, sensor interpolation,
automatic mesh-sizing frequency, default implicit sponge damping and FFT windows
have changed. Existing output files are not migrated or overwritten by tests.

The regularized elastic fluid model remains an approximation. Acoustic-interface
validation and publication-scale convergence/scaling studies remain separate
scientific verification work; successful software regressions are not a substitute.
