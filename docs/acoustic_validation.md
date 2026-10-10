# Acoustic formulation: numerical verification

Measurements below were produced on 2026-10-10 with the installed `wavesim`
environment (DOLFINx 0.10.0, PETSc 3.24, double precision, one MPI rank).
`tools/verify_acoustics.py` assembles the production acoustic operators.
`tools/acoustic_reference.py` contains independent continuum references and
does not import the FEM implementation. Assertions fail the command on a
tolerance violation; JSON contains the measured values rather than plot-based
judgements.

This verifies the pressure equation for a stationary, linear, inviscid fluid.
The spatial and temporal approximations retain numerical error. Agreement with
the previous regularized-elastic model is useful in its applicable regime;
the independent continuum solutions determine correctness.

## Commands and actual outcomes

From the repository root, with the FEM environment active:

```sh
python tools/verify_acoustics.py --case all --output docs/acoustic_validation_results.json
python -m pytest tests/test_acoustics.py -q
python -m pytest tests/test_acoustics.py -m 'not slow' -q
```

The full benchmark command passed all eight cases. Measured output is saved in
[acoustic_validation_results.json](acoustic_validation_results.json). The
benchmark test file contains 15 passing tests, including its slow-marked
convergence and dispersion checks. The routine subset command is supplied for
convenience. The actual Python executable used was
`/Users/ductho/miniforge3/envs/wavesim/bin/python`.
Local MPI initialization required execution outside the default process sandbox;
the installed runtime and all numerical dependencies were available.

Individual cases use `--case cavity`, `temporal`, `boundaries`, `fluid-fluid`,
`normal`, `oblique`, `scholte`, or `plate` in the same command. Matrix extraction
and SciPy reference solves in this tool require one MPI rank. This restriction
belongs to the verification harness, not the distributed production solver.

| Case / command selector | Independent reference | Measured result | Outcome |
| --- | --- | --- | --- |
| `cavity`, Q1 | Rigid square, first nonzero frequency 750 Hz | 751.205359 Hz on 16×16 cells; relative error `1.6071e-3`; refinement order 2.002 | Pass |
| `cavity`, Q2 | Same 750 Hz mode | 750.000773 Hz on 16×16 cells; relative error `1.0301e-6`; refinement order 3.991 | Pass |
| `temporal`, explicit | Exact oscillation of the lumped-mass semidiscrete cavity mode | State error `4.9048e-8` at 160 steps; refinement order 4.000 | Pass |
| `temporal`, implicit | Exact oscillation of the consistent-mass semidiscrete cavity mode | State error `4.6162e-4` at 160 steps; refinement order 1.999 | Pass |
| `boundaries`, pressure release | Rigid bottom / pressure-release top, first frequency 375 Hz | 375.000386 Hz; relative error `1.0301e-6` | Pass |
| `boundaries`, radiation | Normally outgoing 400 Hz plane wave, zero reflected wave | Reflected pressure amplitude `1.1398e-8`; field error `1.3997e-8` | Pass |
| `fluid-fluid` | Impedance reflection/transmission with different densities and sound speeds | Pressure error `1.4284e-7`; reflection error `1.1328e-7`; transmission error `1.1798e-7` | Pass |
| `normal` | Acoustic/P-wave impedance matching | Displacement error `1.2617e-6`; pressure error `2.0625e-7` | Pass |
| `oblique` | Independent fluid/P/S boundary matching at 20° | Displacement error `1.3273e-6`; pressure error `2.2152e-7` | Pass |
| `scholte` | Scholte secular equation, speed 1009.041117 m/s | Refined speed 1009.054317 m/s; relative error `1.3082e-5` | Pass |
| `plate`, `k=π rad/m` | Exact elastic layer propagator between fluid halfspaces | 224.940359 versus 224.930389 m/s; relative error `4.4324e-5` | Pass |
| `plate`, `k=π/0.65 rad/m` | Same nonleaky flexural branch | 364.154968 versus 364.142165 m/s; relative error `3.5160e-5` | Pass |

## Comparison with the existing approximation

Both formulations use the same Q2 mesh: 16 cells across, 24 cells in each of
the fluid and solid layers, and the same 400 Hz continuum boundary fields.
The reference fluid has `rho=1000 kg/m³`, `c=1500 m/s`. The solid has
`rho=2000 kg/m³`, `c_P=2500 m/s`, `c_S=1200 m/s`. The legacy run calls the
existing `build_problem` vector-displacement assembly with the explicit
`fluid_formulation: regularized_elastic` selector.

The table compares Cartesian solid displacement on **identical interior nodes**,
excluding the prescribed outer boundary and material interface. Errors are
relative Euclidean norms against the independent continuum field. The final
column compares the two numerical fields directly on those same nodes.

| Incidence | Legacy `fluid_shear_reg` | Legacy error | Acoustic error | Legacy/acoustic difference |
| --- | --- | --- | --- | --- |
| Normal | `1e-4` | `2.8239e-5` | `1.3089e-6` | `2.7276e-5` |
| Normal | `1e-6` | `2.9516e-7` | `1.3089e-6` | `1.1020e-6` |
| Normal | `1e-8` | `9.5733e-9` | `1.3089e-6` | `1.3025e-6` |
| 20° | `1e-4` | `4.1471e-2` | `1.3770e-6` | `4.1471e-2` |
| 20° | `1e-6` | `2.9557e-2` | `1.3770e-6` | `2.9557e-2` |
| 20° | `1e-8` | `2.9490e-2` | `1.3770e-6` | `2.9490e-2` |

Normal incidence agrees to about `1.3e-6` between formulations at the current
default legacy regularization. The legacy result happens to have lower
discretization error in this particular one-dimensional longitudinal test;
pressure and displacement discretizations need not have identical errors.

At 20°, the acoustic result follows the continuum solution with permitted
tangential slip. The legacy formulation differs by approximately **2.949%**
at `fluid_shear_reg=1e-8` on this mesh. Reducing the regularization alone does
not remove that fixed-mesh discrepancy. This is evidence for this resolved
acoustic benchmark, not a universal bound on the legacy approximation or a
claim about its joint mesh/regularization limit. The tiny artificial fluid
shear wavelength is not resolved by this acoustic-scale mesh.

## Interface amplitudes, signs, and energy flux

The transmission checks fit incident, reflected, and transmitted plane-wave
amplitudes to the computed interior nodal fields. The balances below use those
**fitted FEM amplitudes**; they are not merely evaluations of the reference
coefficients. They are modal checks rather than pointwise FEM stress-gradient
estimates at the interface. Incident pressure amplitude is 1 Pa and phasors
use `exp(i*k*x-i*omega*t)`.

| Check | Normal incidence | 20° incidence |
| --- | --- | --- |
| Absolute reflected pressure coefficient error | `9.1756e-8` | `1.1687e-7` |
| Normal-velocity jump / incident normal velocity | `3.3548e-7` | `3.4081e-7` |
| Normal-traction residual / incident pressure | `1.4609e-6` | `1.5307e-6` |
| Tangential traction / incident pressure | `1.5910e-7` | `1.8256e-7` |
| Reflected + transmitted energy-flux fraction | `0.9999994251` | `0.9999996335` |
| Tangential slip, m/s per Pa incident | `6.6290e-14` | `3.281602843e-7` |

The independent oblique slip is `3.281603342e-7 m/s` per Pa incident. The
two-fluid test uses `(rho,c)=(1000,1500)` and `(650,2100)` in SI units. Its
normal-velocity jump relative to the incident velocity is `2.1349e-7`, and
its measured reflected/transmitted energy sum is `1.0000000172`. These checks
exercise the density weighting in `div(grad(p)/rho)`.

The flux calculation uses physical pressure/velocity or elastic traction/velocity
plane-wave flux. It does not treat the mathematical energy of the second-order
pressure equation, involving `p_t`, as physical acoustic energy.

## Convergence and benchmark scope

- Cavity meshes use 4, 8, and 16 cells per side. Required eigenfrequency orders
  are greater than 1.8 for Q1 and 3.8 for Q2. Finest relative-error bounds are
  `2e-3` and `2e-6`, respectively. The rigid constant-pressure mode is retained:
  the finest Q2 stiffness residual on a constant vector is `6.85e-17`.
  A pure-fluid mesh has one scalar pressure field and zero solid DOFs.
- Temporal checks use 40, 80, and 160 steps over 0.83 periods, holding the
  spatial discretization fixed. Each method is compared against its own
  semidiscrete mass operator so spatial mass-lumping error is not counted as
  time-integration error. Required orders are greater than 3.9 for RK4 and 1.9
  for Newmark. Pressure time-integral errors at 160 steps are `2.5623e-8` and
  `2.5367e-4`, respectively, supporting transient velocity reconstruction.
- Scattering outer boundaries are prescribed from the continuum fields.
  These tests isolate internal interface assembly from absorbing-boundary
  error. Field and normalized interface-balance limits are `1e-4`, with the
  independent convergence checks guarding against a fortuitous coarse-grid
  match. Fluid-fluid reflection and transmission limits are `1e-5`.
- Pressure release is tested by eliminating its boundary pressure DOFs and
  measuring the mixed rigid/release resonance. The separate radiation test
  uses no sponge; its reflected amplitude must be below `1e-5`. First-order
  radiation is exact in the continuum at normal incidence. This result does
  not certify low reflection at oblique incidence or for guided evanescent
  fields.
- Scholte refinement changes `(nx, rows per region)` from `(8,32)` to
  `(12,48)`. The speed error decreases from `4.6085e-5` to `1.3082e-5`; the
  refined pressure exponential-decay shape error is `9.1695e-5`. The fluid
  and solid depths are each four horizontal standing-mode half-wavelengths.
  The reference uses infinite halfspaces; the finite truncation contributes
  a small residual shape error. Required speed and shape errors are `2e-4`
  and `3e-3`, with decreasing speed error under refinement.
- The plate is 0.2 m thick with 3 m of fluid on each face. The independent
  reference propagates the exact elastic displacement/traction state and
  imposes evanescent acoustic halfspace impedances. The measured points are
  on the **nonleaky A0 flexural branch**; both speed errors must be below
  `1e-3`. This verifies fluid loading and both interface orientations without
  conflating leakage with numerical damping. Leaky-branch attenuation is not
  established by these two points.

## Runnable examples

Both commands below were executed successfully with their configured plotting
and structure-analysis workflows enabled:

```sh
python -m wavesimfem -c configs/07_fluid_acoustic.yaml --output /tmp/wavesim_example07_validation
python -m wavesimfem -c configs/08_fluid_solid_acoustic.yaml --output /tmp/wavesim_example08_validation
```

The pure-fluid example advanced 3,417 pressure DOFs for 1,080 RK4 steps. The
coupled example (v0.4.0: aluminium on a sponge-terminated water half-space,
81 receivers over 40 mm) advanced 11,538 solid and 41,665 pressure DOFs for
10,584 RK4 steps in 39.5 s wall time (one MPI rank). Both produced schema-2
pressure archives, quick plots, pressure FFT figures, and wave-physics
reports. The pressure arrays were finite and nonzero, with shapes `(271,31)`
and `(253,81)`, final times 25 and 50 µs, and maximum recorded absolute
pressures 16,128.31 and 80,761.69 Pa, respectively. Example 07's amplitude is
after the source-footprint correction to `sigma = excitation_width / 3`
(previously 11,741.19 Pa with a footprint narrower by √2). Those amplitudes
are run checks, not independent accuracy references. The example 08 f-k map
shows the fluid-loaded A0 branch crossing the water sound line near 250 kHz.

Use `--solver implicit` to run either example with coupled Newmark integration.
Use `--no-postprocess` for a solver-only run. The benchmark evidence above is
serial CPU evidence; distributed solver checks, representative solid
regressions, precision behavior, and invalid backend requests are tested by
the separate solver and regression test files. Serial SciPy/CuPy RK4 supports
single and double precision; PETSc RK4/Newmark supports double precision on CPU.
CuPy is implemented, but its hardware tests were skipped because CUDA/CuPy was
unavailable. This document does not claim GPU numerical validation or speedup.

The physical-energy test separately reconstructs particle velocity from the
pressure time integral in a closed, undamped cavity. Total pressure-plus-kinetic
energy remains constant within a relative tolerance of `1e-9` under Newmark.
Archive and plotting tests check pressure [Pa], displacement [m], and velocity
[m/s], including anisotropic and Kelvin–Voigt solid coupling, memory/disk
histories, and the public build/solve APIs. Distributed histories on two and
four ranks agree with serial histories within relative `1e-8` for both
integrators. An injected nonfinite pressure on one rank is also verified to
raise collectively without hanging the direct solver API.

## Regression commands and outcomes

| Command | Actual outcome |
| --- | --- |
| `python -m pytest -q --maxfail=3` | **758 passed, 18 skipped**, 246.69 s; includes all slow physics tests, eight shipped examples, and MPI checks |
| `python -m pytest -q` (after the source-footprint correction and its regression tests) | **768 passed, 18 skipped**, 150.40 s |
| `python -m pytest -q` (v0.4.0: separable device forcing, stage-1 reuse, lazy mirrors, matrix energies, assembly refactor, redesigned example 08) | **777 passed, 18 skipped**, 143.48 s |
| `FI_TCP_IFACE=lo0 python -m pytest tests/test_acoustic_sources.py tests/test_acoustic_simulation.py tests/test_analysis.py tests/test_postprocessing_quality.py -q -x` | **61 passed**, 6.38 s, exit 0, after the final pressure-spectrum presentation change; includes seven additional source tests introduced after full-suite collection |
| `FI_TCP_IFACE=lo0 mpiexec -n 2 python -m pytest tests/test_acoustic_time.py -q -k distributed` | **4 passed per rank**, 27 deselected per rank, exit 0 |
| `ruff check src tests tools` | Passed |
| `ruff format --check src tests tools` | Passed |
| `git diff --check` | Passed |

The 18 skips all require CuPy/CUDA, unavailable on this machine. The full suite
includes the existing S0 plate-speed, Rayleigh-speed/sponge, A0 spatial/temporal
convergence, and anisotropic/Kelvin–Voigt regressions. The volume-source tests
verify `integral(p/K)'' = integral(q)` across fluids with different densities and
bulk moduli against independent Gaussian and tone-burst integrals. They also
reject incompatible prescribed-pressure/normal-acceleration boundary data.

An initial full-suite run stopped on three outdated example-test assumptions:
the bone example now records particle velocity, and the two new examples needed
smoke-test mesh entries. The checks now select the field from the example's
declared observable and verify its units; numerical tolerances were unchanged.
The successful complete run above includes those corrections. Full-size
examples 07/08 were also run independently, with plotting enabled, as recorded
above. Their final pressure figures were visually checked for units and layout.

The final targeted run initially passed its 61 assertions but encountered an
OFI `MPI_Finalize` error on VPN interface `utun8`. Its successful rerun above
used `FI_TCP_IFACE=lo0` to select macOS loopback; the two-rank integrator check
also exited cleanly with that setting. This is a local runtime workaround, not
a solver setting. Other platforms should use their own interface names if
the same network-selection issue occurs.
