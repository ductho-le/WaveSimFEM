# Acoustic fluid implementation

## Contract and equations

The fluid unknown is compression-positive acoustic pressure, with stationary
positive density and isentropic bulk modulus. The solid unknown is displacement.
With the interface normal pointing out of the solid, the discrete equations are

    Ms u_tt + Ds u_t + Ks u + C p = fs
    Mp p_tt + Dp p_t + Kp p - C.T u_tt = fp.

Ms and Mp are assembled only over their physical domains. Mp integrates 1/K;
Kp integrates grad(test).grad(trial)/rho. C integrates normal solid test
displacement times fluid trial pressure. No tangential displacement is shared.

Integration by parts gives the solid boundary term `-v·sigma*n`; substituting
`sigma*n=-p*n` yields `+C*p`. The fluid outward normal on the interface is
`n_f=-n`. Its boundary term is `-w*grad(p)·n_f/rho`; the normal acceleration
condition `grad(p)·n/rho=-u_tt·n` therefore gives `-C.T*u_tt`. This fixes the
two coupling signs independently of mesh facet ordering.

The existing conforming layered quadrilateral mesh is retained. Separate parent
vector/scalar spaces are algebraically restricted to the closures of solid and
acoustic cells. Inactive rows and columns are removed, including on MPI ranks.
Internal facet normals are selected by material side, never UFL facet ordering.
Pressure is continuous across directly touching fluids, and weighted normal
flux continuity follows from the weak form. Legacy fluid regions remain elastic
regions when explicitly selected.

The previous path used one continuous vector-displacement space, symmetric
elastic operators, two-component displacement histories, and an artificial
fluid shear modulus. Those assumptions remain confined to the explicitly
selected legacy path. The acoustic path reuses mesh creation, cell tags,
sampling grids, storage, and provenance, with separate physical fields and
compatible block solvers. No additional dependency is required.

## Integration, conditions, and outputs

RK4 evaluates both coupled accelerations at each stage, using separately lumped
field masses. Its spectral stability bound includes stiffness, damping, and
mass-normalized interface coupling; there is no lagged interface update or
global coupled solve in this explicit method. Newmark uses consistent masses
and the full effective operator, including acceleration coupling. Multiplying
its fluid row by `-1/a0`, where `a0=1/(beta*dt^2)`, yields a symmetric indefinite
operator; diagonal field scaling and a field-preconditioned GMRES solve balance
the displacement and pressure units. The raw equations above are nonsymmetric.

Initial pressure rate and fluid velocity must satisfy continuity. Configured
runs start from rest; the low-level time steppers accept initial states for
verification. General initial fluid velocity is not a configuration feature.
Prescribed pressure is eliminated consistently with its first and second time
derivatives. Rigid walls impose zero normal acceleration and require zero
initial normal velocity, as supplied by the default initial state. Pressure
release and first-order outgoing radiation are defined per exterior face.
Sponge damping is numerical truncation.

Source kinds separate elastic traction [Pa], acoustic pressure [Pa], normal
acceleration into the fluid [m/s^2], and volume wave-equation forcing [s^-2].
Sensor fields carry explicit observable, component, region, and unit metadata.
Pressure is never stored under displacement names. Version 1 archives remain
readable with their legacy-fluid interpretation. Fluid velocity is reconstructed
from `-grad(integral(p dt))/rho`, with RK4 stage quadrature or Newmark trapezoidal
integration. Physical acoustic energy uses aligned `p^2/(2K)` and
`rho*|v|^2/2`, integrated only over fluid cells, and is stored in J/m for this 2D
model. Energy inside numerical sponges is not a measure of physical leakage.

## Implementation map and public API

| Files | Responsibility |
| --- | --- |
| `materials.py`, `config.py` | Physical acoustic properties, explicit legacy selection, source/sensor validation, physical-speed sizing |
| `acoustics.py` | Restricted field assembly, material-side interface normals, sources, boundaries, sensors, velocity/energy reconstruction, schema-2 archives |
| `acoustic_time.py` | Coupled PETSc Newmark/RK4 and serial SciPy/CuPy RK4 with field mass normalization |
| `solvers.py` | Dispatch from the existing build/solve/run APIs; retain the original elastic path for default solid and explicit legacy runs |
| `analysis.py`, `characterisation*.py`, `wave_reports.py`, `structure_analysis.py`, `reporting.py`, `units.py` | Physical observable labels, units, acoustic references, legacy archive compatibility |
| `tools/acoustic_reference.py`, `tools/verify_acoustics.py`, `tests/test_acoustic*.py` | Independent continuum solutions, numerical benchmarks, integration and end-to-end checks |

`build_problem` returns an `AcousticProblem` when separate physical fields or
explicit acquisition placement are needed. Both `solve_explicit` and
`solve_implicit` accept that object and return `AcousticSolveResult`; named local
sensor arrays are available through `result.histories`. Ordinary elastic calls
retain their existing result type. `run_simulation` collects MPI histories and
writes the archive and resolved configuration as before. Low-level callers own
the assembled problem and should call `problem.destroy()` when finished.

There is one configured receiver array and observable per run. A solid-fluid
interface selects the field's physical side; `sensors.region` makes the side
explicit, and is required for particle velocity at a fluid-fluid interface.
Optional VTX files use the parent mesh with zeros outside each field's physical
domain; those zeros are not additional solver unknowns.

## Backend support

| Path | Precision | Verification |
| --- | --- | --- |
| Serial SciPy RK4 | Single/double | End-to-end histories, scaling and PETSc agreement |
| Distributed PETSc RK4 | Double | One, two and four ranks; memory and disk histories |
| Serial/distributed PETSc Newmark | Double | Analytical convergence and one, two and four ranks |
| Serial CuPy RK4 | Single/double | Implemented with optional parity tests; CUDA hardware unavailable locally |
| Distributed CUDA | Unsupported | Explicit request raises before assembly |
| Implicit CUDA / single precision | Ignored with a warning | Same policy as the elastic solver (`_resolve_device`) |

Acoustic CUDA steps run on the device. The physical load is a
`SeparableForcing` (fixed vectors times `g`, `-g''`, `-g'`), uploaded once and
combined on the device at each stage; arbitrary forcing callbacks still use
per-stage host uploads. Host PETSc mirrors are downloaded lazily, only when a
state is read (sampling, snapshots, periodic finite checks). Both RK4 backends
reuse the end-of-step acceleration as the next first-stage derivative. No
acoustic GPU speedup or hardware correctness claim is made.

## Work list

- [x] Inspect current mesh, integration, material, source, and output assumptions.
- [x] Establish equations, signs, ownership, and shared configuration contract.
- [x] Implement acoustic/legacy material and configuration handling.
- [x] Assemble restricted pressure/displacement operators and interface coupling.
- [x] Implement and verify coupled Newmark and RK4, including backend behavior.
- [x] Integrate boundaries, sources, field sensors, archives, and analysis.
- [x] Verify analytic acoustic and transmission solutions and convergence.
- [x] Compare the legacy approximation in regimes where agreement is expected.
- [x] Verify oblique conversion, Scholte waves, and fluid-loaded plate dispersion.
- [x] Run solid regressions and available MPI checks; record CUDA hardware skips.
- [x] Document commands, measured errors, backend status, and limitations.
- [ ] Execute acoustic parity tests on CUDA hardware when available.

## Validation principle

The regularized elastic approximation is a comparison model, not ground truth.
Normal-incidence longitudinal results should approach the acoustic solution as
regularization and discretization errors decrease. Oblique and interface waves
can differ because the legacy space imposes tangential displacement continuity.
Independent impedance, mode-conversion, cavity, and dispersion references decide
correctness. Numerical results are reported only for commands actually executed.
See [the measured validation report](acoustic_validation.md) and
[machine-readable results](acoustic_validation_results.json). The plate benchmark
checks the nonleaky A0 branch; leaky-mode attenuation is not validated. First-order
radiation has only a normal-incidence reflection certificate here. Application
studies still require mesh, time-step, and truncation convergence.

## Environment

The installed wavesim environment provides DOLFINx 0.10.0 and PETSc 3.24.
MPI cannot initialize its local OFI socket inside the default sandbox; the
approved external Python invocation successfully imports DOLFINx and MPI.
