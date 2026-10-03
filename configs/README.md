# Example configurations

One runnable example per supported structure mode, each with materials and
dimensions taken from a real application. The header of every file explains the
physics, the application, the modelling choices, and references.

| # | File | `structure.mode` | System | Field | Excitation | Scale |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 01 | [`01_plate_aluminum_lamb.yaml`](01_plate_aluminum_lamb.yaml) | `1_layer` | 1 mm aluminium plate in vacuum | NDE / SHM | Ricker, 1 MHz | mm, μs, MHz |
| 02 | [`02_layers_bonded_joint.yaml`](02_layers_bonded_joint.yaml) | `N_layers` | Al / 0.2 mm viscoelastic epoxy / Al bonded joint | Aerospace bond inspection | 5-cycle Hann tone burst, 300 kHz (implicit solver) | mm, μs, kHz |
| 03 | [`03_halfspace_steel_rayleigh.yaml`](03_halfspace_steel_rayleigh.yaml) | `halfspace` | Rail steel halfspace | Rail / surface-crack NDE | Ricker, 1 MHz | mm, μs, MHz |
| 04 | [`04_plate_on_halfspace_soil_bedrock.yaml`](04_plate_on_halfspace_soil_bedrock.yaml) | `plate_on_halfspace` | 10 m soft soil on bedrock | Seismic MASW / site characterisation | Ricker, 15 Hz | m, s, Hz |
| 05 | [`05_plate_in_halfspaces_bone_qus.yaml`](05_plate_in_halfspaces_bone_qus.yaml) | `plate_in_halfspaces` | Anisotropic viscoelastic cortical bone between soft tissue and marrow (fluids) | Biomedical QUS (axial transmission) | Ricker, 500 kHz | mm, μs, kHz |

Together they cover every material model: `isotropic` (01–04), `viscoelastic`
(02, 05; anisotropic stiffness in 05), and `fluid` (05).

## Preflight sizes (`--dry-run`, default settings)

| # | Q2 elements | DOFs | Duration | Notes |
| :-- | --: | --: | --: | :-- |
| 01 | 9.2 k | 80 k | 118 μs | Quick tutorial case |
| 02 | 12.6 k | 105 k | 90 μs | Implicit Newmark: the viscous adhesive would cap the explicit step |
| 03 | 36.4 k | 293 k | 31 μs | Truncated halfspace with a bottom sponge |
| 04 | 12.5 k | 101 k | 800 ms | Same solver at seismic scale; reduced sponge `beta` (high Vp contrast) |
| 05 | 82.4 k | 662 k | 55 μs | Largest case; `mpiexec -n 4` recommended |

The explicit time step is finalised from the assembled operator bound and is
usually smaller than the preflight estimate, so expect more steps than shown.

## Running

```sh
# Inspect mesh, DOFs and duration without solving
guided-waves -c configs/04_plate_on_halfspace_soil_bedrock.yaml --dry-run

# Serial run
guided-waves -c configs/01_plate_aluminum_lamb.yaml

# Parallel run with CLI overrides (no file edits)
mpiexec -n 4 guided-waves -c configs/05_plate_in_halfspaces_bone_qus.yaml \
    --max-frequency 800e3 --duration 40e-6

# Re-plot an existing archive with different limits (SI numbers or unit strings)
guided-waves --postprocess results_01_plate_aluminum_lamb/sensor_data.npz \
    --f-max "1.5 MHz" --k-max "2.5 rad/mm"
```

## Writing your own

Start from `guided-waves --generate-config my_case.yaml` or copy the example
closest to your structure. Useful conventions:

- **Units:** all inputs are SI. Dispersion plot limits (`postprocess.f_min`,
  `f_max`, `k_min`, `k_max`) also accept strings such as `"2 MHz"`, `"40 Hz"`,
  `"3 rad/mm"` or `"1.2 rad/m"`; `null` selects them automatically.
- **Excitation:** `source_type: ricker` (broadband, best for dispersion maps) or
  `source_type: tone_burst` with `n_cycles` (narrowband, mode-selective; the NDE/SHM
  standard). The default duration adds the pulse length to the travel time.
- **Sponges:** `absorbing_boundaries.alpha/beta: null` scale with `f_center`, so the
  same damping profile works from seismic to ultrasonic frequencies. Sponges should
  be a few wavelengths wide at `f_center`. If the solver warns that damping limits
  the explicit step, lower `beta` (as in example 04) or use the implicit solver.
- **Viscoelastic layers:** `Cv = eta * C` with `eta = tan(delta) / (2 pi f0)` gives
  a loss factor `tan(delta)` at `f0` (see example 02). Strong viscosity favours the
  implicit solver, because Kelvin-Voigt damping grows with the mesh frequency.
- **Halfspaces:** `halfspace`, `substrate` (bottom) and `superstrate` (top) take a
  `thickness` (truncation depth) and an `absorb_depth` (sponge inside it).

## Numerical notes

1. **Explicit solver:** classical RK4 with lumped mass; the critical step comes from
   absolute row sums of the assembled stiffness and damping operators.
2. **Mesh sizing:** elements resolve the slowest physical wavelength at
   `mesh.max_frequency` (default `3 * f_center`) with `elems_per_wavelength`
   elements; override with `--max-frequency` for quick evaluations.
3. **Fluid formulation:** `fluid` is a regularised small-shear elastic model
   (`mu_f = 1e-8 K_f`) with full tangential continuity, not mixed inviscid
   acoustics. Scholte/Stoneley interface waves and leaky-wave attenuation are
   approximate; treat fluid-loaded results qualitatively unless validated.
4. **Truncated halfspaces:** body waves longer than the bottom sponge (for example
   bedrock P-waves in example 04) are only partially absorbed.
5. **Strict schema:** unknown keys and constitutive fields not used by a region's
   `material_model` are rejected (e.g. `Cv` requires `viscoelastic`).
