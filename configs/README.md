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
| 05 | [`05_layers_bone_qus.yaml`](05_layers_bone_qus.yaml) | `N_layers` | Stratified solid-fluid plate: bone marrow / cortical bone / soft tissue | Biomedical QUS (axial transmission) | Ricker, 500 kHz | mm, μs, kHz |
| 06 | [`06_plate_in_halfspaces_immersed_water.yaml`](06_plate_in_halfspaces_immersed_water.yaml) | `plate_in_halfspaces` | Aluminium plate immersed in water (bilateral fluid loading) | Ultrasonic immersion NDE | Ricker, 500 kHz | mm, μs, kHz |

Together they cover every structure mode (`1_layer`, `N_layers`, `halfspace`,
`plate_on_halfspace`, `plate_in_halfspaces`) and every material model: `isotropic`
(01, 03, 04, 06), `viscoelastic` (02, 05; anisotropic stiffness in 05), and
`fluid` (05, 06).

## Preflight sizes (`--dry-run`, default settings)

| # | Q2 elements | DOFs | Duration | Steps (preflight) | Notes |
| :-- | --: | --: | --: | --: | :-- |
| 01 | 13.8 k | 120 k | 147 μs | ≈ 34 k | Mesh resolves the 2 MHz plot edge; long window for slow A0 |
| 02 | 12.6 k | 105 k | 90 μs | 2.2 k | Implicit Newmark: the viscous adhesive would cap the explicit step |
| 03 | 36.4 k | 293 k | 31 μs | 4.5 k | Truncated halfspace with a bottom sponge |
| 04 | 12.5 k | 101 k | 800 ms | 9.8 k | Same solver at seismic scale; reduced sponge `beta` (high Vp contrast) |
| 05 | 22.1 k | 179 k | 55 μs | 13.6 k | Stratified solid-fluid plate in vacuum (Le et al., 2025) |
| 06 | 63.0 k | 507 k | 55 μs | 13.4 k | Plate with bilateral acoustic sponges; `mpiexec -n 4` recommended |

The preflight explicit step evaluates the solver's Gershgorin bound element by
element, so it is never larger than the final step, and the step counts agree
up to rounding to the sensor stride (example 01: 34,082 estimated, 34,083 run;
example 03: 4,511 vs 4,512). On 4 MPI ranks example 02 takes about 40 s,
example 03 about 100 s and example 01 about 270 s.

## Running

```sh
# Inspect mesh, DOFs and duration without solving
guided-waves -c configs/04_plate_on_halfspace_soil_bedrock.yaml --dry-run

# Serial run
guided-waves -c configs/01_plate_aluminum_lamb.yaml

# Parallel run with CLI overrides (no file edits)
mpiexec -n 4 guided-waves -c configs/05_layers_bone_qus.yaml \
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

1. **Explicit solver:** classical RK4 with lumped mass; the critical step
   `2.6 / max(sqrt(B_K), B_C)` comes from absolute row sums of the assembled
   stiffness and damping operators and the RK4 half-disk stability radius.
   `solver.CFL` is the fraction of that limit actually used.
2. **Implicit solver:** unconditionally stable; `solver.steps_per_period` steps
   per period of `mesh.max_frequency` (default 40, about 0.2 % period error).
3. **Mesh sizing:** elements resolve the slowest physical wavelength at
   `mesh.max_frequency` (default `3 * f_center`) with `elems_per_wavelength`
   elements; override with `--max-frequency` for quick evaluations.
4. **Geometry and sampling defaults:** `excitation_width`, `source_x`, and the
   sensor array may be left `null`; they scale with `lambda_ref = c_slow / f_center`.
   Sensor traces are stored at 20 samples per period of `max_frequency` unless
   `sensors.sample_rate` is set.
5. **Fluid formulation:** `fluid` is a regularised small-shear elastic model
   (`mu_f = 1e-8 K_f`) with full tangential continuity, not mixed inviscid
   acoustics. Scholte/Stoneley interface waves and leaky-wave attenuation are
   approximate; treat fluid-loaded results qualitatively unless validated.
6. **Truncated halfspaces:** body waves longer than the bottom sponge (for example
   bedrock P-waves in example 04) are only partially absorbed.
7. **Strict schema:** unknown keys and constitutive fields not used by a region's
   `material_model` are rejected (e.g. `Cv` requires `viscoelastic`).
