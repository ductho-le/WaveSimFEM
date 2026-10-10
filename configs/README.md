# Simulation configurations

This directory contains application examples and a complete parameter reference
for WaveSimFEM. Each YAML file documents the physical model, numerical scope,
units, and execution command. Use the closest application as a starting point,
then establish convergence for the quantities required by your study.

## Configuration catalogue

| Configuration | Model and application | Excitation | Sizing upper frequency |
| :-- | :-- | :-- | :-- |
| [`template.yaml`](template.yaml) | Complete parameter reference; 1 mm free aluminium plate | Ricker, 500 kHz | 1 MHz |
| [`01_plate_aluminum_lamb.yaml`](01_plate_aluminum_lamb.yaml) | 3 mm aluminium plate; Lamb-wave NDE and structural health monitoring | Ricker, 1 MHz | 2 MHz |
| [`02_plate_cfrp.yaml`](02_plate_cfrp.yaml) | 1 mm T300/914 CFRP cross-ply plate, `[0/90/0/90]_s`; anisotropic laminate dispersion | Ricker, 1 MHz | 2 MHz |
| [`03_halfspace_steel_rayleigh.yaml`](03_halfspace_steel_rayleigh.yaml) | Truncated steel halfspace; Rayleigh-wave velocity measurement | Five-cycle Hann burst, 1 MHz | 1.5 MHz |
| [`04_plate_on_halfspace_soil_bedrock.yaml`](04_plate_on_halfspace_soil_bedrock.yaml) | 10 m soil layer over bedrock; seismic surface-wave analysis | Ricker, 15 Hz | 30 Hz |
| [`05_layers_bone_qus.yaml`](05_layers_bone_qus.yaml) | Marrow / cortical bone / tissue; axial-transmission ultrasound | Ricker, 500 kHz | 1 MHz |
| [`06_plate_in_halfspaces_immersed_water.yaml`](06_plate_in_halfspaces_immersed_water.yaml) | Aluminium plate between water regions; immersion loading | Ricker, 500 kHz | 1 MHz |
| [`07_fluid_acoustic.yaml`](07_fluid_acoustic.yaml) | Water channel; scalar pressure and outgoing radiation | Ricker, 250 kHz | 500 kHz |
| [`08_fluid_solid_acoustic.yaml`](08_fluid_solid_acoustic.yaml) | Aluminium on a sponge-terminated water half-space; interface traction and an 81-receiver fluid pressure array | Ricker, 200 kHz | 500 kHz |

The examples cover all supported structure modes and the isotropic, anisotropic,
viscoelastic, and fluid material models. The examples use explicit RK4 integration;
implicit Newmark integration is available through the solver setting or CLI.
Material properties are
representative application inputs; calibrate them to the specimen or site when
comparing with measurements. References in each file provide physical context.

## Quick start

Activate the FEM environment described in the repository README and run commands
from the repository root:

```sh
# Inspect the configuration and estimate resources without running the solver.
wavesim -c configs/template.yaml --dry-run

# Run with four CPU processes.
sh tools/run_cpu4.sh -c configs/template.yaml

# Create a separate configuration for a new study.
cp configs/template.yaml my_case.yaml
wavesim -c my_case.yaml --dry-run
```

Set a unique `output.output_dir` for each case and resolution. The output archive
contains receiver histories, resolved configuration, and runtime metadata.
Use the saved metadata to identify the actual mesh, integration step, and
recording times used for a result.

### Four-CPU execution

`tools/run_cpu4.sh` launches four MPI ranks, sets numerical-library thread pools
to one thread per rank, and selects CPU/double precision. It uses the active
environment's Python and MPI. Rank count is a launch parameter; it is not a YAML
input. A direct `wavesim -c ...` invocation uses a single process unless
started under MPI.

```sh
sh tools/run_cpu4.sh -c configs/03_halfspace_steel_rayleigh.yaml

# Run without figure generation.
sh tools/run_cpu4.sh -c configs/06_plate_in_halfspaces_immersed_water.yaml \
    --no-postprocess --output results_immersed_cpu4
```

On a cluster, request four CPU slots and use the scheduler's supported MPI
placement policy. The launcher sets process and thread counts; affinity is
managed by MPI or the scheduler. Measure scaling on the target hardware, since
MPI communication and per-process memory can outweigh parallel gains for small
models.

## File structure and conventions

| Section | Purpose |
| :-- | :-- |
| `solver` | Integration method, time-step controls, recording duration, device, and precision |
| `geometry` | Horizontal observation length and lateral sponge widths |
| `structure` | Physical regions, thicknesses, constitutive models, and material properties |
| `absorbing_boundaries` | Mass- and stiffness-proportional damping in sponge regions |
| `excitation` | Traction waveform, frequency, amplitude, footprint, position, and direction |
| `mesh` | Explicit element counts, polynomial degree, and sizing frequency band |
| `sensors` | Receiver array position, spacing, count, and storage rate |
| `postprocess` | Dispersion figures and display limits |
| `output` | Output directory, history storage, field snapshots, and figure formats |

- **Units:** numeric quantities use m, s, Hz, Pa, and kg/m³. Viscous stiffness
  `Cv` uses Pa·s. Counts, Poisson ratios, and control factors are dimensionless.
  Unit strings are accepted for `sensors.sample_rate` and postprocessing limits.
- **Automatic values:** `null` selects the parameter's documented default or
  automatic rule; it does not mean zero or disabled.
- **Precedence:** command-line overrides take precedence over file values.
  Unknown keys and constitutive fields incompatible with the selected material
  model are rejected.
- **Reference:** [`template.yaml`](template.yaml) documents the full schema,
  automatic rules, constraints, alternative structures, and compatibility aliases.

## Physical model

All examples use linear 2D plane-strain wave propagation. Layers are listed
bottom-to-top, with displacement continuity at interfaces. An excitation is a
distributed traction; transducer mechanics are not included.

### Geometry and absorbing regions

The total horizontal length is
`Lx = sensing_length + 2 * absorb_length`. Each lateral sponge lies outside the
observation region. For a halfspace, `thickness` is its finite truncation depth
or height and includes `absorb_depth`. The remaining portion is undamped.

Choose undamped depth to accommodate modal penetration, and sponge dimensions to
attenuate outgoing waves over the retained band. Compare larger domains at fixed
element spacing to distinguish boundary effects from spatial discretisation error.
Automatic damping coefficients scale with source frequency; absorption still
depends on wavelength, incidence angle, and sponge dimensions.

### Materials

- **Isotropic:** specify density, Young's modulus, and Poisson ratio.
- **Anisotropic:** specify the plane-strain stiffness tensor in global x1/x2 axes.
  Tensor entries follow `[C11, C12, C16, C22, C26, C66]`. Example 02 uses eight
  elastic CFRP plies with fibres along x1 at 0 degrees and along the invariant
  plate direction x3 at 90 degrees. Its tensors are reconstructed from the
  full three-dimensional compliance and restricted to plane strain after
  rotation; each layer supplies its global tensor directly.
- **Viscoelastic:** provide both `C` and `Cv` in the same order. Kelvin-Voigt loss
  grows with frequency. Example 05 uses anisotropic elastic and viscous tensors
  for its cortical-bone layer; example 02 is purely elastic.
- **Fluid:** scalar pressure satisfies the density-weighted acoustic wave equation
  with `K = rho*c^2`. Solid interfaces transfer normal traction and normal
  acceleration, with tangential slip. Specify one of `c`, `K`, or `E` plus `nu`
  (the latter pair determines bulk modulus only). No shear speed is assigned.

### Fluid formulation and migration

`material_model: fluid` now defaults to `fluid_formulation: acoustic`. This changes
the equations for existing fluid input files. Remove `fluid_shear_reg` to use
pressure acoustics. To reproduce a prior run or compare with the old approximation,
select the legacy formulation explicitly in **each** fluid region:

```yaml
material_model: fluid
rho: 1000.0
c: 1480.0
fluid_formulation: regularized_elastic
fluid_shear_reg: 1.0e-8       # Artificial shear modulus / bulk modulus
```

The legacy model retains displacement degrees of freedom, artificial shear and
tangential displacement continuity. Agreement is expected for converged, mainly
longitudinal waves such as normal-incidence transmission. Interface and oblique
waves also require independent inviscid reference solutions; matching the legacy
model alone cannot establish acoustic correctness.

Acoustic pressure sources need an explicit `source_kind` and `source_amplitude`.
The amplitude multiplies the configured Ricker pulse or tone burst:

| `excitation.source_kind` | Amplitude input | Meaning |
| :-- | :-- | :-- |
| `traction` | `traction_max` [Pa] | Applied solid traction, with direction `source_fx`, `source_fy` |
| `acoustic_pressure` | `source_amplitude` [Pa] | Prescribed compression-positive pressure on an exterior fluid face |
| `acoustic_normal_acceleration` | `source_amplitude` [m/s²] | Boundary acceleration, positive into the fluid |
| `acoustic_volume` | `source_amplitude` [s⁻²] | Right-hand side of `(1/K) p_tt - div((1/rho) grad(p)) = q` |

Acoustic Ricker sources use a smooth causal turn-on during the first `0.25/f_center`
seconds so that pressure and its first two time derivatives start at zero,
consistent with the default resting fields. The pulse beyond that interval and
Hann-windowed tone bursts retain their configured waveforms.

For a volume source, `source_region` must name a fluid region. Here `q` is the
time derivative of a volumetric injection rate per unit volume, in the absence
of fluid body forces. Boundary sources use `source_boundary: top`, `bottom`,
`left`, or `right`. A solid interface traction uses `source_boundary: interface`,
`source_y`, and a solid `source_region`. Lateral sources resolve `source_x` to
`0` or the total domain length; contradictory coordinates are rejected. The
automatic receiver array for a right source lies to its left in increasing x.
`source_y` applies only to `interface` loads; a value contradicting a `top` or
`bottom` face, or any value with a lateral face, is rejected.

Spatial distributions use the same Gaussian as the traction footprint,
`exp(-(x - source_x)² / (2σ²))` with `σ = excitation_width / 3`, on `top`,
`bottom` and `interface` loads; lateral faces are loaded uniformly over the
selected region's height. A volume source is this Gaussian in `x` and uniform
through the thickness of its `source_region`. An `acoustic_pressure` source
prescribes pressure on **every** node of the driven fluid face: away from the
footprint the prescribed value tends to zero, so the rest of that face behaves
as pressure release rather than rigid. Normal-acceleration faces stay rigid
outside the footprint.

`acoustic_boundaries` maps `top`, `bottom`, `left`, and `right` to `rigid`
(the default), `pressure_release`, or `radiation`. These apply only to exterior
fluid faces. Radiation is a local first-order outgoing condition, exact for
normal incidence in a homogeneous medium; vary domain size for oblique waves.
`geometry.absorb_length: 0` is supported for acoustic cavities or radiation tests.
Sponges add numerical damping and are separate from physical material losses.

Set `sensors.observable` to `pressure` [Pa] or `particle_velocity` [m/s] in fluids,
or `displacement` [m] or `velocity` [m/s] in solids. `auto` selects displacement
on a solid side and pressure in a fluid. At an interface, use `sensors.region`
to select the material side, particularly for particle velocity. Pressure
archives use `sensor_data_pressure`; vector velocity uses `sensor_data_vx` and
`sensor_data_vy`. Schema 2 stores the observable and units explicitly. Older
displacement archives remain readable and retain legacy-fluid interpretation;
pressure is never reinterpreted as displacement. Scalar pressure reports omit
vector polarization diagnostics.

Example 05 applies pressure at the skin surface and records fluid particle
velocity. Its bottom is pressure-release and its top has prescribed Gaussian
pressure, tending to zero away from the source. Pressure receivers on those
faces would simply measure the prescribed boundary value. Example 06 applies
traction to the plate and selects solid displacement
on the plate side of the upper interface.

See examples [07](07_fluid_acoustic.yaml) and [08](08_fluid_solid_acoustic.yaml)
for runnable pure-fluid and coupled inputs, and the
[implementation notes](../docs/acoustics_design.md) for equations and validation.

### Excitation

`ricker` provides a broadband pulse for dispersion studies; example 02 uses it
at 1 MHz. `tone_burst` provides a Hann-windowed sinusoid with duration
`n_cycles / f_center`. Example 03 uses five cycles at 1 MHz, giving a 5 µs pulse.

The Gaussian traction footprint has `sigma = excitation_width / 3`. The vector
`(source_fx, source_fy)` is normalised, and `traction_max` sets the resultant
traction scale. Normal loading favours flexural or surface motion; tangential
loading can improve coupling to extensional motion.

## Numerical resolution

### Fundamental Lamb-mode report

For a free, single isotropic plate, `postprocess.lamb_modes: true` generates
theory-guided A0/S0 displacement reconstructions. Example 01 enables this report
and disables generic mixed-packet fits. Its PNG and PDF figures are saved under
`lamb_modes/`: `modal_separation` overlays Rayleigh–Lamb branches and shows modal
propagation maps; `modal_waveforms` compares the tapered input, A0, S0, and residual
at three interior receivers. Time, distance, and displacement use scale-adaptive
units. Propagation maps share a colour limit at the input's 99th absolute
percentile, with extended colourbars indicating clipping.

`modal_signals.npz` contains both displacement-component reconstructions and
`diagnostics.json` records the archived model, frequency band, mask width, and
number of assigned frequency bins. The physical model and sizing band come from
the archive, not a potentially modified simulation configuration.

Separation applies a 10% Tukey taper in time and space, then compact cosine masks
with a two-bin wavenumber half-width around theoretical A0/S0 branches. Both
propagation directions are retained. Frequencies where these supports overlap
or reach spatial Nyquist remain unassigned. The sum of A0, S0, and residual equals
the **tapered** input to floating-point precision. The residual includes higher
modes, out-of-band content, mask rejection, and unresolved branches; it is not
necessarily noise. Other higher branches may contaminate a filter if they lie
within its support. This is approximate spectral separation, not an orthogonal
modal projection or an energy measurement.

Fourier filtering is nonlocal: finite-window ringing and periodic edge effects
can produce apparent pre-arrivals. Interpret interior packets, compare aperture
and window choices, and assess mask-width sensitivity before reporting modal
amplitudes. For the elastic aluminium plate, amplitude evolution is not intrinsic
material attenuation. No automated velocity or loss estimate is inferred from
these reconstructed signals.

```sh
wavesim -c configs/01_plate_aluminum_lamb.yaml \
    --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

### Spatial discretisation

All configurations use quadratic quadrilateral elements (Q2). `nx` is the number
of elements across the entire horizontal domain, including both lateral sponges.
`ny_per_region` lists element counts bottom-to-top: substrate, finite layers, and
superstrate where present. Sponge depths do not create additional regions.

| Case | `nx` | `ny_per_region` | Horizontal spacing |
| :-- | --: | :-- | :-- |
| Template | 640 | `[6]` | 0.219 mm |
| 01 | 1440 | `[12]` | 0.236 mm |
| 02 | 3300 | `[2, 2, 2, 2, 2, 2, 2, 2]` | 0.121 mm |
| 03 | 920 | `[60]` | 0.283 mm |
| 04 | 180 | `[30, 10]` | 1 m |
| 05 | 757 | `[6, 22, 6]` | 0.185 mm |
| 06 | 600 | `[25, 6, 25]` | 0.233 mm |

A useful initial Q2 resolution is 6–10 elements per shortest retained physical
wavelength, with sufficient rows to resolve through-thickness gradients. Guided
and interface modes can have shorter wavelengths than bulk-speed estimates.
Use an independent dispersion solution where available, and refine both directions
to establish accuracy over the propagation distance of interest.

**Element counts are explicit.** Changing frequency, polynomial degree, or
`mesh.min_wavelength` does not automatically refine the mesh. Compute counts from
the desired spacing, for example `nx = ceil(Lx / dx)`.

### Time integration

The explicit examples use lumped-mass RK4 with `CFL: 0.8`. CFL is a fraction of
the assembled stiffness/damping stability certificate, not simply `c * dt / dx`.
The final step also depends on time-grid and sampling alignment. Halve CFL or a
specified `dt` to assess temporal sensitivity.

Example 02 uses explicit RK4 for the elastic CFRP laminate. Refine the cells
through each ply and reduce the time step independently when comparing its
anisotropic dispersion with a layered reference. Implicit Newmark remains
available for separate studies, with average-acceleration defaults
`beta: 0.25` and `gamma: 0.5`. Compare 40 and 80 steps per period to assess
temporal sensitivity; unconditional stability does not establish accuracy.

### Frequency band and duration

`mesh.max_frequency` sets the sizing upper frequency, implicit time-step target,
and automatic sampling rate. `mesh.min_frequency` participates in arrival sizing.
Neither value filters the source or recorded histories. Postprocessing limits
also control display only. Inspect source energy outside the nominal band when
choosing numerical resolution and stored sampling rate.

Automatic duration combines source-pulse duration, estimated travel time, and
`solver.time_factor`. Free isotropic plates use an A0-aware estimate; higher modes
near cutoffs can arrive later. Layered and fluid-loaded configurations use a
reference-speed estimate that does not bound all guided-mode arrivals.

For a mode-specific window, provide `solver.min_group_velocity` from an independent
dispersion calculation or set `solver.duration` explicitly. Compare longer windows
to verify arrival coverage. Frequency resolution is approximately `1/T`; longer
windows also increase computation and may include boundary reflections.

## Receiver arrays and output

Array aperture is `(n_sensors - 1) * sensor_pitch`. Approximate wavenumber
resolution is `2*pi/aperture`; zero padding does not improve physical resolution.
Spatial Nyquist requires `sensor_pitch < lambda_min/2`; a quarter-wavelength pitch
provides additional margin.

Examples 01/02 use 200 mm apertures; 03/05/06 use 60 mm; example 04 uses 60 m.
Example 02 uses 801 receivers at 0.25 mm pitch, with spatial Nyquist
12.57 rad/mm, while the displayed range ends at 6 rad/mm. Check the independently
calculated CFRP wavelengths over the excited band and reduce pitch if needed,
increasing receiver count to preserve aperture. Unresolved modes can alias into the displayed range;
plot limits do not prevent aliasing.

The requested storage rates provide at least ten samples per upper-band period. Actual
recording intervals are quantised to solver steps. Stored sampling controls history
size rather than integration accuracy, and recording does not apply an anti-alias
filter.

Full-field VTX output is disabled in the examples. PNG figures and uncompressed
receiver archives provide a lightweight output path. Select `history_storage: disk`
when receiver buffers exceed available RAM; allow scratch and archive space.
Postprocessing can still require substantial memory.

All figures use 190 mm width and 10-point text, including tick numbers,
legends and colourbars. `output.usetex: false` is the default and uses Matplotlib's
bundled serif fonts and mathtext, so no LaTeX installation is needed. The template
and examples 02–06 select this setting. Example 01 selects `output.usetex: true`
for external LaTeX rendering; this requires `latex`,
`dvipng` and Ghostscript. The legacy `auto` setting uses LaTeX when its toolchain
is available, otherwise built-in rendering. Panel sizing and styling stay the
same in either mode.
Figure explanations and provenance are written to accompanying README files.

Replot an existing archive without rerunning the simulation:

```sh
wavesim --postprocess results_03_halfspace_steel_rayleigh/sensor_data.npz \
    --f-min "0.5 MHz" --f-max "1.5 MHz"
```

## Resource planning

### Wave-characterisation reports

All six examples also enable `postprocess.structure_analysis: true`. This adds
two four-panel reports in `wave_physics/`: dispersion/mode interpretation and
propagation/component balance. Analysis follows the archived structure:
free-plate Lamb references, halfspace Rayleigh/polarization comparisons,
layered-ground wavelength/thickness, viscoelastic spatial-decay fits, or
solid–fluid sound-line comparisons. Qualified measurements, exclusion reasons
and physical references accompany the figures in CSV, JSON and NPZ files.
The [main post-processing guide](../README.md#outputs-and-post-processing)
describes the sampling checks and limits of each measurement.

Examples 03–06 enable the receiver report in `postprocess`. Example 01 uses the
dedicated Lamb-mode report, while example 02 uses dispersion reports; both
disable packet measurements.
Set `wave_characterisation.enabled: false` to omit packet measurements; set
`structure_analysis: false` separately to omit the structure-specific reports.
The report settings follow this structure:

```yaml
postprocess:
  wave_characterisation:
    enabled: true
    component: y
    frequency_band: ["0.7 MHz", "1.3 MHz"]
    time_window: null
    min_relative_amplitude: 0.01
```

The supplied starting bands are:

| Configuration | Report frequency band |
| :-- | :-- |
| Template | 0.3–0.8 MHz |
| 01 aluminium plate (disabled) | 0.5–1.5 MHz |
| 02 CFRP plate (disabled) | 0–2 MHz |
| 03 steel halfspace | 0.7–1.3 MHz |
| 04 soil/bedrock | 8–25 Hz |
| 05 bone | 0.3–0.8 MHz |
| 06 immersed plate | 0.3–0.8 MHz |

These bands support initial inspection; they do not guarantee isolation of a
particular mode. All examples select the vertical component and initially use
the full record. Inspect arrival picks before interpreting velocity or attenuation,
especially for overlapping modes. Report filtering does not change the simulation
or the standard dispersion plot. `null` leaves the report signal unfiltered.
A zero lower edge, such as `["0 MHz", "2 MHz"]`, selects a low-pass filter.
A positive lower edge selects a band-pass filter. The upper edge must remain
strictly below the stored sampling Nyquist frequency. The complete-record mean
is removed before filtering. Low-pass filtering still retains slowly varying
trends, which can affect envelope-based packet measurements.
A time window is a pair of seconds
within the archive, used to isolate a packet; filtering precedes windowing.
An omitted or disabled report leaves standard dispersion processing unchanged.
Use the same settings when analysing an existing archive:

```sh
wavesim -c my_case.yaml --postprocess results_case/sensor_data.npz
```

The `characterisation/` subdirectory contains four figure families:
`arrival_velocity`, `pulse_evolution`, `particle_motion`, and `time_frequency`,
using the configured figure formats. `receivers.csv` records per-receiver SI
measurements and fit inclusion; `metrics.json` records settings, fit results,
method, source archive path, and interpretation notes.

Arrival detection interpolates the Hilbert-envelope peak within the selected gate.
Receivers are excluded for weak signals, truncated packets, or multiple resolved
packets. Velocity is the reciprocal of the signed ordinary-least-squares
travel-time slope along x; the fit requires at least three accepted receivers,
R-squared of at least 0.9, and a fitted arrival span of at least two stored samples.
It is an apparent packet velocity, not automatic modal identification. Apparent
attenuation fits log peak amplitude along the inferred propagation direction;
unqualified spatial trends are withheld and growing signals can give negative
values. It includes interference and radiation, not only material loss. Pulse
width is the envelope-energy RMS width. Inspect the plotted picks and fit
diagnostics before interpreting a measurement.

Particle-motion plots use both displacement components at the first, middle, and
last receivers. Spectrogram power is normalised separately at each receiver.
Reports require both components, at least 32 uniform time samples, and at least
three increasing receiver coordinates. Gates require at least 16 samples and
frequency bands must lie below stored-data Nyquist. Unavailable fits are JSON
`null`; these reports do not perform material inversion or provide confidence
intervals. Multi-archive comparison and field-derived stress/energy quantities
are separate analyses.

Run `--dry-run` after changing geometry, frequency, mesh, or receivers. Preflight
reports element spacing, estimated degrees of freedom, duration, step count,
history size, and memory. Explicit estimates use a conservative element-level
certificate; the assembled operator determines the final integration limit.

Treat preflight memory as an estimate rather than a reservation limit. Sparse
factorization fill, MPI imbalance, per-process runtime overhead, operating-system
page cache, and FFT allocations affect actual usage. Measure wall time and peak
memory on the intended hardware with the benchmark utility:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    mpiexec -n 4 python tools/benchmark.py \
    -c configs/template.yaml --output results_template_benchmark
```

Use a fresh output directory. Benchmark measurements include setup, integration,
collection, and archive writing, excluding plots and VTX snapshots. Repeat runs
when comparing performance and distinguish cold compilation from warm-cache runs.

## Verification workflow

1. **Define acceptance criteria.** Select the mode, frequency band, arrival-time
   interval, and tolerances for phase velocity, waveforms, or attenuation.
2. **Check temporal convergence.** Halve the explicit step control or double
   implicit steps per period, keeping physical inputs fixed.
3. **Check spatial convergence.** Increase `nx` and each relevant region count by
   1.5–2 with the tighter time control. Hold source and receiver coordinates fixed.
4. **Check the complete measurement.** Include the farthest receiver and all
   relevant arrivals, where accumulated phase error may be largest.
5. **Check domain truncation.** Extend the recording window and vary undamped
   depth and sponge dimensions at fixed spatial resolution.
6. **Compare with a reference.** Use Rayleigh-Lamb theory, homogeneous Rayleigh
   speed, or a suitable layered/fluid-loaded dispersion solution.
7. **Select the least expensive passing configuration.** Retain inputs, metadata,
   comparison results, and hardware details with the study.

The resolution comparison utility provides an initial early-arrival check:

```sh
python tools/compare_resolution.py \
    --config configs/template.yaml \
    --output results_template_resolution_check \
    --duration 30e-6 --refinement-factor 1.5 --step-factor 0.5
```

It compares a refined mesh with tighter integration, the original mesh with
tighter integration, and the original settings. Waveform errors use the first
nine receivers and the specified window, without amplitude normalisation or
time alignment. Follow this check with full-aperture and full-window verification
for the intended application.

## Compatibility

Use explicit `mesh.nx` and `mesh.ny_per_region`; removed automatic-count keys
`elems_per_wavelength`, `min_elems_x`, and `min_elems_y` are rejected. Explicit
runs require `solver.CFL`, including when `solver.dt` is supplied. Prefer canonical
postprocessing keys with unit strings over legacy MHz/rad-per-mm aliases. The
template lists supported aliases and alternative layouts.
