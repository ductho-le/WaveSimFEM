# WaveSimFEM

**A GPU-accelerated multi-scale finite element framework for acoustic and elastic wave simulation in heterogeneous layered media**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![DOLFINx 0.10.0](https://img.shields.io/badge/DOLFINx-0.10.0-orange.svg)](https://fenicsproject.org/)
[![MPI / PETSc](https://img.shields.io/badge/MPI-PETSc-green.svg)](https://petsc.org/)
[![CI](https://github.com/ductho-le/WaveSimFEM/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ductho-le/WaveSimFEM/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**WaveSimFEM** is a GPU-accelerated multi-scale finite element framework for acoustic and elastic wave simulation in heterogeneous layered media. It combines DOLFINx finite element assembly with distributed PETSc solvers and a single-device CuPy CUDA engine, carrying physical parameters and acquisition settings through to archived receiver measurements and frequency–wavenumber representations.

The framework supports guided- and surface-wave studies in metallic and composite plates, layered ground, and biomedical and immersed structures across ultrasonic and seismic scales. Its common simulation-to-data workflow provides reusable infrastructure for parameter studies, acquisition design, inverse problems, and labeled training collections for machine-learning and deep-learning applications.

**Configure the structure and acquisition → assemble → integrate → record receivers → archive → analyze and reuse**

## Contents

- [Framework capabilities](#framework-capabilities)
- [Physical model](#physical-model)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CPU and GPU execution](#cpu-and-gpu-execution)
- [Example models](#example-models)
- [Configuration](#configuration)
- [Outputs and post-processing](#outputs-and-post-processing)
- [Synthetic-data campaigns](#synthetic-data-campaigns)
- [Numerical accuracy and verification](#numerical-accuracy-and-verification)
- [Development](#development)
- [Citation and license](#citation-and-license)

## Framework capabilities

| Stage | Implementation |
| :--- | :--- |
| Model configuration | Validated YAML, TOML, or JSON inputs for geometry, materials, excitation, receivers, numerical controls, and output |
| Finite element model | Interface-conforming quadrilateral meshes and shared DOLFINx/UFL assembly of mass, stiffness, and damping operators |
| Transient integration | Lumped-mass explicit RK4 on SciPy, CuPy, or distributed PETSc; consistent-mass implicit Newmark on PETSc |
| Receiver acquisition | Solid displacement or velocity, fluid pressure or particle velocity, evaluated at specified coordinates with explicit units and material-side selection |
| Analysis | Waveforms, B-scans, frequency–wavenumber spectra, structure-specific diagnostics, and optional mode or packet reports |
| Reproducible data | Receiver archives with resolved inputs, acquisition geometry, realized time grids, backend and precision metadata, and source hashes |

The elastic CUDA engine retains sparse operators and evolving state on the device. It evaluates stiffness and damping through a combined sparse block, using one internal-force product per RK4 stage, and buffers receiver samples before transferring them to host storage. The coupled acoustic engine also advances its fields on the selected device, but uploads forcing each stage and updates host field mirrors each step for acquisition. Mesh generation, finite element assembly, and spectral post-processing run on the CPU.

## Physical model

WaveSimFEM solves **linear, two-dimensional plane-strain elastodynamics coupled to pressure acoustics** in flat, layered structures. The propagation and depth/thickness coordinates are `x1` and `x2`; solid displacement components `u1` and `u2` are named `x` and `y` in the API and archives. Pressure and velocity observables have their own field names and physical units. The model is invariant in the third direction and assumes perfectly bonded solid interfaces.

Supported structures are free single or multilayer plates, truncated halfspaces, plates on substrates, and plates between lower and upper halfspaces. Solid materials may be isotropic elastic, anisotropic elastic, or Kelvin–Voigt viscoelastic. Anisotropic stiffness and any Kelvin–Voigt viscosity are supplied in the simulation axes; three-dimensional material orientations must permit decoupling of in-plane and out-of-plane motion. Solid excitation uses distributed traction; acoustic excitation supports prescribed pressure, normal acceleration, and volume forcing, with Ricker or Hann-windowed tone-burst waveforms. Graded mass- and stiffness-proportional sponge damping truncates the domain.

Fluid regions solve **linear inviscid pressure acoustics** by default:

\[
\frac{1}{K}\frac{\partial^2 p}{\partial t^2}
-\nabla\cdot\left(\frac{1}{\rho}\nabla p\right)=q,
\qquad K=\rho c^2.
\]

Pressure is compression-positive. At solid–fluid interfaces, pressure applies normal traction and solid normal acceleration drives the fluid; tangential slip is allowed. Separate pressure and displacement fields occupy only their material regions. Acoustic materials have no artificial shear modulus or shear-wave speed. Rigid, pressure-release, and first-order radiation conditions are available on exterior fluid boundaries, alongside numerical sponges. The finite element and time discretizations still require convergence checks.

For reproducibility and comparisons, the previous displacement approximation remains available through an explicit `fluid_formulation: regularized_elastic` on each fluid region. It retains artificial shear and tangential displacement continuity. See the [fluid migration and source guide](configs/README.md#fluid-formulation-and-migration) for inputs, source units, receiver fields, and archive compatibility. Curved or three-dimensional geometries, out-of-plane polarization, mean flow, nonlinear response, and detailed transducer coupling are outside the current model.

## Installation

### Create the environment

The supplied Conda environment supports Linux and macOS and specifies Python 3.11/3.12, DOLFINx 0.10.0, PETSc/petsc4py 3.24, MPI, and the scientific Python dependencies.

```bash
git clone https://github.com/ductho-le/WaveSimFEM.git
cd WaveSimFEM
conda env create -f environment.yml
conda activate wavesim
python -m pip install --no-deps -e .
```

Check the installed command:

```bash
wavesim --version
wavesim --help
```

> [!IMPORTANT]
> Install the DOLFINx/PETSc/MPI stack through `environment.yml`. Installing the Python package with pip alone does not supply these runtime dependencies.

### Enable CUDA execution

On a system with a supported NVIDIA GPU and a compatible CUDA driver, install the optional CuPy dependency:

```bash
python -m pip install -e ".[cuda]"
```

The extra installs `cupy-cuda12x`. Each CUDA simulation uses one process and one GPU. CPU execution is available on systems without a supported NVIDIA device, including Apple Silicon.

## Quick start

Run these commands from the repository root with the `wavesim` environment active.

### Inspect the model

```bash
wavesim -c configs/01_plate_aluminum_lamb.yaml --dry-run
```

Preflight reports element spacing, estimated degrees of freedom, duration, step count, receiver-history size, and memory without constructing the finite element model. The supplied examples are research-scale configurations; use these estimates to plan a run.

### Simulate and archive receiver data

```bash
wavesim -c configs/01_plate_aluminum_lamb.yaml --device cpu --no-postprocess
```

This writes the aluminum-plate measurements and resolved run configuration to `results_01_plate_aluminum_lamb/`.

### Analyze the saved record

```bash
wavesim --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

Standalone post-processing uses the archived physical model and generates waveforms, B-scans, frequency–wavenumber spectra, and the default structure report. A plotting configuration selects optional reports such as Lamb reconstruction. Without a plotting configuration, it uses built-in fonts and requires no LaTeX installation.

Omit `--no-postprocess` to simulate and plot in one invocation. Example 01 sets `output.usetex: true`; change that setting to `false` for built-in rendering, or provide `latex`, `dvipng`, and Ghostscript. The reference template and Examples 02–06 already use `usetex: false`.

`python -m wavesimfem` can replace `wavesim` in these commands.

## CPU and GPU execution

Select an execution path while retaining the same physical and acquisition configuration:

```bash
# Single NVIDIA GPU, explicit RK4
wavesim -c configs/01_plate_aluminum_lamb.yaml --device cuda --no-postprocess

# Four CPU processes, distributed PETSc RK4
mpiexec -n 4 wavesim -c configs/01_plate_aluminum_lamb.yaml --device cpu --no-postprocess

# Four CPU processes, consistent-mass Newmark
mpiexec -n 4 wavesim -c configs/01_plate_aluminum_lamb.yaml --solver implicit --device cpu --no-postprocess
```

| Device request | Single-process explicit | Distributed explicit | Implicit |
| :--- | :--- | :--- | :--- |
| `auto` | CuPy CUDA when available; otherwise SciPy CPU | PETSc CPU | PETSc CPU |
| `cpu` | SciPy CPU | PETSc CPU | PETSc CPU |
| `cuda` or `cuda:N` | Requested GPU; error if unavailable | Unsupported; raises an error | Acoustic: error; elastic: warning and PETSc CPU |

The examples explicitly select `cpu`. Use `--device auto` for automatic selection or `--device cuda:1` to choose a specific GPU. Distributed multi-GPU integration is outside the current execution model; independent cases can run on separate devices through external campaign scripts.

Coupled acoustic runs support single-process SciPy/CuPy RK4 in single or double
precision and PETSc RK4/Newmark in double precision. The acoustic CUDA path has
parity tests, but has not been exercised on CUDA hardware in the current local
verification environment. CPU and MPI results establish the available backend
checks; GPU-specific tests skip when CUDA is unavailable.
Implicit and distributed acoustic runs reject single precision and explicit CUDA
requests before assembly; they do not silently change the requested backend.

Double precision is the default. `--precision single` enables FP32 in the single-process SciPy/CuPy engine; PETSc paths use FP64. Assess precision over the complete observation window because phase differences can accumulate with propagation distance and step count.

The CLI limits numerical-library threads per MPI rank and respects explicit thread settings. The supplied launcher selects four ranks, one library thread per rank, CPU execution, and double precision:

```bash
sh tools/run_cpu4.sh -c configs/template.yaml --no-postprocess
```

## Example models

The application configurations use Q2 quadrilateral elements and explicit RK4. They include material references, source and receiver settings, exact element counts, and output options.

| Configuration | Structure | Application | Excitation |
| :--- | :--- | :--- | :--- |
| [01 · Aluminum plate](configs/01_plate_aluminum_lamb.yaml) | 3 mm free isotropic plate | Lamb-wave inspection | 1 MHz Ricker |
| [02 · CFRP plate](configs/02_plate_cfrp.yaml) | 1 mm, eight-ply T300/914 `[0/90/0/90]_s` laminate | Anisotropic guided-wave dispersion | 1 MHz Ricker |
| [03 · Steel halfspace](configs/03_halfspace_steel_rayleigh.yaml) | Truncated isotropic halfspace | Rayleigh surface waves | 1 MHz, five-cycle tone burst |
| [04 · Soil over bedrock](configs/04_plate_on_halfspace_soil_bedrock.yaml) | 10 m soil layer on a truncated substrate | Seismic surface-wave analysis | 15 Hz Ricker |
| [05 · Cortical bone](configs/05_layers_bone_qus.yaml) | Marrow / viscoelastic bone / tissue stack | Axial-transmission ultrasound | 500 kHz Ricker |
| [06 · Immersed plate](configs/06_plate_in_halfspaces_immersed_water.yaml) | 1.5 mm aluminum plate between water regions | Fluid-loaded guided waves | 500 kHz Ricker |
| [07 · Acoustic water channel](configs/07_fluid_acoustic.yaml) | Pure fluid with rigid and radiation boundaries | Pressure propagation | 250 kHz Ricker |
| [08 · Coupled plate and water](configs/08_fluid_solid_acoustic.yaml) | Aluminum plate on a water half-space | Fluid pressure array (40 mm) from interface traction | 200 kHz Ricker |

These inputs are starting points for application studies. Establish resolution, recording-window, and boundary requirements for the observables of interest before using them in a data-generation campaign. See the [example guide](configs/README.md) for material conventions and acquisition details, and the [annotated reference configuration](configs/template.yaml) for the complete parameter guide.

## Configuration

Copy the closest example or the annotated reference configuration. The CLI can also generate a starter file:

```bash
wavesim --generate-config my_waveguide.yaml
wavesim -c my_waveguide.yaml --dry-run
```

YAML, TOML, and JSON share the same validated schema. Settings resolve in the order **built-in defaults → configuration file → CLI overrides**. Unknown, duplicated, or incompatible fields produce an input error.

| Section | Controls |
| :--- | :--- |
| `structure` | Structure mode, region thicknesses, densities, and constitutive properties |
| `geometry` | Observation-region length and lateral sponge widths |
| `mesh` | Exact element counts, polynomial degree, and frequency/wavelength sizing inputs |
| `solver` | Integration scheme, device, precision, step controls, and duration |
| `excitation` | Pulse type, field-specific source kind and amplitude, Gaussian width, region, boundary, and direction |
| `sensors` | Observable, material side, receiver count, pitch, source offset, height, and storage rate |
| `absorbing_boundaries` | Peak mass- and stiffness-proportional sponge damping |
| `acoustic_boundaries` | Rigid, pressure-release, or first-order radiation on exterior fluid faces |
| `postprocess` | Spectral display limits and structure, mode, or packet reports |
| `output` | Destination, figure formats, history storage, compression, and field snapshots |

Numeric physical inputs use **SI units**. Receiver sampling rates and spectral display limits also accept unit strings such as `"20 MHz"`, `"2 MHz"`, and `"3 rad/mm"`. Logs and figures choose displayed units appropriate to the problem scale.

Choose the following controls for each study:

- **Mesh counts:** supply positive integers for `mesh.nx` and each entry of `mesh.ny_per_region`. Horizontal counts cover the observation region and both lateral sponges. Vertical counts follow physical regions from bottom to top, including any halfspaces; sponge depths remain part of those regions. Frequency, wavelength, and polynomial-degree changes do not refine the counts automatically.
- **Explicit step:** supply `solver.CFL` in `(0, 1]`, including when specifying `solver.dt`. CFL scales the assembled stiffness–damping stability limit. A requested step is checked against that limit and may shrink to align the time and recording grids.
- **Implicit step:** `solver.steps_per_period` controls temporal resolution at `mesh.max_frequency` when `dt` is automatic. Average-acceleration Newmark defaults to `beta: 0.25` and `gamma: 0.5`.
- **Duration:** set `solver.duration` for the required arrivals, or supply `solver.min_group_velocity` from an independent dispersion calculation. Automatic duration is an estimate, with A0-aware sizing for free isotropic plates; slow layered modes and cutoff tails may need longer records.
- **Acquisition:** choose receiver pitch, aperture, and storage rate for the retained signal band. Frequency sizing and display limits do not filter the source or recorded histories, and stored-data decimation applies no anti-alias filter.

Common overrides include:

```bash
wavesim -c my_waveguide.yaml --output results_my_waveguide
wavesim -c my_waveguide.yaml --duration 2e-4 --cfl 0.4
wavesim -c my_waveguide.yaml --f-max "2 MHz" --k-max "3 rad/mm"
wavesim -c my_waveguide.yaml --no-postprocess
```

Use `-q` for warnings only, `-v` for diagnostic logging, and `wavesim --help` for all options. Migration notes for older mesh and parameter conventions are in [CHANGELOG.md](CHANGELOG.md).

## Outputs and post-processing

### Run archives and figures

| Output | Contents |
| :--- | :--- |
| `sensor_data.npz` | Stored times, receiver coordinates, selected field (displacement, velocity, pressure, or particle velocity), units, model metadata, run identifier, and resolved configuration |
| `simulation_config.yaml` | Resolved inputs and runtime provenance: mesh, realized step and sampling, backend, precision, package/runtime versions, and source hashes |
| `quick_results.png` | Optional waveform and B-scan overview |
| `full_analysis.*` | Receiver waveforms, B-scan, and component-resolved frequency–wavenumber spectra |
| `wave_physics/` | Structure-specific figures, qualified measurements, spectral arrays, and interpretation notes |
| `lamb_modes/` | Optional A0/S0 reconstructions, residual signals, figures, and separation diagnostics |
| `characterisation/` | Optional packet reports, per-receiver measurements, fit diagnostics, and interpretation notes |
| `guided_wave.bp` | Optional elastic displacement snapshots for ParaView when `output.write_vtx: true`; acoustic runs write separate `solid_displacement.bp` and `acoustic_pressure.bp` fields when present |

Receiver histories have shape **`(number of stored times, number of receivers)`**. Load the measured components with NumPy:

```python
import numpy as np

with np.load("results_01_plate_aluminum_lamb/sensor_data.npz", allow_pickle=False) as data:
    times = data["times"]
    receiver_x = data["sensor_x"]
    u1 = data["sensor_data_x"]
    u2 = data["sensor_data_y"]
    run_id = str(data["run_id"].item())

print(run_id, times.shape, receiver_x.shape, u1.shape, u2.shape)
```

For schema-2 acoustic records, inspect `sensor_observable` and `sensor_units`.
Pressure is stored in `sensor_data_pressure` [Pa]; solid or fluid velocity
components use `sensor_data_vx` and `sensor_data_vy` [m/s]. Only displacement
records use `sensor_data_x` and `sensor_data_y` [m]. Older archives remain readable.

Use the saved time array: storage intervals are integer multiples of the integration step, so the realized sampling rate can differ from the requested rate. The configuration embedded in the archive is authoritative if an interrupted output update leaves the YAML sidecar out of sync.

### Reprocess an existing record

Change plotting settings without repeating the finite element solve:

```bash
wavesim --postprocess results_01_plate_aluminum_lamb/sensor_data.npz \
    --output reprocessed_figures --f-max "2 MHz"
```

A plotting-only configuration can select reports and formats:

```yaml
postprocess:
  structure_analysis: true
  lamb_modes: true
output:
  figure_formats: [png, pdf]
  usetex: false
```

Save it as `plot_settings.yaml`, then run:

```bash
wavesim -c plot_settings.yaml \
    --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

| Report | Interpretation |
| :--- | :--- |
| Structure analysis | Uses archived geometry and materials to select applicable dispersion references, phase-velocity diagnostics, component and polarization comparisons, directionality, or spatial-decay measurements; enabled by default |
| Lamb modes | Approximate theory-guided A0/S0 spectral reconstruction for a free, single isotropic elastic plate with archived model provenance; enabled in Example 01 |
| Wave characterisation | Optional packet arrival, apparent velocity and decay, pulse evolution, particle motion, and time–frequency reports; enabled in Examples 03–06 |

Lamb reconstruction is a spectral filter of the tapered input. Its residual includes higher modes, out-of-band signals, and unresolved or rejected content. Interpret it as an approximate displacement separation; it does not measure modal energy.

Enable packet reports through `postprocess.wave_characterisation.enabled` and select a band and time window containing the packet of interest. The analysis screens weak, incomplete, ambiguous, or insufficiently resolved signals and records exclusion reasons. Fitted velocity is an apparent packet velocity; fitted decay can include interference, radiation, geometric effects, and numerical error as well as material loss. Report directories contain README files describing methods, qualifications, and provenance.

Frequency–wavenumber maps show the selected observable's spectral power. Finite duration and receiver aperture set physical resolution; zero padding only provides denser spectral bins. Figures use a common 190 mm width and 10-point text, with PNG, PDF, SVG, and EPS export options. Built-in serif fonts and mathtext require no external LaTeX installation.

## Synthetic-data campaigns

A synthetic example consists of a complete physical and acquisition configuration and its recorded measurement. Retaining geometry, constitutive properties, source settings, and receiver sampling with each archive preserves their association with candidate learning targets. Waveforms can be reused to produce different spectral or preprocessing representations without another forward solve.

Build campaigns by scripting repeated CLI invocations or calling `wavesimfem.run_simulation`. The framework supplies forward simulation, receiver extraction, archiving, and analysis; campaign code supplies parameter sampling, scheduling, label definitions, dataset assembly, and ML/DL training. Assign a unique output directory to every case. Independent CUDA cases can run on separate devices, each with one process and one GPU.

For reproducible collections:

- Keep a manifest containing parameter distributions, units, sampling seeds, target definitions, run identifiers, output paths, and accepted, failed, or rejected status with reasons.
- Establish spatial, temporal, boundary, and acquisition requirements on representative cases and parameter extremes, then check complete histories and adequate sampling for each retained case.
- Define common physical coordinates, sample times, and preprocessing. Equal array shapes alone do not establish comparable acquisition when automatically resolved time grids differ.
- Keep receivers, windows, and augmented representations from the same physical realization in the same training or evaluation partition.
- Use `--no-postprocess` during generation and analyze selected archives afterward. Choose `output.history_storage: disk` for large host histories and `output.compress_output: true` when archive savings justify compression cost. Solver workspaces and FFT analysis still require memory.

Assess campaign performance as **accepted independent configurations per elapsed campaign time**, including setup, acquisition, archive writing, retries, and shared-resource contention. Record numerical acceptance criteria and hardware alongside throughput; a count of receiver traces represents a different quantity from a count of independent model realizations.

## Numerical accuracy and verification

The assembled RK4 step restriction accounts for stiffness and damping and supplies a stability bound. Accuracy depends separately on spatial resolution, integration step, receiver sampling, observation window, and domain truncation. Implicit stability likewise leaves phase and amplitude accuracy dependent on time-step choice and linear-solver tolerances.

Refine horizontal and through-thickness element counts while keeping the physical source and receiver coordinates fixed. Reduce the explicit step or increase implicit steps per period independently. Compare the complete measurement window, including distant receivers where phase error accumulates, and vary sponge dimensions and undamped depth at fixed spacing. Use Rayleigh–Lamb theory, homogeneous Rayleigh speed, or an independent anisotropic, layered, or acoustic reference appropriate to the model and observable.

The resolution utility provides an initial early-arrival comparison:

```bash
python tools/compare_resolution.py \
    --config configs/01_plate_aluminum_lamb.yaml \
    --output resolution_check --duration 40e-6 \
    --refinement-factor 1.5 --step-factor 0.5
```

It compares refined spatial and temporal settings over a shortened record and a small receiver subset. Follow it with full-aperture and full-window checks for the intended study.

Measure complete-run cost on the intended hardware with the benchmark utility:

```bash
python tools/benchmark.py \
    -c configs/template.yaml --output results_template_benchmark
```

The utility records simulation wall time and peak host memory, including setup, integration, receiver collection, and archive writing while disabling plots and VTX snapshots. Use fresh output directories, repeat matched runs, and distinguish cold compilation from warm-cache execution. CPU/GPU comparisons should retain the same model, acquisition, time grid, and precision; GPU memory and campaign throughput require additional measurements.

The automated tests provide implementation and selected physics checks. Application-wide accuracy, speedup, dataset yield, and fluid-model validity require the corresponding reference and performance studies; the supplied configurations alone do not establish those results.

The [acoustic validation report](docs/acoustic_validation.md) records analytical
cavity, transmission, Scholte-wave, and loaded-plate comparisons, including a
matched comparison with the legacy approximation. Reproduce its measured JSON:

```bash
python tools/verify_acoustics.py --case all --output docs/acoustic_validation_results.json
```

## Development

```text
WaveSimFEM/
├── src/wavesimfem/   # Configuration, FEM assembly, integration, acquisition, and analysis
├── configs/            # Annotated reference configuration and six application examples
├── tests/              # Unit, numerical, physics, backend, MPI, and CLI checks
├── tools/              # CPU launcher, benchmarking, refinement, and report previews
├── environment.yml     # Scientific-computing environment
├── pyproject.toml      # Package metadata and optional dependencies
├── CITATION.cff        # Software citation metadata
└── CHANGELOG.md        # Version history and migration notes
```

[`solvers.py`](src/wavesimfem/solvers.py) handles finite element construction and PETSc integration. [`solvers_gpu.py`](src/wavesimfem/solvers_gpu.py) provides the shared SciPy/CuPy RK4 engine. Configuration, material tensors, numerical bounds, storage, spectra, and report generation have separate modules under [`src/wavesimfem/`](src/wavesimfem/).

With the runtime environment active:

```bash
python -m pip install -e ".[dev]"
pre-commit install

python -m pytest
# Skip longer end-to-end physics runs
python -m pytest -m "not slow"

ruff check src tests tools
ruff format --check src tests tools
```

Tests cover configuration validation, analytical wave references, integration accuracy, receiver interpolation, backend agreement, MPI collection, archive integrity, and post-processing. CUDA checks require suitable hardware. To require the CuPy backend instead of skipping unavailable-device checks:

```bash
WAVESIM_REQUIRE_CUDA_BACKENDS=cupy python -m pytest tests/test_backend_validation.py
```

Report issues through [GitHub Issues](https://github.com/ductho-le/WaveSimFEM/issues), including the configuration, command, package/environment versions, and error output.

## Citation and license

WaveSimFEM is distributed under the [MIT License](LICENSE), which permits reuse and modification. If you use the framework in research, cite the software using [`CITATION.cff`](CITATION.cff) and identify the version or commit associated with the reported simulations or dataset. Retain the study configurations, campaign scripts, and processing settings with your results.
