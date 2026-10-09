# GuidedWaveFEM

### GPU-accelerated wave simulation and receiver-data analysis

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.10.0-orange.svg?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8%2F9hAAABN0lEQVR42oWSIUuDURSGH7YpuA0sBhcMWqbNJYNlLLggDIvNYJtZxGQR%2FAM2sTosJotFESxi2k8QwSboGE5BFB%2BDd%2Fjtc9%2B3Fw6Xe%2B5733Pvew4qI6Kkrqq5YecZ0lECDoBx4GsoI6XytHquttWpJF7SCwqhcgN4BT4S35igXFN7%2FuJd3Vazw7jxRE4tqhX1yT901S01M0pgWT1VZ9WWg3hRN9IE8upZIB%2BqK%2BpzTORRrSYJ1CP%2FflOb6pH%2FcRtm418XasF9gDywC9wB9zHfl4C1%2FiYTWWdixDmgDrQAI%2FkMUAWyUYFv4Ab4jImUgQvgIZbvhjsDHhTU%2FdAygx%2Bb6kIwr4%2B2upjUxjG1oV6pe%2BqkehwZqBO1nDYH%2FSiqE%2BqO2lGv1fWQG%2BDmEia8B1SAeaAJXAKdYcQfQWUiB%2F9coZIAAAAASUVORK5CYII%3D)](https://fenicsproject.org/)
[![MPI Parallel](https://img.shields.io/badge/MPI-MPI4Py%20%7C%20PETSc-green.svg?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZD0iTTEyIDVWNTRNNSAxOVYxNEgxOVYxOSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8%2BPGcgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iNSIgcj0iNCIvPjxjaXJjbGUgY3g9IjUiIGN5PSIxOSIgcj0iNCIvPjxjaXJjbGUgY3g9IjE5IiBjeT0iMTkiIHI9IjQiLz48L2c%2BPC9zdmc%2B)](https://petsc.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/ductho-le/GuidedWaveFEM/tests.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/ductho-le/GuidedWaveFEM/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-D7FF64?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

**GuidedWaveFEM** is an open-source Python framework for simulating transient guided and surface waves in two-dimensional layered media. It connects finite element modeling to receiver measurements, waveform archives, and frequency–wavenumber analysis in one configurable workflow.

Use it to study wave propagation, compare acquisition settings, and generate traceable synthetic data for inverse problems and machine-learning applications. Documented examples span ultrasonic inspection, composite laminates, geophysics, and bone ultrasound.

**Configure → inspect resources → simulate → record receivers → analyze and reuse**

---

## Contents

- [What it provides](#what-it-provides)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CPU and GPU execution](#cpu-and-gpu-execution)
- [Example models](#example-models)
- [Configure your own simulation](#configure-your-own-simulation)
- [Outputs and post-processing](#outputs-and-post-processing)
- [Synthetic-data workflows](#synthetic-data-workflows)
- [Model scope and numerical accuracy](#model-scope-and-numerical-accuracy)
- [Development and verification](#development-and-verification)
- [Citation and license](#citation-and-license)

## What it provides

| Capability | What you can do |
| :--- | :--- |
| **Layered models** | Simulate free plates, bonded layers, truncated halfspaces, layers on substrates, and immersed plates. |
| **Material options** | Use isotropic or anisotropic elasticity, Kelvin–Voigt viscosity, and a regularized fluid approximation. |
| **CPU and GPU solvers** | Run explicit RK4 on a single CUDA GPU through CuPy, or use SciPy/PETSc on CPUs; implicit Newmark integration is also available through PETSc. |
| **Receiver acquisition** | Specify receiver coordinates and sampling rate; evaluate displacement using finite element interpolation, including off-node positions. |
| **Integrated analysis** | Generate waveforms, B-scans, dispersion spectra, and structure-specific reports from the recorded data. |
| **Reproducible outputs** | Save resolved configurations, model metadata, execution settings, and source hashes alongside receiver histories. |

The CUDA engine keeps operators and state on the device, combines stiffness and damping products, and buffers receiver data to reduce transfer overhead. CPU and GPU paths share the finite element assembly and acquisition conventions.

## Installation

### 1. Create the environment

Use the supplied Conda environment on Linux or macOS. It installs Python 3.11/3.12, DOLFINx 0.10.0, PETSc 3.24, MPI, and the scientific Python dependencies.

```bash
git clone https://github.com/ductho-le/GuidedWaveFEM.git
cd GuidedWaveFEM
conda env create -f environment.yml
conda activate gwfem
python -m pip install --no-deps -e .
```

Check the installed command:

```bash
gwfem --version
gwfem --help
```

> [!IMPORTANT]
> Install the DOLFINx/PETSc/MPI stack through `environment.yml`. Installing the Python package with pip alone does not supply these runtime dependencies.

### 2. Optional: enable CUDA

On an NVIDIA GPU system with a compatible CUDA driver:

```bash
python -m pip install -e ".[cuda]"
```

This extra installs `cupy-cuda12x`. CUDA acceleration supports single-process explicit runs. Systems without a supported NVIDIA device, including Apple Silicon, can use the CPU solvers.

## Quick start

Run the following commands from the repository root with the `gwfem` environment active.

### 1. Inspect an example

```bash
gwfem -c configs/01_plate_aluminum_lamb.yaml --dry-run
```

The preflight report shows mesh dimensions, degrees of freedom, estimated time stepping, and memory requirements without constructing the finite element model.

### 2. Run the simulation

```bash
gwfem -c configs/01_plate_aluminum_lamb.yaml --device cpu --no-postprocess
```

This records the aluminum-plate receiver signals in `results_01_plate_aluminum_lamb/`. The supplied examples are research-scale models; inspect the preflight before running them.

### 3. Plot the saved data

```bash
gwfem --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

This separate analysis command uses built-in fonts and needs no LaTeX installation. It produces waveform and dispersion figures plus the enabled structure reports.

For simulation and plotting in one command, omit `--no-postprocess`. The shipped example configurations currently set `output.usetex: true`; change it to `false` to use built-in fonts, or install a LaTeX toolchain with `latex`, `dvipng`, and Ghostscript.

`python -m guidedwavefem` can replace `gwfem` in these commands.

## CPU and GPU execution

Choose an execution path using the same physical configuration:

```bash
# Single NVIDIA GPU, explicit integration
gwfem -c configs/01_plate_aluminum_lamb.yaml --device cuda

# Four CPU processes, distributed PETSc integration
mpiexec -n 4 gwfem -c configs/01_plate_aluminum_lamb.yaml --device cpu

# Implicit Newmark integration on CPUs
mpiexec -n 4 gwfem -c configs/01_plate_aluminum_lamb.yaml --solver implicit --device cpu
```

| Device request | Single-process explicit | MPI explicit, multiple processes | Implicit |
| :--- | :--- | :--- | :--- |
| `auto` | CuPy CUDA if available; otherwise SciPy CPU | PETSc CPU | PETSc CPU |
| `cpu` | SciPy CPU | PETSc CPU | PETSc CPU |
| `cuda` or `cuda:N` | Requested GPU; error if unavailable | Unsupported; raises an error | Ignored with a warning |

The examples select `cpu` explicitly. Use `--device auto` to request automatic selection or `--device cuda:1` to select another GPU.

**Precision.** Double precision is the default. `--precision single` enables FP32 for the single-process explicit engine; PETSc paths use FP64. Check waveform accuracy before adopting FP32 for long propagation distances.

**MPI threads.** The CLI limits library threads per rank to avoid oversubscription and respects explicit thread settings. For example:

```bash
OMP_NUM_THREADS=1 mpiexec -n 4 gwfem -c configs/01_plate_aluminum_lamb.yaml --device cpu
```

## Example models

All six examples use explicit RK4 and include material definitions, source and receiver settings, mesh counts, and output options.

| Configuration | Physical model | Application | Source |
| :--- | :--- | :--- | :--- |
| [01 · Aluminum plate](configs/01_plate_aluminum_lamb.yaml) | 3 mm free plate | Lamb-wave inspection | 1 MHz Ricker |
| [02 · CFRP laminate](configs/02_plate_cfrp.yaml) | 1 mm, eight-ply `[0/90/0/90]_s` plate | Composite guided waves | 500 kHz tone burst |
| [03 · Steel halfspace](configs/03_halfspace_steel_rayleigh.yaml) | Truncated solid halfspace | Rayleigh surface waves | 1 MHz tone burst |
| [04 · Soil over bedrock](configs/04_plate_on_halfspace_soil_bedrock.yaml) | 10 m soil layer on a substrate | Seismic surface-wave analysis | 15 Hz Ricker |
| [05 · Cortical bone](configs/05_layers_bone_qus.yaml) | Marrow/bone/tissue layers | Axial-transmission ultrasound | 500 kHz Ricker |
| [06 · Immersed plate](configs/06_plate_in_halfspaces_immersed_water.yaml) | 1.5 mm aluminum plate between water regions | Fluid-loaded guided waves | 500 kHz Ricker |

See [the example guide](configs/README.md) for material references, geometry conventions, and application-specific interpretation. Fluid regions use the approximation described under [model scope](#model-scope-and-numerical-accuracy).

## Configure your own simulation

Start from the closest example, or generate an annotated template:

```bash
gwfem --generate-config my_waveguide.yaml
gwfem -c my_waveguide.yaml --dry-run
```

YAML, TOML, and JSON are supported. Settings are resolved in this order: **built-in defaults → configuration file → CLI overrides**. Unknown or incompatible fields produce an input error.

### Configuration at a glance

| Section | Main choices |
| :--- | :--- |
| `structure` | Structure mode, layers, thicknesses, densities, and material properties |
| `geometry` | Observation-region length and lateral absorbing extensions |
| `mesh` | Exact element counts and polynomial degree |
| `solver` | Explicit/implicit integration, device, precision, time step, and duration |
| `excitation` | Ricker pulse or tone burst, frequency, width, position, and traction direction |
| `sensors` | Receiver count, pitch, offset, height, and recording rate |
| `absorbing_boundaries` | Damping in the terminating sponge layers |
| `postprocess` | Analysis options, reports, and displayed frequency/wavenumber limits |
| `output` | Destination, figure formats, history storage, compression, and field snapshots |

Numeric physical inputs use **SI units**. Supported frequency and plot-limit fields also accept unit strings such as `"20 MHz"` or `"3 rad/mm"`. Logs and figures adapt their displayed units to the problem scale.

### Settings to choose explicitly

- **Mesh:** supply `mesh.nx` and `mesh.ny_per_region` as positive element counts. Regions are ordered bottom to top; absorbing depths belong to their existing regions. Changing frequency or material properties does not automatically refine the mesh.
- **Explicit step:** supply `solver.CFL` in `(0, 1]`. The solver computes a stiffness-and-damping step limit from the assembled operators. An optional `solver.dt` is checked against that limit.
- **Recording window:** set `solver.duration` when the required arrivals are known. Automatic duration is an estimate; `solver.min_group_velocity` can provide a conservative bound for slower packets.
- **Receivers:** choose pitch, aperture, and sample rate for the modes and frequency band of interest. Display limits do not filter the source or recorded signals.

The [annotated template](configs/template.yaml) contains the full parameter reference. Older wavelength-based mesh settings must be replaced by explicit element counts; see [the changelog](CHANGELOG.md).

### Common command-line overrides

```bash
# Save a run to a separate directory
gwfem -c my_waveguide.yaml --output results_my_waveguide

# Set physical duration and explicit stability fraction
gwfem -c my_waveguide.yaml --duration 2e-4 --cfl 0.4

# Adjust displayed spectral limits
gwfem -c my_waveguide.yaml --f-max "2 MHz" --k-max "3 rad/mm"

# Record data without generating figures
gwfem -c my_waveguide.yaml --no-postprocess
```

Use `-q` for warnings only, `-v` for diagnostic logging, and `gwfem --help` for the complete option list.

## Outputs and post-processing

### What a run saves

| Output | Contents |
| :--- | :--- |
| `sensor_data.npz` | Sample times, receiver coordinates, both displacement components, model metadata, and the embedded run configuration |
| `simulation_config.yaml` | Resolved inputs and runtime metadata, including mesh, time grid, backend, precision, versions, and source hashes |
| `quick_results.png` | Quick waveform/B-scan overview when enabled |
| `full_analysis.*` | Waveforms, B-scan, and frequency–wavenumber spectra in the requested formats |
| `wave_physics/` | Structure-specific figures, numerical diagnostics, and explanatory report files |
| `guided_wave.bp` | Optional full-field displacement snapshots for ParaView (`output.write_vtx: true`) |

Receiver histories have shape **`(number of stored times, number of receivers)`**. Read them directly with NumPy:

```python
import numpy as np

with np.load("results_01_plate_aluminum_lamb/sensor_data.npz", allow_pickle=False) as data:
    times = data["times"]
    u_x = data["sensor_data_x"]
    u_y = data["sensor_data_y"]

print(times.shape, u_x.shape, u_y.shape)
```

Use the saved time array rather than assuming the requested recording rate was realized exactly. The configuration embedded in the NPZ archive is authoritative if an interrupted run leaves the YAML sidecar out of sync.

### Analyze an existing archive

Change figure settings without repeating the simulation:

```bash
gwfem --postprocess results_01_plate_aluminum_lamb/sensor_data.npz \
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
gwfem -c plot_settings.yaml \
    --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

| Report | Purpose |
| :--- | :--- |
| Structure analysis | Dispersion and phase-velocity diagnostics, component comparison, and propagation direction; enabled by default |
| Lamb modes | Theory-guided A0/S0 reconstruction for a free, single isotropic plate with archived model provenance |
| Wave characterisation | Optional packet arrival, apparent decay, pulse evolution, particle motion, and time–frequency reports |

Enable packet analysis through `postprocess.wave_characterisation.enabled` and choose a frequency band and time window containing the packet of interest. Report directories include README files explaining their methods, exclusions, and interpretation. Fitted decay is an apparent signal property, not automatically intrinsic material attenuation.

Figures use a consistent **190 mm width and 10-point text**. PNG, PDF, SVG, and EPS are supported; built-in text rendering requires no external LaTeX.

## Synthetic-data workflows

Run parameter studies by scripting repeated CLI invocations or using the Python API. Give each configuration its own output directory and retain its resolved parameters as labels alongside the receiver archive. Parameter sampling, scheduling, and ML training are handled by your campaign code.

For large collections:

- Use `--no-postprocess` during generation and analyze selected archives afterward.
- Keep a manifest linking each configuration, output path, target labels, and run status.
- Use `output.history_storage: disk` for disk-backed recording when histories are large. Solver workspace and FFT analysis still require RAM.
- Enable `output.compress_output` when reduced archive size is worth the compression cost.
- Check representative cases and parameter extremes for adequate mesh, recording duration, and receiver sampling.

The same waveform archive can support multiple preprocessing choices without another finite element solve. For reproducible time and memory measurements, see [`tools/benchmark.py`](tools/benchmark.py).

## Model scope and numerical accuracy

The framework solves **linear, two-dimensional plane-strain elastodynamics** in flat layered structures. It uses conforming quadrilateral finite elements, explicit lumped-mass RK4 or implicit consistent-mass Newmark integration, and graded absorbing layers. Solid interfaces are perfectly bonded.

Fluid regions use a **weak-shear elastic approximation**, rather than a dedicated inviscid acoustic formulation. Interface-sensitive phase and leakage predictions need comparison with an independent acoustic reference. Out-of-plane polarization, arbitrary 3D geometry, and nonlinear response are outside the current model.

The shipped inputs are starting points for studies. A stable time step or a successful run does not establish accuracy: refine mesh counts and time stepping, check absorbing-boundary effects, and ensure the observation window captures the intended arrivals.

For an early-arrival refinement comparison:

```bash
python tools/compare_resolution.py \
    --config configs/01_plate_aluminum_lamb.yaml \
    --output resolution_check --duration 40e-6 \
    --refinement-factor 1.5 --step-factor 0.5
```

This tool compares spatial and temporal refinement using a shortened record and a small receiver subset. Validate the full recording window separately. Detailed formulations and verification studies belong in the accompanying manuscript; this README focuses on operating the software.

## Development and verification

### Project layout

```text
GuidedWaveFEM/
├── src/guidedwavefem/   # Configuration, materials, solvers, storage, and analysis
├── configs/            # Annotated template and application examples
├── tests/              # Unit, numerical, physics, MPI, and CLI tests
├── tools/              # Benchmarking, refinement, and report-preview utilities
├── environment.yml     # Supported scientific-computing environment
├── pyproject.toml      # Package metadata and optional dependencies
└── CHANGELOG.md         # Version history and migration notes
```

The CPU and GPU array engines are implemented in [`solvers_gpu.py`](src/guidedwavefem/solvers_gpu.py); FEM assembly and PETSc integration are in [`solvers.py`](src/guidedwavefem/solvers.py). Configuration, material handling, storage, and analysis have separate modules under [`src/guidedwavefem/`](src/guidedwavefem/).

### Run the checks

```bash
python -m pip install -e ".[dev]"
pre-commit install

# Full test suite, or skip longer end-to-end physics runs
python -m pytest
python -m pytest -m "not slow"

# Lint and formatting
ruff check src tests tools
ruff format --check src tests tools
```

Tests cover analytical wave references, integration accuracy, receiver interpolation, backend agreement, configuration validation, output integrity, and post-processing. CUDA tests require suitable hardware; to require that backend rather than silently skip it:

```bash
GWFEM_REQUIRE_CUDA_BACKENDS=cupy python -m pytest tests/test_backend_validation.py
```

Report bugs and feature requests through [GitHub Issues](https://github.com/ductho-le/GuidedWaveFEM/issues). Include the configuration, package/environment versions, command, and error output where applicable.

## Citation and license

GuidedWaveFEM is distributed under the [MIT License](LICENSE). If you use it in research, cite the software using [`CITATION.cff`](CITATION.cff) and identify the version or commit used.

```bibtex
@software{Le_GuidedWaveFEM_2026,
  author       = {Le, Ductho},
  title        = {{GuidedWaveFEM: Unified Guided \& Surface Wave Finite Element Framework}},
  year         = {2026},
  version      = {0.3.0},
  publisher    = {GitHub},
  url          = {https://github.com/ductho-le/GuidedWaveFEM}
}
```
