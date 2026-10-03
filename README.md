# GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.8%2B-orange.svg)](https://fenicsproject.org/)
[![MPI Parallel](https://img.shields.io/badge/MPI-MPI4Py%20%7C%20PETSc-green.svg)](https://petsc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A high-performance, parallel finite element simulation and signal processing framework for elastodynamic guided wave propagation in plates, multilayered composites, semi-infinite media, and coupled fluid-solid waveguides.

Powered by **DOLFINx / FEniCSx**, **PETSc**, and **MPI**, `GuidedWaveFEM` provides an end-to-end computational pipeline—from multi-format configuration parsing, conforming multi-domain mesh generation, and dual time-integration schemes to automated 2D spatio-temporal Fourier transform ($k-\omega$) dispersion extraction.

---

## Table of Contents

- [Code Architecture & Modular Structure](#code-architecture--modular-structure)
  - [Architecture Tree](#architecture-tree)
  - [Detailed Functional Specification](#detailed-functional-specification)
- [Theoretical & Numerical Formulations](#theoretical--numerical-formulations)
  - [1. Governing Elastodynamic Equations](#1-governing-elastodynamic-equations)
  - [2. Constitutive Material Models](#2-constitutive-material-models)
  - [3. Absorbing Boundary Conditions (ALID / Sponge Layers)](#3-absorbing-boundary-conditions-alid--sponge-layers)
  - [4. Structural Waveguide Configurations](#4-structural-waveguide-configurations)
  - [5. Dual Time-Integration Solvers](#5-dual-time-integration-solvers)
- [Installation & Environment Setup](#installation--environment-setup)
- [Usage Guide & Command-Line Interface](#usage-guide--command-line-interface)
  - [1. Running a Simulation](#1-running-a-simulation)
  - [2. Parallel MPI Execution](#2-parallel-mpi-execution)
  - [3. Runtime CLI Overrides](#3-runtime-cli-overrides)
  - [4. Standalone Signal Post-Processing](#4-standalone-signal-post-processing)
  - [5. Generating Configuration Templates](#5-generating-configuration-templates)
- [Pre-Configured Validation Suite](#pre-configured-validation-suite)
- [Output Artifacts & Publication Figures](#output-artifacts--publication-figures)
- [Code Quality & Developer Workflow](#code-quality--developer-workflow)
- [Citation & License](#citation--license)

---

## Code Architecture & Modular Structure

The entire finite element simulation engine, material constitutive modeling, numerical time-steppers, and signal processing workflows are consolidated in [`guided_waves.py`](guided_waves.py). The file is partitioned into five distinct, numbered architectural sections separated by standardized double-line banners.

### Architecture Tree

```text
guided_waves.py
├── Module Docstring & High-Level Architecture
├── MPI Communicator Initialization
├── SECTION 1: Configuration Management (Loaders, Serializers, & Templates)
│   ├── load_config(...)
│   ├── dump_config(...)
│   ├── flatten_config(...)
│   ├── parse_and_resolve_config(...)
│   └── generate_template_config(...)
├── SECTION 2: Structural Geometry & Material Constitutive Modeling
│   ├── _canonicalize_mode(...)
│   ├── compute_rayleigh_speed(...)
│   ├── _normalize_region(...)
│   ├── _normalize_structure(...)
│   ├── _format_banner_box(...)
│   └── _print_structure_table(...)
├── SECTION 3: Signal Processing & 2D FFT Dispersion Analysis
│   └── plot_full_analysis(...)
├── SECTION 4: Finite Element Wave Propagation Simulation
│   └── run_simulation(...)  [Implicit Newmark-β & Explicit Lumped Verlet]
└── SECTION 5: Command-Line Interface & Application Orchestration
    └── main()
```

---

### Detailed Functional Specification

#### MPI Communicator Setup
Preceding all execution blocks, the global MPI communicator (`comm = MPI.COMM_WORLD`) initializes `rank` and `nprocs`. Parallel distributed-memory execution scales the finite element domain partitioning, assembly, and time integration across processes, while rank 0 handles formatted console reporting, serialization, and figure generation.

#### Section 1: Configuration Management (Loaders, Serializers, & Templates)
- **`load_config(path: str) -> dict[str, Any]`**  
  Multi-format parser supporting YAML (`.yaml`, `.yml`), TOML (`.toml` via Python 3.11 `tomllib`), and JSON (`.json`). Detects file extensions and provides actionable diagnostics if third-party parsers (`PyYAML`) are absent.
- **`dump_config(cfg: dict[str, Any], path: str) -> None`**  
  Sanitizes and serializes the exact resolved configuration into `simulation_config.yaml` inside the output directory. Recursively converts NumPy scalar/array types, UFL symbolic constants, and tuples to native Python primitives for complete auditability and provenance.
- **`flatten_config(cfg: dict[str, Any]) -> dict[str, Any]`**  
  Unpacks nested hierarchical configuration schemas (e.g., sections `simulation`, `geometry`, `material`, `excitation`, `sensors`, `dispersion`) into flat parameter dictionaries consumed by the finite element assembly routines.
- **`parse_and_resolve_config(args: argparse.Namespace) -> dict[str, Any]`**  
  Pure configuration resolver. Loads the mandatory user-supplied configuration file, applies runtime CLI overrides with strict precedence, validates physical constraints (positive geometry, positive wave speeds, CFL stability limits), and derives secondary numerical constants.
- **`generate_template_config(output_path: str) -> None`**  
  Generates a production-ready, fully commented template YAML configuration documenting every parameter, layer definition, material constitutive option, and numerical solver setting.

#### Section 2: Structural Geometry & Material Constitutive Modeling
- **`_canonicalize_mode(mode: str) -> str`**  
  Standardizes configuration mode strings and user aliases (e.g., `"1"`, `"single"`, `"plate"` $\to$ `"1_layer"`; `"halfspace"`, `"rayleigh"` $\to$ `"halfspace"`) into canonical identifiers.
- **`compute_rayleigh_speed(E: float, nu: float, rho: float) -> float`**  
  Solves Rayleigh's cubic secular equation:
  $$\xi^3 - 8\xi^2 + 8(3 - 2\eta)\xi - 16(1 - \eta) = 0, \quad \text{where } \xi = \left(\frac{c_R}{c_T}\right)^2, \;\; \eta = \left(\frac{c_T}{c_L}\right)^2$$
  to calculate the exact theoretical Rayleigh surface wave speed $c_R$, with analytical fallback to the Bergmann/Achenbach approximation.
- **`_normalize_region(item, default_name, role, is_halfspace, halfspace_type, y_bot) -> dict[str, Any]`**  
  Validates subdomain properties and precomputes constitutive tensors for four material models:
  1. *Isotropic elastic solid* ($E, \nu \to \lambda, \mu$).
  2. *Anisotropic elastic solid* (full 2D Voigt tensor $[C_{11}, C_{12}, C_{16}, C_{22}, C_{26}, C_{66}]$).
  3. *Viscoelastic solid* (Kelvin-Voigt stiffness tensor $\mathbf{C}$ and viscosity tensor $\mathbf{C}_v$).
  4. *Acoustic fluid* (Navier-Stokes regularization with bulk modulus $K_f = \rho_f c_f^2$ and numerical shear viscosity $\mu_f = 10^{-8} K_f$).
- **`_normalize_structure(cfg: dict[str, Any]) -> dict[str, Any]`**  
  Constructs multi-domain vertical coordinate intervals, internal interface coordinates, global bounds $[y_{\min}, y_{\max}]$, and active boundary locator tags (`top` or `bottom`) based on the waveguide configuration mode.
- **`_format_banner_box(lines: Sequence[str], title: str = "", width: int = 76) -> str`**  
  Renders formatted Unicode double-line banners with dynamic column alignment for terminal output.
- **`_print_structure_table(cfg: dict[str, Any]) -> None`**  
  Outputs a formatted table of all physical subdomains, layer thicknesses, densities, and wave speeds ($c_L, c_T, c_R, c_f$) on MPI rank 0.

#### Section 3: Signal Processing & 2D FFT Dispersion Analysis
- **`plot_full_analysis(sensor_data_filepath, output_dir, ...) -> str`**  
  Automated post-processing pipeline for sensor array data:
  1. Loads spatio-temporal displacement arrays $u_1(x, t)$ and $u_2(x, t)$ from `sensor_data.npz`.
  2. Computes 2D Discrete Fourier Transforms using Hann windowing and zero-padding:
     $$S(k, f) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x, t) \, w(x, t) \, e^{-i(kx - 2\pi f t)} \, dx \, dt$$
  3. Formats $k-f$ dispersion spectra with user-defined dynamic range clipping (e.g., 40 dB).
  4. Overlays theoretical bulk wave speed trajectories ($c_L, c_T, c_R, c_f$) as reference slopes.
  5. Exports a consolidated 2x2 master publication figure in four formats: raster (`.png`, 300 DPI) and vector (`.pdf`, `.svg`, `.eps`).

#### Section 4: Finite Element Wave Propagation Simulation
- **`run_simulation(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[str, str]`**  
  Core elastodynamic simulation driver:
  - Generates conforming rectangular meshes using `dolfinx.mesh.create_rectangle`.
  - Tags physical cell subdomains and external/internal boundary facets (`meshtags`).
  - Implements 5th-order polynomial Rayleigh damping sponge layers ($\alpha(x), \beta(x)$) for reflectionless truncation.
  - Applies Hann-windowed tone-burst surface traction excitations:
    $$t_{\text{surf}}(t) = \frac{1}{2}\left[1 - \cos\left(\frac{2\pi f_c t}{N_c}\right)\right] \sin(2\pi f_c t), \quad t \in [0, N_c / f_c]$$
  - Executes one of two time-stepping solvers:
    - **Explicit Lumped Velocity Verlet**: Diagonal mass lumping with strict positivity checks ($\mathbf{M}_L > 0$), power iteration estimation of maximum natural frequency $\omega_{\max}$, analytical CFL critical step calculation $\Delta t_{\text{crit}}$, and pre-assembled spatial load vectors achieving ~450+ steps/second.
    - **Implicit Newmark-$\beta$**: Unconditionally stable average acceleration scheme ($\beta=0.25, \gamma=0.5$) with parallel PETSc linear solvers (MUMPS direct factorisation or GMRES+BJacobi).
  - Probes displacement responses along the surface sensor array and saves `sensor_data.npz` and `simulation_config.yaml`.

#### Section 5: Command-Line Interface & Application Orchestration
- **`main() -> None`**  
  CLI entry point parsing user flags (`--config`, `--solver`, `--structure-mode`, `--output`, `--f-max`, `--k-max`, `--postprocess`, `--generate-config`). Enforces mandatory configuration specification, manages execution flow, and triggers post-processing upon simulation completion.

---

## Theoretical & Numerical Formulations

### 1. Governing Elastodynamic Equations
The boundary value problem for elastodynamics on domain $\Omega \subset \mathbb{R}^2$ with boundary $\partial\Omega = \Gamma_D \cup \Gamma_N$ is:

$$\rho \ddot{\mathbf{u}} + \mathbf{C}_{\text{damp}} \dot{\mathbf{u}} - \nabla \cdot \boldsymbol{\sigma} = \mathbf{0} \quad \text{in } \Omega \times (0, T)$$

$$\mathbf{u} = \mathbf{0} \quad \text{on } \Gamma_D, \qquad \boldsymbol{\sigma} \cdot \mathbf{n} = \mathbf{t}_{\text{surf}}(t) \quad \text{on } \Gamma_N$$

Tested against admissible virtual displacements $\mathbf{w} \in \mathcal{V}$:

$$\int_\Omega \rho \, \ddot{\mathbf{u}} \cdot \mathbf{w} \, d\Omega + \int_\Omega \boldsymbol{\sigma}_{\text{damp}}(\dot{\mathbf{u}}) : \boldsymbol{\varepsilon}(\mathbf{w}) \, d\Omega + \int_\Omega \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{w}) \, d\Omega = \int_{\Gamma_N} \mathbf{t}_{\text{surf}}(t) \cdot \mathbf{w} \, d\Gamma$$

where $\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}(\nabla \mathbf{u} + (\nabla \mathbf{u})^T)$ is the linearized strain tensor.

---

### 2. Constitutive Material Models

`GuidedWaveFEM` natively supports four constitutive laws:

1. **Isotropic Elastic Solid**:
   $$\boldsymbol{\sigma} = \lambda \, \text{tr}(\boldsymbol{\varepsilon}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}$$
2. **Anisotropic Elastic Solid**:
   Full 2D Voigt constitutive stiffness tensor ($6$ independent elastic constants):
   $$\begin{bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{bmatrix} = \begin{bmatrix} C_{11} & C_{12} & C_{16} \\ C_{12} & C_{22} & C_{26} \\ C_{16} & C_{26} & C_{66} \end{bmatrix} \begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ 2\varepsilon_{xy} \end{bmatrix}$$
3. **Viscoelastic Solid (Kelvin-Voigt)**:
   $$\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon} + \mathbf{C}_v : \dot{\boldsymbol{\varepsilon}}$$
4. **Acoustic Fluid (Regularized Elasticity)**:
   Acoustic fluids with bulk modulus $K_f = \rho_f c_f^2$ are modeled via Navier-Stokes elasticity regularization:
   $$\mu_f = 10^{-8} K_f, \qquad \lambda_f = K_f - \frac{2}{3}\mu_f$$
   This enables monolithic solid-fluid interface coupling with conforming meshes without requiring mixed displacement-pressure elements.

---

### 3. Absorbing Boundary Conditions (ALID / Sponge Layers)

To truncate semi-infinite domains and prevent boundary reflections from corrupting sensor signals, 5th-order polynomial Rayleigh damping sponge layers are applied at lateral boundaries and halfspace truncations:

$$\alpha(x) = \alpha_{\text{peak}} \left(\frac{x - x_{\text{sponge}}}{L_{\text{sponge}}}\right)^5, \qquad \beta(x) = \beta_{\text{peak}} \left(\frac{x - x_{\text{sponge}}}{L_{\text{sponge}}}\right)^5$$

$$\mathbf{C}_{\text{sponge}} = \alpha(x) \mathbf{M} + \beta(x) \mathbf{K}$$

The smooth $C^4$ profile eliminates acoustic impedance mismatch reflections at the sponge-waveguide interface while ensuring complete energy dissipation.

---

### 4. Structural Waveguide Configurations

| Configuration Mode | System Geometry | Physics & Observable Wave Modes |
| :--- | :--- | :--- |
| `1_layer` | Free single plate in vacuum | Fundamental Lamb waves ($S_0$, $A_0$), higher-order Lamb modes |
| `N_layers` | Multilayered plate laminate in vacuum | Lamb dispersion, interfacial shear coupling, laminate mode conversion |
| `halfspace` | Semi-infinite elastic halfspace | Non-dispersive Rayleigh surface acoustic waves ($c_R$) |
| `plate_on_halfspace` | Plate bonded to substrate halfspace | Leaky Lamb waves, Scholte interface waves, energy radiation |
| `plate_in_halfspaces` | Plate embedded between substrate & superstrate | Bilateral fluid loading, Stoneley interface waves, leaky guided modes |

---

### 5. Dual Time-Integration Solvers

#### Explicit Solver: Velocity Verlet with Lumped Mass
- **Row-Sum Diagonal Mass Lumping**: $\mathbf{M}_L = \text{diag}\left(\sum_j M_{ij}\right)$, strictly validated for positivity ($\mathbf{M}_L > 0$).
- **Eigenvalue Power Iteration**: Accurately computes maximum natural frequency $\omega_{\max} \approx \sqrt{\lambda_{\max}(\mathbf{M}_L^{-1}\mathbf{K})}$.
- **Analytical Critical Time Step**:
  $$\xi_{\max} = \frac{1}{2}\left(\frac{\alpha_0}{\omega_{\max}} + \beta_0 \omega_{\max} + \frac{\gamma_{\text{visco}}}{\omega_{\max}}\right)$$
  $$\Delta t_{\text{crit}} = \frac{2}{\omega_{\max}}\left(\sqrt{1 + \xi_{\max}^2} - \xi_{\max}\right), \qquad \Delta t = \text{CFL} \times \Delta t_{\text{crit}}$$
- **Pre-Assembled Spatial Force**: The spatial traction vector $\mathbf{F}_{\text{spatial}}$ is assembled once prior to time-stepping, achieving ~450+ steps/second throughput.

#### Implicit Solver: Newmark-$\beta$
- **Unconditionally Stable**: Average acceleration scheme ($\beta = 0.25, \gamma = 0.5$).
- **Parallel PETSc KSP Solvers**: Direct parallel LU factorisation (MUMPS) for small-to-medium meshes, or iterative Krylov Conjugate Gradient with Geometric Algebraic Multigrid (CG + GAMG) for large-scale systems.

---

## Installation & Environment Setup

`GuidedWaveFEM` requires Python 3.11+, DOLFINx 0.8+, PETSc, and MPICH.

```bash
# 1. Clone the repository
git clone https://github.com/ductho/GuidedWaveFEM.git
cd GuidedWaveFEM

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate fenicsx-waves
```

---

## Usage Guide & Command-Line Interface

All simulations require an explicit configuration file via `-c` / `--config`.

### 1. Running a Simulation
```bash
python guided_waves.py -c configs/01_single_layer_aluminum.yaml
```

### 2. Parallel MPI Execution
Scale across multi-core processors or HPC clusters using `mpirun`:
```bash
mpirun -np 4 python guided_waves.py -c configs/03_multilayer_bone_aluminum.yaml
```

### 3. Runtime CLI Overrides
Command-line flags override configuration file settings dynamically:
```bash
# Override time-integration solver to implicit Newmark-beta
python guided_waves.py -c configs/01_single_layer_aluminum.yaml --solver implicit

# Override output destination directory
python guided_waves.py -c configs/04_halfspace_rayleigh_aluminum.yaml --output results_rayleigh_run

# Override dispersion frequency and wavenumber plotting limits
python guided_waves.py -c configs/01_single_layer_aluminum.yaml --f-max 3.0 --k-max 4.0
```

### 4. Standalone Signal Post-Processing
Reprocess existing simulation data without re-running the finite element solver:
```bash
python guided_waves.py --postprocess results_guided_waves_explicit/sensor_data.npz --output reprocessed_figures
```

### 5. Generating Configuration Templates
Generate a fully annotated template YAML configuration file:
```bash
python guided_waves.py --generate-config my_custom_waveguide.yaml
```

---

## Pre-Configured Validation Suite

Thirteen pre-configured validation cases covering isotropic, anisotropic, viscoelastic, and biological waveguides are provided in [`configs/`](configs):

| Config File | Mode | Physical System & Materials | Key Phenomena |
| :--- | :--- | :--- | :--- |
| [`01_single_layer_aluminum.yaml`](configs/01_single_layer_aluminum.yaml) | `1_layer` | 1 mm Aluminum plate in vacuum | Fundamental Lamb waves ($S_0$, $A_0$) |
| [`01_single_layer_aluminum.toml`](configs/01_single_layer_aluminum.toml) | `1_layer` | 1 mm Aluminum plate (TOML format) | Standard library `tomllib` parsing |
| [`01_single_layer_aluminum.json`](configs/01_single_layer_aluminum.json) | `1_layer` | 1 mm Aluminum plate (JSON format) | Automated workflow & API schema |
| [`02_single_layer_cortical_bone_viscoelastic.yaml`](configs/02_single_layer_cortical_bone_viscoelastic.yaml) | `1_layer` | 3 mm Cortical Bone (Kelvin-Voigt) | Material damping & wave attenuation |
| [`03_multilayer_bone_aluminum.yaml`](configs/03_multilayer_bone_aluminum.yaml) | `N_layers` | 2 mm Bone + 1 mm Aluminum coating | Heterogeneous laminate, mode conversion |
| [`04_halfspace_rayleigh_aluminum.yaml`](configs/04_halfspace_rayleigh_aluminum.yaml) | `halfspace` | 40 mm Aluminum halfspace | Non-dispersive Rayleigh waves ($c_R \approx 2906\text{ m/s}$) |
| [`05_plate_on_halfspace_bone_marrow.yaml`](configs/05_plate_on_halfspace_bone_marrow.yaml) | `plate_on_halfspace` | 2 mm Bone on 12 mm Marrow substrate | Leaky Lamb waves, substrate radiation |
| [`06_plate_in_halfspaces_bone_tissue_marrow.yaml`](configs/06_plate_in_halfspaces_bone_tissue_marrow.yaml) | `plate_in_halfspaces` | 2 mm Bone embedded in Tissue & Marrow | Dual-sponge truncation, interface modes |
| [`07_plate_on_water_halfspace.yaml`](configs/07_plate_on_water_halfspace.yaml) | `plate_on_halfspace` | 1.5 mm Aluminum on Water substrate | Scholte interface waves ($c_S \approx 1470\text{ m/s}$) |
| [`08_plate_immersed_in_water.yaml`](configs/08_plate_immersed_in_water.yaml) | `plate_in_halfspaces` | 1.5 mm Aluminum immersed in Water | Immersion testing, bilateral radiation |
| [`09_bonedc_case1_cortical_bone.yaml`](configs/09_bonedc_case1_cortical_bone.yaml) | `1_layer` | 4 mm Cortical Bone (BoneDC Case 1) | Semi-Analytical Finite Element benchmark |
| [`10_bonedc_case2_cb_st_fluid.yaml`](configs/10_bonedc_case2_cb_st_fluid.yaml) | `N_layers` | 4 mm Bone + 1 mm Soft Tissue fluid | Solid-fluid coupled waveguide validation |
| [`11_bonedc_case3_st_cb_mr_fluid.yaml`](configs/11_bonedc_case3_st_cb_mr_fluid.yaml) | `N_layers` | 1 mm Marrow + 4 mm Bone + 1 mm Soft Tissue | Full tri-layer biological benchmark |

---

## Output Artifacts & Publication Figures

Each simulation automatically generates structured data and publication-quality figures:

1. **`sensor_data.npz`**: Compressed NumPy binary archive containing:
   - `time`: Discrete simulation time array ($t$).
   - `sensor_x`: Coordinate array of sensor locations ($x$).
   - `sensor_u1`: In-plane displacement time-histories ($u_1(x_i, t)$).
   - `sensor_u2`: Out-of-plane displacement time-histories ($u_2(x_i, t)$).
   - `metadata`: Serialized geometry, material properties, wave speeds, and layer configurations.
2. **`simulation_config.yaml`**: Serialized snapshot of the exact resolved configuration for provenance.
3. **`quick_results.png`**: Preliminary overview showing excitation waveform, displacement waterfall, and B-scan.
4. **`full_analysis.{png, pdf, svg, eps}`**: Consolidated 2x2 master publication figure generated in raster (`.png`, 300 DPI) and vector formats (`.pdf`, `.svg`, `.eps`):
   - **Panel (a)**: Out-of-plane waveform traces ($u_2$, nm) at first, middle, and last sensors.
   - **Panel (b)**: Spatio-temporal B-scan image of wave propagation along the array.
   - **Panel (c)**: 2D FFT wavenumber-frequency ($k-f$) dispersion spectrum for in-plane displacement ($u_1$).
   - **Panel (d)**: 2D FFT wavenumber-frequency ($k-f$) dispersion spectrum for out-of-plane displacement ($u_2$).
   - **Analytical Overlays**: Bulk wave speed trajectories ($c_L, c_T, c_R, c_f$) plotted over spectra.

---

## Code Quality & Developer Workflow

The codebase strictly adheres to PEP 8 standards, PEP 484/585/604 type hinting, and complete NumPy/SciPy docstrings.

```bash
# Verify Python syntax and bytecode compilation
python -m py_compile guided_waves.py

# Run static analysis and linting (0 errors, 0 warnings)
ruff check guided_waves.py

# Format code according to strict PEP 8 standards
ruff format guided_waves.py
```

---

## Citation & License

This project is licensed under the MIT License. If you use `GuidedWaveFEM` in your academic research or industrial publications, please cite this repository.
