# GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.10.0-orange.svg)](https://fenicsproject.org/)
[![MPI Parallel](https://img.shields.io/badge/MPI-MPI4Py%20%7C%20PETSc-green.svg)](https://petsc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A parallel finite element simulation and signal processing framework for **2D plane-strain elastodynamic guided wave propagation** in plates, multilayered composites, truncated semi-infinite media, and regularized fluid–solid waveguides.

Powered by **DOLFINx / FEniCSx**, **PETSc**, and **MPI**, `GuidedWaveFEM` provides an end-to-end computational pipeline—from multi-format configuration parsing, conforming multi-domain mesh generation, and dual time-integration schemes to exact-point sensor measurements and automated 2D spatio-temporal Fourier transform ($k$–$\omega$) dispersion analysis.

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
  - [6. Mesh, Time-Step & Output Controls](#6-mesh-time-step--output-controls)
- [Pre-Configured Examples & Benchmark Setups](#pre-configured-examples--benchmark-setups)
- [Output Artifacts & Publication Figures](#output-artifacts--publication-figures)
- [Code Quality & Developer Workflow](#code-quality--developer-workflow)
- [Version Compatibility & Numerical Changes](#version-compatibility--numerical-changes)
- [Citation & License](#citation--license)

---

## Code Architecture & Modular Structure

`GuidedWaveFEM` is organized following modern Python packaging standards (`src` layout) under `src/guidedwavefem/`. The framework cleanly separates configuration management, material constitutive models, testable numerical algorithms, finite element wave propagation solvers, spectral signal processing, and CLI execution into dedicated modules.

### Architecture Tree

```text
GuidedWaveFEM/
├── src/
│   └── guidedwavefem/
│       ├── __init__.py         # Package entry point & public API exports
│       ├── __main__.py         # Direct execution entry point (`python -m guidedwavefem`)
│       ├── config.py           # Multi-format loaders (YAML/TOML/JSON), resolution, template generator
│       ├── materials.py        # Voigt tensors, Kelvin-Voigt, Rayleigh speed, structure normalization
│       ├── numerics.py         # RK4 integrator, assembled-operator bounds, exact PointSensors
│       ├── solvers.py          # Conforming mesh generation, ALID sponges, RK4 & Newmark-β time loops
│       ├── analysis.py         # 2D spatio-temporal FFT, Hann tapering, 2x2 publication figures
│       └── cli.py              # Argparse, preflight estimator (--dry-run), main() orchestration
├── configs/                    # YAML, TOML, and JSON examples
├── tests/                      # Numerical and serial/MPI regressions
├── .github/workflows/tests.yml # Continuous integration
├── environment.yml             # Supported conda numerical stack
└── pyproject.toml              # Package, CLI, test, and lint configuration
```

---

### Detailed Functional Specification

#### MPI Communicator Setup

The global communicator (`comm = MPI.COMM_WORLD`) initializes `rank` and `nprocs`. Domain partitioning, assembly, and time integration are distributed across processes. Each rank records only its owned sensor columns; rank zero gathers complete histories and handles serialization and figures. Runtime exceptions in decorated simulation/CLI entry points abort multi-rank execution to prevent peers remaining blocked at collectives.

#### Section 1: Configuration Management (Loaders, Serializers, & Templates)

- **`load_config(config_path: str) -> dict[str, Any]`**
  Multi-format parser supporting YAML (`.yaml`, `.yml`), TOML (`.toml`, via Python 3.11 `tomllib`), and JSON (`.json`). Checks that the input root is a mapping and reports missing parsers or unsupported extensions.
- **`dump_config(cfg_dict: dict[str, Any], out_path: str) -> None`**
  Serializes resolved configuration and runtime provenance into `simulation_config.yaml`. Converts NumPy arrays/scalars and tuples to native types while preserving booleans.
- **`flatten_config(cfg_in: dict[str, Any]) -> dict[str, Any]`**
  Unpacks hierarchical sections (`solver`, `geometry`, `structure`, `absorbing_boundaries`, `excitation`, `mesh`, `sensors`, `postprocess`, and `output`) into solver parameters. Flat configuration dictionaries are also accepted.
- **`parse_and_resolve_config(config_path=None, args=None, custom_cfg=None, validate_simulation=True) -> dict[str, Any]`**
  Applies defaults and CLI overrides, validates geometry, sampling, source placement, and solver-specific integration settings, and resolves substrate/superstrate aliases. Material tensors are checked during region normalization.
- **`generate_template_config(output_path: str) -> None`**
  Generates a commented single-layer YAML template covering the principal material, mesh, integration, sensor, and output settings. The supplied configurations demonstrate multilayer and halfspace definitions.

#### Section 2: Structural Geometry & Material Constitutive Modeling

- **`_canonicalize_mode(mode_str: str | None) -> str`**
  Standardizes aliases such as `"1"` or `"single"` to `"1_layer"`, and `"rayleigh"` to `"halfspace"`.
- **`compute_rayleigh_speed(c_T: float, c_L: float | None = None, nu: float | None = None) -> float`**
  Solves Rayleigh's cubic secular equation for an isotropic elastic halfspace:
  $$\xi^3 - 8\xi^2 + 8(3 - 2\eta)\xi - 16(1 - \eta) = 0, \qquad \xi = \left(\frac{c_R}{c_T}\right)^2, \quad \eta = \left(\frac{c_T}{c_L}\right)^2.$$
  Selects a physical root in $(0,1)$, with an approximate fallback if needed.
- **`_normalize_region(item, default_name, role, is_halfspace, halfspace_type, y_bot) -> dict[str, Any]`**
  Validates subdomain properties and constructs parameters for isotropic elasticity, anisotropic elasticity, Kelvin–Voigt viscoelasticity, or regularized elastic fluids. Checks positive-definite stiffness and positive-semidefinite viscosity tensors; calculates wave-speed values or conservative anisotropic speed bounds.
- **`_normalize_structure(mode=None, single_layer_input=None, layers_input=None, halfspace_input=None, bottom_hs_input=None, top_hs_input=None) -> dict[str, Any]`**
  Constructs vertical subdomain intervals, interfaces, global bounds $[y_{\min},y_{\max}]$, and halfspace sponge depths.
- **`_format_banner_box(lines: Sequence[str], min_width: int = 60) -> str`**
  Renders Unicode double-line terminal banners with dynamically sized columns.
- **`_print_structure_table(summary: dict[str, Any], rank: int = 0, f0: float | None = None) -> None`**
  Reports subdomains, thicknesses, densities, wave-speed summaries, and sensor/excitation locations on rank zero.

#### Section 3: Signal Processing & 2D FFT Dispersion Analysis

- **`plot_full_analysis(sensor_data_filepath, output_dir=None, ...) -> str`**
  Automated sensor-array analysis:
  1. Loads and validates displacement histories and uniformly spaced coordinates.
  2. Applies a fourth-order Butterworth bandpass using zero-phase forward/backward filtering.
  3. Computes separate component spectra with separable Hann windows and twofold zero-padding:
     $$S(k,f)=\int\!\!\int u(x,t)\,w(x,t)\,e^{-i(kx-2\pi ft)}\,dx\,dt.$$
  4. Selects the forward-propagating branch using positive temporal and negative spatial FFT frequencies. Displays physical FFT-bin coordinates with a 40 dB range, normalized separately for each displacement component.
  5. Adds applicable speed references and exports a consolidated 2×2 figure as `.png`, `.pdf`, `.svg`, and `.eps`.

Zero-padding interpolates the spectral display; physical frequency and wavenumber resolution remain governed by record duration and sensor aperture. Display limits do not change the FFT coordinates or extend the sampled bandwidth.

#### Section 4: Finite Element Wave Propagation Simulation

- **`run_simulation(args=None, cfg=None) -> tuple[str, str]`**
  Core elastodynamic driver:
  - Generates quadrilateral meshes using `dolfinx.mesh.create_rectangle`, then aligns horizontal grid lines with all material interfaces.
  - Tags subdomain cells and exterior/interior excitation facets.
  - Assembles mass, stiffness, Kelvin–Voigt damping, and spatially varying sponge damping.
  - Applies a Gaussian spatial traction profile multiplied by a Ricker pulse:
    $$r(t)=\left[1-2\pi^2 f_c^2(t-t_0)^2\right]e^{-\pi^2 f_c^2(t-t_0)^2}, \qquad t_0=\frac{1.5}{f_c}.$$
  - Executes explicit lumped-mass RK4 or implicit Newmark-$\beta$ integration.
  - Evaluates displacements at exact sensor coordinates and writes histories and effective runtime settings.

#### Section 5: Command-Line Interface & Application Orchestration

- **`main() -> None`**  
  Parses `--config`, `--solver`, `--structure-mode`, `--output`, numerical controls (`--dry-run`, `--max-frequency`, `--duration`, `--dt`, `--cfl`), frequency/wavenumber limits, `--postprocess`, `--no-postprocess`, and `--generate-config`. A configuration is required for simulation; existing sensor files can be postprocessed independently.

#### Reusable Numerical Utilities (`guidedwavefem.numerics`)

| Utility | Responsibility |
| :--- | :--- |
| `rk4_step(u, v, t, dt, acceleration)` | Four-stage integration with consistent displacement, velocity, and forcing evaluations |
| `operator_bound(matrix, mass)` | Conservative spectral upper bound from distributed absolute row sums of the lumped-mass-scaled operator |
| `time_grid(duration, requested_dt)` | Fits an integer step count to the requested final time |
| `PointSensors` | Exact FE point evaluation, single-owner MPI assignment, and root-only history gathering |
| `validate_tensor(values, name, positive=True)` | Validates six-coefficient Voigt tensors and material energy conditions |
| `dispersion_spectrum(values, times, positions)` | Validates uniform sampling and returns forward-wave power on physical FFT bins |

---

## Theoretical & Numerical Formulations

### 1. Governing Elastodynamic Equations

The plane-strain model is posed on $\Omega\subset\mathbb{R}^2$, with in-plane displacement $\mathbf{u}=(u_1,u_2)$. The second component is transverse to the plate, not an out-of-plane degree of freedom of the 2D FE domain.

The weak momentum balance is

$$\int_\Omega \rho\ddot{\mathbf{u}}\cdot\mathbf{w}\,d\Omega + \int_\Omega \alpha(\mathbf{x})\rho\dot{\mathbf{u}}\cdot\mathbf{w}\,d\Omega + \int_\Omega \left[\boldsymbol{\sigma}_e(\mathbf{u})+\beta(\mathbf{x})\boldsymbol{\sigma}_e(\dot{\mathbf{u}})+\boldsymbol{\sigma}_v(\dot{\mathbf{u}})\right]:\boldsymbol{\varepsilon}(\mathbf{w})\,d\Omega = \ell(\mathbf{w},t),$$

where

$$\boldsymbol{\varepsilon}(\mathbf{u})=\frac{1}{2}\left(\nabla\mathbf{u}+(\nabla\mathbf{u})^T\right).$$

Here $\boldsymbol{\sigma}_e$ is elastic stress, $\boldsymbol{\sigma}_v$ is Kelvin–Voigt viscous stress when present, and $\ell$ represents the applied traction. Exterior boundaries are traction-free except for excitation. Embedded-plate excitation is applied on the internal plate–superstrate interface. Domain truncation uses damping sponges rather than prescribed zero-displacement boundaries.

The assembled system has the form

$$\mathbf{M}\ddot{\mathbf{u}}+\mathbf{C}\dot{\mathbf{u}}+\mathbf{K}\mathbf{u}=\mathbf{F}(t).$$

---

### 2. Constitutive Material Models

`GuidedWaveFEM` supports four material descriptions:

1. **Isotropic Elastic Solid**:
   $$\boldsymbol{\sigma}_e=\lambda\,\operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{I}+2\mu\boldsymbol{\varepsilon}, \qquad \mu=\frac{E}{2(1+\nu)}, \quad \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.$$
   Parameters must satisfy $E>0$, $\rho>0$, and $-1<\nu<0.5$.

2. **Anisotropic Elastic Solid**:
   Full symmetric 2D Voigt stiffness with six independent coefficients:
   $$\begin{bmatrix}\sigma_{xx}\\\sigma_{yy}\\\sigma_{xy}\end{bmatrix}=\begin{bmatrix}C_{11}&C_{12}&C_{16}\\C_{12}&C_{22}&C_{26}\\C_{16}&C_{26}&C_{66}\end{bmatrix}\begin{bmatrix}\varepsilon_{xx}\\\varepsilon_{yy}\\2\varepsilon_{xy}\end{bmatrix}.$$
   The stiffness matrix must be positive definite. The mesh-sizing speed bounds use the orthonormal Kelvin representation; they are not exact directional longitudinal/shear speeds.

3. **Viscoelastic Solid (Kelvin–Voigt)**:
   $$\boldsymbol{\sigma}=\mathbf{C}:\boldsymbol{\varepsilon}+\mathbf{C}_v:\dot{\boldsymbol{\varepsilon}}.$$
   Elastic coefficients have units of Pa; viscosity coefficients have units of Pa·s. The viscosity tensor must be positive semidefinite.

4. **Fluid Approximation (Regularized Elasticity)**:
   A supplied acoustic bulk modulus $K_f=\rho_f c_f^2$ defines
   $$\mu_f=\epsilon_f K_f, \qquad \lambda_f=K_f-\frac{2}{3}\mu_f, \qquad \epsilon_f=\texttt{fluid\_shear\_reg}.$$
   The default is $\epsilon_f=10^{-8}$, with accepted values in $(0,0.01]$. The regularized longitudinal speed is $\sqrt{(K_f+4\mu_f/3)/\rho_f}$.

   This enables monolithic coupling on a conforming displacement mesh, but represents a small-shear elastic approximation, **not Navier–Stokes viscosity or mixed inviscid acoustics**. The shared space imposes tangential as well as normal displacement continuity; inviscid acoustic interfaces generally do not require tangential continuity. Quantitative fluid-loaded studies therefore require independent acoustic–elastic reference comparisons and mesh/regularization sensitivity checks.

---

### 3. Absorbing Boundary Conditions (ALID / Sponge Layers)

Fifth-order polynomial Rayleigh damping sponges attenuate waves before they reach lateral and halfspace truncation boundaries. For inward-to-outward normalized distance $s\in[0,1]$,

$$\alpha(s)=\alpha_{\mathrm{peak}}s^5, \qquad \beta(s)=\beta_{\mathrm{peak}}s^5.$$

Their assembled contribution is

$$C^{\mathrm{sponge}}_{ij}=\int_\Omega\alpha(\mathbf{x})\rho\,\boldsymbol{\phi}_j\cdot\boldsymbol{\phi}_i\,d\Omega+\int_\Omega\beta(\mathbf{x})\boldsymbol{\sigma}_e(\boldsymbol{\phi}_j):\boldsymbol{\varepsilon}(\boldsymbol{\phi}_i)\,d\Omega.$$

Horizontal and vertical sponge indicators add in overlapping regions. Both solvers default to $\alpha_{\mathrm{peak}}=10^7\,\mathrm{s}^{-1}$ and $\beta_{\mathrm{peak}}=10^{-8}\,\mathrm{s}$; configured values are preserved rather than automatically clipped.

The smooth onset reduces abrupt damping transitions. It does not guarantee reflectionless truncation or complete absorption. Reflection levels depend on frequency, incidence, material properties, sponge dimensions, and discretization, and should be measured for the intended application.

---

### 4. Structural Waveguide Configurations

| Configuration Mode | System Geometry | Physics & Target Wave Phenomena |
| :--- | :--- | :--- |
| `1_layer` | Free single plate in vacuum | Fundamental and higher-order Lamb waves |
| `N_layers` | Multilayered plate laminate | Layer-dependent guided dispersion and interfacial coupling |
| `halfspace` | Finite-depth approximation with bottom sponge | Rayleigh surface waves for elastic media |
| `plate_on_halfspace` | Plate on a truncated substrate | Substrate loading, radiation, and interface-wave behavior |
| `plate_in_halfspaces` | Plate between truncated substrate and superstrate | Bilateral loading, leaky guided waves, and interface-wave behavior |

Specific interface modes depend on material parameters and boundary conditions; selecting a geometry does not guarantee that every named mode exists. Fluid-loaded configurations use the approximation described above.

---

### 5. Dual Time-Integration Solvers

#### Explicit Solver: Classical RK4 with Lumped Mass

- **Row-Sum Diagonal Mass Lumping**:
  $$\mathbf{M}_L=\operatorname{diag}\left(\sum_j M_{ij}\right), \qquad (M_L)_{ii}>0.$$
- **Consistent Four-Stage Integration**: Classical fourth-order Runge–Kutta advances displacement and velocity, evaluating $\mathbf{M}_L^{-1}[\mathbf{F}(t)-\mathbf{K}\mathbf{u}-\mathbf{C}\mathbf{v}]$ at every stage. This preserves fourth-order temporal accuracy for sufficiently smooth semidiscrete solutions, including velocity-dependent damping.
- **Conservative Operator Bounds**:
  $$B_K=\max_i\sum_j\left|(\mathbf{M}_L^{-1}\mathbf{K})_{ij}\right|, \qquad B_C=\max_i\sum_j\left|(\mathbf{M}_L^{-1}\mathbf{C})_{ij}\right|.$$
  The default requested step is
  $$\Delta t_{\mathrm{limit}}=\frac{1}{\max(\sqrt{B_K},B_C)}, \qquad \Delta t_{\mathrm{requested}}=\mathrm{CFL}\,\Delta t_{\mathrm{limit}}, \quad 0<\mathrm{CFL}\le1.$$
  For symmetric positive-(semi)definite assembled operators, the first-order-system eigenvalues lie in the left half-plane with magnitude bounded by this denominator. The left half of the unit disk lies inside RK4's stability region. This conservative spectral criterion includes actual damping and overlapping sponges; it is not an accuracy estimate. Undamped rigid modes retain physical constant-velocity motion.
- **Reproducible Step Selection**: No random power iteration is used. An optional requested step above the conservative limit is rejected; the accepted step is fitted to the exact final time.
- **Pre-Assembled Spatial Force**: The spatial traction vector is assembled once. Each RK4 step requires four stiffness and four damping matrix-vector products. Strong damping can make explicit integration expensive.

#### Implicit Solver: Newmark-$\beta$

- **Average Acceleration Defaults**: $\beta_N=0.25$, $\gamma_N=0.5$, providing second-order accuracy and unconditional linear stability for the standard admissible system.
- **Parameter Validation**: Other accepted parameters satisfy $\gamma_N\ge0.5$ and $\beta_N\ge(\gamma_N+0.5)^2/4$. Values of $\gamma_N\ne0.5$ generally lose second-order accuracy.
- **PETSc KSP Solvers**: Serial LU or parallel MUMPS for smaller systems, GMRES/block-Jacobi fallback when parallel direct setup is unavailable, and CG/GAMG for larger systems. Solver nonconvergence raises an error.
- **Consistent Mass**: The implicit branch uses $\mathbf{M}$ rather than $\mathbf{M}_L$. Explicit/implicit agreement should be assessed under refinement, not expected to be exact on a fixed mesh.

---

## Installation & Environment Setup

The supported conda environment uses **Python 3.11**, **DOLFINx 0.10.0**, **PETSc/petsc4py 3.24**, and **MPICH**, together with NumPy, SciPy, Matplotlib, PyYAML, pytest, and Ruff.

```bash
# 1. Clone the repository
git clone https://github.com/ductho-le/GuidedWaveFEM.git
cd GuidedWaveFEM

# 2. Create and activate the numerical environment
conda env create -f environment.yml
conda activate fenicsx-waves

# 3. Optionally install the guided-waves CLI in editable mode
python -m pip install --no-deps -e .
```

Install the FEM/MPI stack through conda; the Python package metadata does not install that stack. `environment.yml` pins the principal FEM APIs but is not a complete platform lockfile. For archived benchmark runs, save an explicit environment specification:

```bash
conda list --explicit > environment-explicit.txt
```

---

## Usage Guide & Command-Line Interface

All simulations require an explicit configuration file via `-c` / `--config`. Simulations can be launched using the installed CLI `guided-waves` or module invocation `python -m guidedwavefem`.

### 1. Running a Simulation

```bash
guided-waves -c configs/01_single_layer_aluminum.yaml
# Or alternatively:
python -m guidedwavefem -c configs/01_single_layer_aluminum.yaml
```

### 2. Parallel MPI Execution

Distribute simulation work across processes:

```bash
mpiexec -n 4 guided-waves -c configs/03_multilayer_bone_aluminum.yaml
# Or:
mpiexec -n 4 python -m guidedwavefem -c configs/03_multilayer_bone_aluminum.yaml
```

### 3. Runtime CLI Overrides

Command-line options override configuration settings:

```bash
# Preflight dry-run: preview mesh sizing, DOF counts, step count, and memory without allocating
guided-waves -c configs/01_single_layer_aluminum.yaml --dry-run

# Use implicit Newmark integration
guided-waves -c configs/01_single_layer_aluminum.yaml --solver implicit

# Select an output directory
guided-waves -c configs/04_halfspace_rayleigh_aluminum.yaml --output results_rayleigh_run

# Override numerical controls (duration, mesh frequency, CFL, dt)
guided-waves -c configs/01_single_layer_aluminum.yaml --duration 5.0e-5 --max-frequency 1.0e6 --cfl 0.4

# Set dispersion display limits (MHz and rad/mm)
guided-waves -c configs/01_single_layer_aluminum.yaml --f-max 3.0 --k-max 4.0

# Run the simulation and save data without generating figures
guided-waves -c configs/01_single_layer_aluminum.yaml --no-postprocess
```

### 4. Standalone Signal Post-Processing

Reprocess existing measurements without rerunning the FEM solver:

```bash
guided-waves --postprocess results_case01_aluminum_1layer/sensor_data.npz --output reprocessed_figures
```

Saved plotting limits are honored unless explicitly overridden. Postprocessing requires at least 32 time samples and three uniformly spaced sensors. Component spectra are normalized separately; their color scales do not establish absolute cross-component amplitude ratios.

### 5. Generating Configuration Templates

```bash
guided-waves --generate-config my_custom_waveguide.yaml
```

### 6. Mesh, Time-Step & Output Controls

The principal numerical controls can be specified hierarchically:

```yaml
solver:
  type: explicit           # Lumped RK4; alternatively implicit Newmark
  CFL: 0.5
  time_factor: 1.2         # Used when duration is omitted
  dt: null                 # Optional maximum requested step [s]
  duration: null           # Optional physical duration [s]
  beta: 0.25              # Implicit Newmark parameter
  gamma: 0.5              # Implicit Newmark parameter

mesh:
  elems_per_wavelength: 10
  min_elems_x: 100
  min_elems_y: 4
  degree: 2
  max_frequency: null      # [Hz]; defaults to 3*f_center

output:
  output_dir: results_custom
  write_vtx: false
  target_snapshots: 200
  quick_plot: true
```

The actual step is reduced as necessary to fit an integer step count to the exact duration. Global `mesh.min_elems_y` applies unless a region supplies its own override. Mesh sizing uses the selected maximum frequency and bulk-speed bounds; guided-mode convergence must still be established. Set duration explicitly when slow packets need longer records. Sensors must lie within the mesh and are evaluated at their exact coordinates.

---

## Pre-Configured Examples & Benchmark Setups

Thirteen configuration files represent **eleven distinct numbered physical setups**, with YAML/TOML/JSON variants for Case 01. They cover isotropic, anisotropic, viscoelastic, and biological waveguides in [`configs/`](configs).

| Config File | Mode | Physical System & Materials | Target Phenomena / Purpose |
| :--- | :--- | :--- | :--- |
| [`01_single_layer_aluminum.yaml`](configs/01_single_layer_aluminum.yaml) | `1_layer` | 1 mm aluminum plate in vacuum | Fundamental and higher-order Lamb waves |
| [`01_single_layer_aluminum.toml`](configs/01_single_layer_aluminum.toml) | `1_layer` | Case 01, TOML representation | Standard-library `tomllib` parsing |
| [`01_single_layer_aluminum.json`](configs/01_single_layer_aluminum.json) | `1_layer` | Case 01, JSON representation | JSON workflow integration |
| [`02_single_layer_cortical_bone_viscoelastic.yaml`](configs/02_single_layer_cortical_bone_viscoelastic.yaml) | `1_layer` | 3 mm cortical bone, Kelvin–Voigt | Material damping and attenuation |
| [`03_multilayer_bone_aluminum.yaml`](configs/03_multilayer_bone_aluminum.yaml) | `N_layers` | 2 mm bone + 1 mm aluminum | Heterogeneous laminate dispersion |
| [`04_halfspace_rayleigh_aluminum.yaml`](configs/04_halfspace_rayleigh_aluminum.yaml) | `halfspace` | 40 mm truncated aluminum halfspace | Rayleigh surface-wave reference setup |
| [`05_plate_on_halfspace_bone_marrow.yaml`](configs/05_plate_on_halfspace_bone_marrow.yaml) | `plate_on_halfspace` | 2 mm bone on 12 mm marrow substrate | Substrate loading and radiation |
| [`06_plate_in_halfspaces_bone_tissue_marrow.yaml`](configs/06_plate_in_halfspaces_bone_tissue_marrow.yaml) | `plate_in_halfspaces` | 2 mm bone between tissue and marrow | Dual-sponge truncation and interface behavior |
| [`07_plate_on_water_halfspace.yaml`](configs/07_plate_on_water_halfspace.yaml) | `plate_on_halfspace` | 1.5 mm aluminum on regularized water | Fluid loading and interface-wave study |
| [`08_plate_immersed_in_water.yaml`](configs/08_plate_immersed_in_water.yaml) | `plate_in_halfspaces` | 1.5 mm aluminum between regularized water domains | Bilateral loading and radiation |
| [`09_bonedc_case1_cortical_bone.yaml`](configs/09_bonedc_case1_cortical_bone.yaml) | `1_layer` | 4 mm cortical bone, BoneDC Case 1 | Independent dispersion comparison setup |
| [`10_bonedc_case2_cb_st_fluid.yaml`](configs/10_bonedc_case2_cb_st_fluid.yaml) | `N_layers` | 4 mm bone + 1 mm regularized tissue fluid | Layered solid–fluid comparison setup |
| [`11_bonedc_case3_st_cb_mr_fluid.yaml`](configs/11_bonedc_case3_st_cb_mr_fluid.yaml) | `N_layers` | 1 mm marrow + 4 mm bone + 1 mm tissue | Tri-layer biological comparison setup |

These inputs are examples and intended benchmark setups, not independently certified validation results. Quantitative Rayleigh–Lamb, anisotropic multilayer, attenuation, acoustic-interface, absorber-reflection, and convergence comparisons remain distinct scientific verification tasks.

---

## Output Artifacts & Publication Figures

Simulations save structured measurements and provenance; figures are generated when postprocessing is enabled.

1. **`sensor_data.npz`**: NumPy archive containing:
   - `times`: Discrete simulation times ($t$).
   - `sensor_x`, `sensor_y`, `sensor_coords`: Actual measurement coordinates.
   - `sensor_data_x`: In-plane longitudinal displacement histories ($u_1(x_i,t)$).
   - `sensor_data_y`: Plate-transverse displacement histories ($u_2(x_i,t)$).
   - Additional named fields: Geometry, material summaries, wave speeds, layer information, source position, and plotting limits.
2. **`simulation_config.yaml`**: Resolved settings and a `runtime` section containing:
   - Effective damping coefficients and actual time step, duration, and step count.
   - Mesh dimensions, elements per region, global DOFs, and MPI rank count.
   - Integrator name, mesh-sizing frequency, and explicit operator bounds when applicable.
   - Python, NumPy, DOLFINx, and PETSc versions, plus SHA-256 hashes of the solver source modules.
3. **`quick_results.png`**: Excitation waveform, displacement waterfall, and B-scan overview; controlled by `output.quick_plot` and the postprocessing switch.
4. **`full_analysis.{png,pdf,svg,eps}`**: Consolidated 2×2 figure with 300-DPI PNG and vector-format exports:
   - **Panel (a)**: Selected $u_2$ waveforms at the first, middle, and last sensors.
   - **Panel (b)**: Spatio-temporal $u_2$ B-scan with applicable speed references.
   - **Panel (c)**: Forward-wave $k$–$f$ spectrum for $u_1$.
   - **Panel (d)**: Forward-wave $k$–$f$ spectrum for $u_2$.
   - **Reference overlays**: Applicable fluid and isotropic-halfspace Rayleigh slopes; these are reference speeds, not analytical guided-mode dispersion solutions.
5. **`guided_wave.bp`** *(optional)*: ADIOS2 VTX displacement snapshots for ParaView when `write_vtx` is enabled.

Each MPI rank stores only its sensor columns. Rank zero gathers the global history, so long recordings still require adequate root memory. Plot ranges outside the sampled bandwidth remain outside the available spectral data rather than rescaling it.

---

## Code Quality & Developer Workflow

Numerical utilities are tested against independent analytical or algebraic references, alongside configuration, provenance, FEM execution, and serial/MPI regression checks.

```bash
# Run serial tests, including subprocess serial/MPI comparisons
conda run -n fenicsx-waves python -m pytest -q

# Run distributed off-node interpolation verification directly
conda run -n fenicsx-waves mpiexec -n 2 python -m pytest -q tests/test_numerics.py -k exact_point

# Check source and tests with the repository's Ruff configuration
conda run -n fenicsx-waves python -m ruff check src tests

# Inspect whitespace errors before submitting changes
git diff --check
```

| Regression Area | Verification |
| :--- | :--- |
| Damped time integration | RK4 convergence against a matrix-exponential oscillator solution |
| FE sensor evaluation | Exact quadratic polynomial values at off-node points and MPI ownership |
| Spectral analysis | Known traveling-wave frequency/wavenumber and Nyquist-independent plot geometry |
| Stability estimates | Assembled operator bound versus a direct generalized eigensolution |
| Input/material handling | Invalid parameter rejection and material energy conditions |
| Reproducibility | Boolean provenance, effective settings, and all shipped configuration schemas |
| FEM execution | Small explicit and implicit plate runs with plotting disabled |
| Distributed solver agreement | Serial/two-rank embedded fluid–solid histories for both integrators |

GitHub Actions runs the regression suite and distributed interpolation checks using the conda environment. These checks establish software behavior; application-specific physical accuracy and large-scale performance require their own benchmarks.

---

## Version Compatibility & Numerical Changes

Version 0.2 replaces damped Verlet with RK4, replaces nearest-node sensing with exact FE evaluation, and introduces assembled-operator time-step bounds. Automatic mesh sizing now defaults to `3*f_center`, both solver branches share the same default sponge coefficients, and spectral processing includes Hann windows and zero-padding.

These changes can alter results and increase runtime relative to older configurations. Existing inline comments naming Verlet refer to the earlier implementation. See [`CHANGELOG.md`](CHANGELOG.md) for the migration summary. Old output files remain readable when they contain the required measurement fields.

---

## Citation & License

This project is licensed under the [MIT License](LICENSE). If you use `GuidedWaveFEM` in academic research or industrial publications, please cite the repository and consult [`CITATION.cff`](CITATION.cff) for citation metadata.
