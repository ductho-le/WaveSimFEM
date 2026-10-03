# GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.10.0-orange.svg)](https://fenicsproject.org/)
[![MPI Parallel](https://img.shields.io/badge/MPI-MPI4Py%20%7C%20PETSc-green.svg)](https://petsc.org/)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A high-performance finite element simulation and signal processing framework for **2D plane-strain elastodynamic guided wave propagation** in plates, multilayered composites, truncated semi-infinite media, and regularized fluid–solid waveguides.

Powered by **DOLFINx / FEniCSx**, **PETSc**, and **MPI**, `GuidedWaveFEM` provides an end-to-end computational pipeline—from multi-format configuration parsing, conforming multi-domain mesh generation, and dual time-integration schemes to exact-point sensor measurements and automated 2D spatio-temporal Fourier transform $f$–$k$ dispersion analysis.

---

## Table of Contents

- [Code Architecture & Modular Structure](#code-architecture--modular-structure)
  - [Architecture Tree](#architecture-tree)
  - [Module Responsibilities](#module-responsibilities)
- [Theoretical & Numerical Formulations](#theoretical--numerical-formulations)
  - [1. Governing Elastodynamic Equations](#1-governing-elastodynamic-equations)
  - [2. Constitutive Material Models](#2-constitutive-material-models)
  - [3. Rayleigh Surface Wave Secular Solution](#3-rayleigh-surface-wave-secular-solution)
  - [4. Absorbing Boundary Conditions (5th-Order ALID Sponges)](#4-absorbing-boundary-conditions-5th-order-alid-sponges)
  - [5. Dual Time-Integration Solvers](#5-dual-time-integration-solvers)
  - [6. Signal Processing & 2D FFT Dispersion Analysis](#6-signal-processing--2d-fft-dispersion-analysis)
- [Installation & Environment Setup](#installation--environment-setup)
- [Usage Guide & Command-Line Interface](#usage-guide--command-line-interface)
  - [1. Running a Simulation](#1-running-a-simulation)
  - [2. Parallel MPI Execution](#2-parallel-mpi-execution)
  - [3. Preflight Resource Estimation (`--dry-run`)](#3-preflight-resource-estimation---dry-run)
  - [4. Runtime CLI Overrides](#4-runtime-cli-overrides)
  - [5. Standalone Signal Post-Processing](#5-standalone-signal-post-processing)
  - [6. Generating Configuration Templates](#6-generating-configuration-templates)
  - [7. Configuration Parameter Hierarchy](#7-configuration-parameter-hierarchy)
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
├── configs/                    # YAML, TOML, and JSON benchmark examples
├── tests/                      # Automated unit, regression, MPI, and CLI tests
├── .github/workflows/tests.yml # GitHub Actions continuous integration pipeline
├── environment.yml             # Pinned Conda numerical environment
├── pyproject.toml              # Build backend, package metadata, dependencies, and tools
├── CHANGELOG.md                # Detailed version release notes
├── CITATION.cff                # Software citation metadata
└── LICENSE                     # MIT open-source license
```

### Module Responsibilities

| Module | Core Responsibilities |
| :--- | :--- |
| [`guidedwavefem.config`](src/guidedwavefem/config.py) | Multi-format parsing (YAML/TOML/JSON), hierarchical-to-flat normalization, schema validation, CLI override application, and template generation. |
| [`guidedwavefem.materials`](src/guidedwavefem/materials.py) | Material normalization (isotropic, anisotropic Voigt, viscoelastic, regularized fluid), exact secular Rayleigh speed calculation, structure summary table, and preflight resource estimation. |
| [`guidedwavefem.numerics`](src/guidedwavefem/numerics.py) | Testable numerical routines: 4-stage classical Runge–Kutta (`rk4_step`), Gershgorin operator spectral bounds (`operator_bound`), exact point sensor evaluation (`PointSensors`), and forward-wave dispersion extraction (`dispersion_spectrum`). |
| [`guidedwavefem.solvers`](src/guidedwavefem/solvers.py) | Quadrilateral mesh creation, interface-conforming coordinate stretching, 5th-order ALID sponge formulation, MPI-safe assembly, explicit lumped RK4, and implicit Newmark-$\beta$ solvers. |
| [`guidedwavefem.analysis`](src/guidedwavefem/analysis.py) | Fourth-order Butterworth filtering, separable 2D Hann windowing, forward-wave 2D spatio-temporal FFT, B-scan plotting with half-pixel alignment, and vector image exports (`.png`, `.pdf`, `.svg`, `.eps`). |
| [`guidedwavefem.cli`](src/guidedwavefem/cli.py) | Command-line argument parsing, fail-fast MPI wrapper, execution dispatching, and run orchestration. |

---

## Theoretical & Numerical Formulations

### 1. Governing Elastodynamic Equations

The plane-strain elastodynamic problem is posed on a bounded physical domain $\Omega \subset \mathbb{R}^2$ with in-plane displacement field $\mathbf{u}(\mathbf{x}, t) = [u_1(\mathbf{x}, t), u_2(\mathbf{x}, t)]^T$, where $x_1$ represents the wave propagation direction along the plate length and $x_2$ is the transverse thickness coordinate.

#### Strong Form
The balance of linear momentum with damping dissipation satisfies:

$$
\rho(\mathbf{x}) \ddot{\mathbf{u}} + \alpha(\mathbf{x}) \rho(\mathbf{x}) \dot{\mathbf{u}} - \nabla \cdot \left[ \boldsymbol{\sigma}_e(\mathbf{u}) + \beta(\mathbf{x}) \boldsymbol{\sigma}_e(\dot{\mathbf{u}}) + \boldsymbol{\sigma}_v(\dot{\mathbf{u}}) \right] = \mathbf{0} \quad \text{in } \Omega \times (0, T)
$$

where $\rho$ is the mass density, $\boldsymbol{\sigma}_e$ is the elastic Cauchy stress tensor, $\boldsymbol{\sigma}_v$ is the internal viscous stress tensor (for viscoelastic media), $\alpha(\mathbf{x})$ is the mass-proportional sponge damping coefficient, and $\beta(\mathbf{x})$ is the stiffness-proportional sponge damping coefficient.

The small-strain tensor $\boldsymbol{\varepsilon}(\mathbf{u})$ is given by:

$$
\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)
$$

#### Weak Variational Formulation
Taking an arbitrary test function $\mathbf{w} \in [H^1(\Omega)]^2$, the weak momentum balance is:

$$
\int_\Omega \rho \, \ddot{\mathbf{u}} \cdot \mathbf{w} \, d\Omega + \int_\Omega \alpha(\mathbf{x}) \rho \, \dot{\mathbf{u}} \cdot \mathbf{w} \, d\Omega + \int_\Omega \left[ \boldsymbol{\sigma}_e(\mathbf{u}) + \beta(\mathbf{x}) \boldsymbol{\sigma}_e(\dot{\mathbf{u}}) + \boldsymbol{\sigma}_v(\dot{\mathbf{u}}) \right] : \boldsymbol{\varepsilon}(\mathbf{w}) \, d\Omega = \ell(\mathbf{w}, t)
$$

where $\ell(\mathbf{w}, t)$ denotes the external boundary or interface traction:

$$
\ell(\mathbf{w}, t) = \int_{\Gamma_{\mathrm{exc}}} \mathbf{t}_{\mathrm{ext}}(\mathbf{x}, t) \cdot \mathbf{w} \, ds
$$

Exterior boundaries are traction-free with $\mathbf{t} = \mathbf{0}$ except on the excitation zone $\Gamma_{\mathrm{exc}}$. Domain truncation is achieved via absorbing sponge layers rather than reflecting zero-displacement Dirichlet constraints.

#### Discretized Semidiscrete Matrix System
Spatial discretization by Lagrange quadrilateral elements $Q_p$ yields the classical second-order ordinary differential system:

$$
\mathbf{M} \ddot{\mathbf{u}} + \mathbf{C} \dot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{F}(t)
$$

where $\mathbf{M}$ is the global consistent mass matrix, $\mathbf{K}$ is the global stiffness matrix, $\mathbf{C} = \mathbf{C}_{\mathrm{sponge}} + \mathbf{C}_{\mathrm{visco}}$ is the dissipation matrix, and $\mathbf{F}(t)$ is the dynamic load vector.

---

### 2. Constitutive Material Models

`GuidedWaveFEM` supports four distinct material constitutive categories:

#### (a) Isotropic Elastic Solid
Linear isotropic Hookean elasticity is parameterized by Young's modulus $E$ and Poisson's ratio $\nu$:

$$
\boldsymbol{\sigma}_e = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}
$$

with Lamé parameters:

$$
\mu = \frac{E}{2(1 + \nu)}, \qquad \lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)}
$$

Bulk longitudinal $c_L$ and shear $c_T$ wave speeds are:

$$
c_L = \sqrt{\frac{\lambda + 2\mu}{\rho}}, \qquad c_T = \sqrt{\frac{\mu}{\rho}}
$$

Admissible parameters must satisfy $E > 0$, $\rho > 0$, and $-1 < \nu < 0.5$.

#### (b) Anisotropic Elastic Solid (General 2D Voigt)
For orthotropic, transverse isotropic, or monoclinic media, the stress-strain relation in Voigt engineering notation is:

$$
\begin{bmatrix} \sigma_{11} \\ \sigma_{22} \\ \sigma_{12} \end{bmatrix} = \begin{bmatrix} C_{11} & C_{12} & C_{16} \\ C_{12} & C_{22} & C_{26} \\ C_{16} & C_{26} & C_{66} \end{bmatrix} \begin{bmatrix} \varepsilon_{11} \\ \varepsilon_{22} \\ 2\varepsilon_{12} \end{bmatrix}
$$

Energy stability requires the $3 \times 3$ Voigt stiffness matrix $\mathbf{C}$ to be strictly positive definite:

$$
\mathbf{v}^T \mathbf{C} \mathbf{v} > 0 \quad \forall \mathbf{v} \neq \mathbf{0} \iff \lambda_i(\mathbf{C}) > 0
$$

Wave speed bounds for mesh sizing are derived from the eigenvalues of the orthonormal Kelvin-scaled tensor $\mathbf{S} \mathbf{C} \mathbf{S}$, where $\mathbf{S} = \mathrm{diag}(1, 1, \sqrt{2})$.

#### (c) Viscoelastic Solid (Kelvin–Voigt)
Viscoelastic materials incorporate rate-dependent dissipation proportional to the strain rate:

$$
\boldsymbol{\sigma}(\mathbf{u}, \dot{\mathbf{u}}) = \mathbf{C} : \boldsymbol{\varepsilon}(\mathbf{u}) + \mathbf{C}_v : \boldsymbol{\varepsilon}(\dot{\mathbf{u}})
$$

where $\mathbf{C}$ is the elastic stiffness tensor (Pa) and $\mathbf{C}_v$ is the viscosity tensor (Pa·s). Thermodynamic admissibility requires $\mathbf{C}_v$ to be positive semidefinite with eigenvalues $\lambda_i(\mathbf{C}_v) \ge 0$.

#### (d) Acoustic Fluid (Regularized Small-Shear Elasticity)
Acoustic fluids with bulk modulus $K_f = \rho_f c_f^2$ are modeled on conforming displacement meshes using an elastic regularization parameter $\epsilon_f \ll 1$ (`fluid_shear_reg`, default $10^{-8}$):

$$
\mu_f = \epsilon_f K_f, \qquad \lambda_f = K_f - \frac{2}{3}\mu_f
$$

The regularized compressional wave speed closely recovers the physical sound speed:

$$
c_L = \sqrt{\frac{\lambda_f + 2\mu_f}{\rho_f}} = \sqrt{\frac{K_f + \frac{4}{3}\mu_f}{\rho_f}} = c_f \sqrt{1 + \frac{4}{3}\epsilon_f} \approx c_f
$$

> [!NOTE]
> This monolithic formulation preserves full displacement continuity across fluid–solid boundaries without introducing Lagrange multipliers or mixed pressure-displacement formulations. It represents a small-shear elastic approximation with continuous tangential displacements.

---

### 3. Rayleigh Surface Wave Secular Solution

For a semi-infinite isotropic elastic halfspace with depth $x_2 \le 0$, the surface wave speed $c_R$ satisfies Rayleigh's cubic secular equation:

$$
\xi^3 - 8\xi^2 + 8(3 - 2\eta)\xi - 16(1 - \eta) = 0
$$

where:

$$
\xi = \left(\frac{c_R}{c_T}\right)^2, \qquad \eta = \left(\frac{c_T}{c_L}\right)^2 = \frac{1 - 2\nu}{2(1 - \nu)}
$$

`GuidedWaveFEM` extracts the exact physical root in $(0, 1)$ via companion-matrix eigenvalue factorization, with the robust Achenbach approximation as a fallback:

$$
c_R \approx c_T \left( \frac{0.862 + 1.14\nu}{1 + \nu} \right)
$$

---

### 4. Absorbing Boundary Conditions (5th-Order ALID Sponges)

To prevent spurious reflections from artificial computational domain boundaries, Absorbing Layers by Increasing Damping (ALID) are applied. The damping profile ramps smoothly from 0 at the interior interface to its peak at the boundary using a 5th-order polynomial:

$$
\alpha(s) = \alpha_{\mathrm{peak}} \, s^5, \qquad \beta(s) = \beta_{\mathrm{peak}} \, s^5, \qquad s \in [0, 1]
$$

where the normalized penetration depth into the sponge is:

$$
s(\mathbf{x}) = \frac{|\mathbf{x} - \mathbf{x}_{\mathrm{int}}|}{L_{\mathrm{abs}}}
$$

The resulting dissipation bilinear form assembled into $\mathbf{C}_{\mathrm{sponge}}$ is:

$$
C^{\mathrm{sponge}}_{ij} = \int_{\Omega_{\mathrm{sponge}}} \alpha(\mathbf{x}) \rho \, \boldsymbol{\phi}_j \cdot \boldsymbol{\phi}_i \, d\Omega + \int_{\Omega_{\mathrm{sponge}}} \beta(\mathbf{x}) \, \boldsymbol{\sigma}_e(\boldsymbol{\phi}_j) : \boldsymbol{\varepsilon}(\boldsymbol{\phi}_i) \, d\Omega
$$

Default peak parameters are $\alpha_{\mathrm{peak}} = 10^7 \, \text{s}^{-1}$ and $\beta_{\mathrm{peak}} = 10^{-8} \, \text{s}$. Horizontal and vertical sponge indicators are combined additively in halfspace corners.

---

### 5. Dual Time-Integration Solvers

`GuidedWaveFEM` offers two complementary time-stepping engines:

```
                               ┌─────────────────────────────────┐
                               │     Spatial Weak Formulation    │
                               │  M ü + C u̇ + K u = F_ext(t)     │
                               └────────────────┬────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │       EXPLICIT BRANCH       │                   │       IMPLICIT BRANCH       │
        │      Lumped-Mass RK4        │                   │      Newmark-β Solver       │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │ • Diagonal M_L = diag(M 1)  │                   │ • Consistent mass matrix M  │
        │ • 4-stage classical RK4     │                   │ • β = 0.25, γ = 0.5         │
        │ • Gershgorin operator bound │                   │ • Unconditional stability   │
        │ • O(N) memory scaling       │                   │ • Parallel MUMPS LU / GAMG  │
        └─────────────────────────────┘                   └─────────────────────────────┘
```

#### Branch A: Explicit 4th-Order Runge–Kutta (RK4) with Mass Lumping

##### (a) Row-Sum Diagonal Mass Lumping

$$
\mathbf{M}_L = \mathrm{diag}\left(\sum_j M_{ij}\right), \qquad (M_L)_{ii} > 0
$$

##### (b) First-Order State-Space Evolution
Defining velocity $\mathbf{v} = \dot{\mathbf{u}}$, the acceleration operator is:

$$
\mathbf{a}(\mathbf{u}, \mathbf{v}, t) = \mathbf{M}_L^{-1} \left[ \mathbf{F}_{\mathrm{ext}}(t) - \mathbf{K}\mathbf{u} - \mathbf{C}\mathbf{v} \right]
$$

##### (c) Consistent 4-Stage RK4 Update

$$
\begin{aligned}
\mathbf{k}_1^u &= \mathbf{v}_n, & \mathbf{k}_1^v &= \mathbf{a}(\mathbf{u}_n, \mathbf{v}_n, t_n) \\
\mathbf{k}_2^u &= \mathbf{v}_n + \frac{\Delta t}{2}\mathbf{k}_1^v, & \mathbf{k}_2^v &= \mathbf{a}\left(\mathbf{u}_n + \frac{\Delta t}{2}\mathbf{k}_1^u, \mathbf{k}_2^u, t_n + \frac{\Delta t}{2}\right) \\
\mathbf{k}_3^u &= \mathbf{v}_n + \frac{\Delta t}{2}\mathbf{k}_2^v, & \mathbf{k}_3^v &= \mathbf{a}\left(\mathbf{u}_n + \frac{\Delta t}{2}\mathbf{k}_2^u, \mathbf{k}_3^u, t_n + \frac{\Delta t}{2}\right) \\
\mathbf{k}_4^u &= \mathbf{v}_n + \Delta t \, \mathbf{k}_3^v, & \mathbf{k}_4^v &= \mathbf{a}\left(\mathbf{u}_n + \Delta t \, \mathbf{k}_3^u, \mathbf{k}_4^u, t_n + \Delta t\right)
\end{aligned}
$$

Displacement and velocity vectors are updated simultaneously:

$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \frac{\Delta t}{6} \left(\mathbf{k}_1^u + 2\mathbf{k}_2^u + 2\mathbf{k}_3^u + \mathbf{k}_4^u\right)
$$

$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \frac{\Delta t}{6} \left(\mathbf{k}_1^v + 2\mathbf{k}_2^v + 2\mathbf{k}_3^v + \mathbf{k}_4^v\right)
$$

##### (d) Conservative Gershgorin Spectral Stability Bound
Instead of heuristic wave speeds or random power iterations, absolute spectral bounds are computed from assembled operators:

$$
\omega_{\max}^2 \le B_K = \max_i \frac{\sum_j |K_{ij}|}{M_{L,ii}}, \qquad \zeta_{\max} \le B_C = \max_i \frac{\sum_j |C_{ij}|}{M_{L,ii}}
$$

$$
\Delta t_{\mathrm{crit}} = \frac{1}{\max\left(\sqrt{B_K}, B_C\right)}, \qquad \Delta t = \mathrm{CFL} \cdot \Delta t_{\mathrm{crit}} \quad (0 < \mathrm{CFL} \le 1)
$$

The integer step count is fitted exactly: $N_{\mathrm{steps}} = \lceil T_{\mathrm{total}} / \Delta t \rceil, \, \Delta t_{\mathrm{actual}} = T_{\mathrm{total}} / N_{\mathrm{steps}}$.

#### Branch B: Implicit Newmark-$\beta$ Solver

##### (a) Newmark Approximations with $\beta = 0.25, \, \gamma = 0.5$

$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t \, \mathbf{v}_n + \Delta t^2 \left[ \left(\frac{1}{2} - \beta\right)\mathbf{a}_n + \beta \mathbf{a}_{n+1} \right]
$$

$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \Delta t \left[ (1 - \gamma)\mathbf{a}_n + \gamma \mathbf{a}_{n+1} \right]
$$

##### (b) Effective Dynamic Linear System

$$
\mathbf{K}_{\mathrm{eff}} \, \mathbf{u}_{n+1} = \mathbf{F}_{\mathrm{eff}}(t_{n+1})
$$

where:

$$
\mathbf{K}_{\mathrm{eff}} = \frac{1}{\beta \Delta t^2} \mathbf{M} + \frac{\gamma}{\beta \Delta t} \mathbf{C} + \mathbf{K}
$$

$$
\mathbf{F}_{\mathrm{eff}} = \mathbf{F}_{\mathrm{ext}}(t_{n+1}) + \mathbf{M}\left[\frac{1}{\beta \Delta t^2}\mathbf{u}_n + \frac{1}{\beta \Delta t}\mathbf{v}_n + \left(\frac{1}{2\beta} - 1\right)\mathbf{a}_n\right] + \mathbf{C}\left[\frac{\gamma}{\beta \Delta t}\mathbf{u}_n + \left(\frac{\gamma}{\beta} - 1\right)\mathbf{v}_n + \Delta t\left(\frac{\gamma}{2\beta} - 1\right)\mathbf{a}_n\right]
$$

##### (c) PETSc Solvers
Parallel MUMPS direct LU solver or iterative Preconditioned Conjugate Gradients (CG) with Geometric-Algebraic Multigrid (GAMG) preconditioning.

---

### 6. Signal Processing & 2D FFT Dispersion Analysis

Sensor measurements are sampled along the observation line at exact coordinates $(x_i, y_s)$ via [`PointSensors`](src/guidedwavefem/numerics.py), avoiding nearest-node spatial distortion.

#### 2D Spatio-Temporal Fourier Transform
The wavefield $u(x, t)$ is transformed into wavenumber-frequency space:

$$
S(k, f) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x, t) \, w(x, t) \, e^{-i(2\pi f t - k x)} \, dt \, dx
$$

where $w(x, t) = w_{\mathrm{Hann}}(x) \otimes w_{\mathrm{Hann}}(t)$ is a separable 2D Hann taper that eliminates spectral leakage.

1. **Directional Separation:** By selecting $+f$ and $+k$, forward-propagating wave modes are isolated without backward-wave spectral folding.
2. **Phase Velocity Reference Slopes:** Nondispersive phase velocities $c_{\mathrm{phase}}$ appear as linear trajectories in $(k, f)$ space:

$$
f = \frac{c_{\mathrm{phase}}}{2000\pi} \, k \quad [f \text{ in MHz}, \, k \text{ in rad/mm}, \, c_{\mathrm{phase}} \text{ in m/s}]
$$

3. **Independent Component Processing:** In-plane longitudinal $u_1$ and transverse out-of-plane $u_2$ spectra are normalized and displayed separately to preserve modal polarity information.

---

## Installation & Environment Setup

The supported environment uses **Python 3.11**, **DOLFINx 0.10.0**, **PETSc 3.24**, and **MPICH** via Conda:

```bash
# 1. Clone the repository
git clone https://github.com/ductho-le/GuidedWaveFEM.git
cd GuidedWaveFEM

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate fenicsx-waves

# 3. Install GuidedWaveFEM in editable mode
python -m pip install --no-deps -e .
```

To archive an exact platform lockfile for reproducibility:
```bash
conda list --explicit > environment-explicit.txt
```

---

## Usage Guide & Command-Line Interface

All simulations require an explicit configuration file via `-c` / `--config`. Simulations can be launched using either the installed CLI `guided-waves` or module invocation `python -m guidedwavefem`.

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

### 3. Preflight Resource Estimation (`--dry-run`)

Preview mesh sizing, DOF counts, time step limits, and memory requirements upfront without allocating FEM structures:

```bash
guided-waves -c configs/01_single_layer_aluminum.yaml --dry-run
```

Output preview:
```text
╔══════════════════════════════════════════════════════════════════╗
║  PREFLIGHT NUMERICAL ESTIMATE (EXPLICIT)                         ║
║  ──────────────────────────────────────────────────────────      ║
║  Mesh frequency:        1.00 MHz (λ_min = 3.122 mm)              ║
║  Target element size h: 0.390 mm (actual min h = 0.167 mm)       ║
║  Grid resolution:       1538 × 6 Q2 elements (9,228 total)       ║
║  Estimated global DOFs: 80,002                                   ║
║  Simulation duration:   115.3 μs                                 ║
║  Estimated step limit:  ~6.72 ns (heuristic CFL=0.5)             ║
║  Estimated step count:  ~17,153 steps                            ║
║  Estimated memory RAM:  ~143.4 MB (single state vector: 0.6 MB)  ║
╚══════════════════════════════════════════════════════════════════╝
```

### 4. Runtime CLI Overrides

Override configuration settings directly from the terminal:

```bash
# Select implicit Newmark time integration
guided-waves -c configs/01_single_layer_aluminum.yaml --solver implicit

# Enable GPU acceleration (auto-detects NVIDIA CUDA via CuPy or Apple Silicon via PyTorch)
guided-waves -c configs/01_single_layer_aluminum.yaml --device auto

# Target specific CUDA GPU device or Apple Silicon MPS
guided-waves -c configs/01_single_layer_aluminum.yaml --device cuda:0
guided-waves -c configs/01_single_layer_aluminum.yaml --device mps

# Select single precision (float32) for 2x memory throughput (Komatitsch et al. 2010)
guided-waves -c configs/01_single_layer_aluminum.yaml --device auto --precision single

# Specify custom output directory
guided-waves -c configs/04_halfspace_rayleigh_aluminum.yaml --output results_rayleigh

# Throttle numerical controls (duration, mesh frequency, CFL, dt)
guided-waves -c configs/01_single_layer_aluminum.yaml --duration 5.0e-5 --max-frequency 1.0e6 --cfl 0.4

# Set dispersion display limits (MHz and rad/mm)
guided-waves -c configs/01_single_layer_aluminum.yaml --f-max 3.0 --k-max 4.0

# Disable postprocessing and plot generation
guided-waves -c configs/01_single_layer_aluminum.yaml --no-postprocess
```

> **GPU Acceleration Architecture:** The explicit RK4 engine implements zero-copy device-resident time integration, fused CUDA kernels, and asynchronous pinned host telemetry streaming based on Trotter et al. (2023), Komatitsch et al. (2010), Cao et al. (2026), and Li et al. (2026).

### 5. Standalone Signal Post-Processing

Reprocess existing measurement archives without rerunning the FEM solver:

```bash
guided-waves --postprocess results_case01_aluminum_1layer/sensor_data.npz --output reprocessed_figures
```

### 6. Generating Configuration Templates

Generate a fully annotated template YAML file:

```bash
guided-waves --generate-config my_custom_waveguide.yaml
```

### 7. Configuration Parameter Hierarchy

```yaml
solver:
  type: explicit           # "explicit" (lumped RK4) or "implicit" (Newmark-beta)
  CFL: 0.5                 # Courant-Friedrichs-Lewy stability factor
  time_factor: 1.2         # Duration multiplier: T = time_factor * L / c_min
  dt: null                 # Maximum requested time step [s] (null = auto from operator bound)
  duration: null           # Physical duration [s] (overrides time_factor)
  beta: 0.25               # Newmark-beta parameter (implicit only)
  gamma: 0.5               # Newmark-gamma parameter (implicit only)

geometry:
  sensing_length: 0.30     # [m] Pristine observation region length
  absorb_length: 0.15      # [m] Absorbing sponge layer thickness on lateral sides

mesh:
  elems_per_wavelength: 10 # Elements per shortest wavelength (default: 10)
  min_elems_x: 100         # Longitudinal element floor
  min_elems_y: 4           # Transverse element floor through thickness
  degree: 2                # Polynomial order (Q2 quadrilateral elements)
  max_frequency: null      # [Hz] Mesh sizing frequency (defaults to 3 * f_center)

output:
  output_dir: null         # Result directory (null = auto: results_guided_waves_{solver})
  write_vtx: false         # ADIOS2 VTX (.bp) Paraview export
  target_snapshots: 200    # Target snapshot count
  quick_plot: true         # Generate fast overview waterfall/B-scan plot
```

---

## Pre-Configured Examples & Benchmark Setups

Thirteen configuration files represent **eleven distinct physical setups** in [`configs/`](configs/):

| Tier | Config File | Mode | Physical System & Materials | Target Wave Phenomena |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | [`01_single_layer_aluminum.yaml`](configs/01_single_layer_aluminum.yaml) | `1_layer` | 1 mm aluminum plate in vacuum | Fundamental Lamb waves $S_0$ and $A_0$ |
| | [`01_single_layer_aluminum.toml`](configs/01_single_layer_aluminum.toml) | `1_layer` | Case 01, TOML format | Standard-library `tomllib` parsing |
| | [`01_single_layer_aluminum.json`](configs/01_single_layer_aluminum.json) | `1_layer` | Case 01, JSON format | JSON schema integration |
| | [`09_bonedc_case1_cortical_bone.yaml`](configs/09_bonedc_case1_cortical_bone.yaml) | `1_layer` | 4 mm cortical bone, BoneDC Case 1 | Independent Lamb dispersion benchmark |
| **Tier 2** | [`02_single_layer_cortical_bone_viscoelastic.yaml`](configs/02_single_layer_cortical_bone_viscoelastic.yaml) | `1_layer` | 3 mm cortical bone (Kelvin–Voigt) | Viscoelastic modal attenuation |
| | [`03_multilayer_bone_aluminum.yaml`](configs/03_multilayer_bone_aluminum.yaml) | `N_layers` | 2 mm bone + 1 mm aluminum | Heterogeneous laminate dispersion |
| | [`10_bonedc_case2_cb_st_fluid.yaml`](configs/10_bonedc_case2_cb_st_fluid.yaml) | `N_layers` | 4 mm bone + 1 mm regularized tissue | Coupled solid–fluid guided modes |
| | [`11_bonedc_case3_st_cb_mr_fluid.yaml`](configs/11_bonedc_case3_st_cb_mr_fluid.yaml) | `N_layers` | 1 mm marrow + 4 mm bone + 1 mm tissue | Tri-layer biological waveguide |
| **Tier 3** | [`04_halfspace_rayleigh_aluminum.yaml`](configs/04_halfspace_rayleigh_aluminum.yaml) | `halfspace` | 40 mm aluminum with bottom sponge | Rayleigh surface wave $c_R \approx 2906$ m/s |
| | [`07_plate_on_water_halfspace.yaml`](configs/07_plate_on_water_halfspace.yaml) | `plate_on_halfspace` | 1.5 mm aluminum on water substrate | Scholte waves & leaky Lamb radiation |
| | [`08_plate_immersed_in_water.yaml`](configs/08_plate_immersed_in_water.yaml) | `plate_in_halfspaces` | 1.5 mm aluminum in water domains | Bilateral fluid loading & radiation |
| | [`05_plate_on_halfspace_bone_marrow.yaml`](configs/05_plate_on_halfspace_bone_marrow.yaml) | `plate_on_halfspace` | 2 mm bone on 12 mm marrow substrate | Leaky Lamb waves & substrate radiation |
| | [`06_plate_in_halfspaces_bone_tissue_marrow.yaml`](configs/06_plate_in_halfspaces_bone_tissue_marrow.yaml) | `plate_in_halfspaces` | 2 mm bone between tissue & marrow | Stoneley/Scholte interface waves |

> [!TIP]
> Run `guided-waves -c <config> --dry-run` to inspect the computational footprint before launching Tier 2 or Tier 3 models.

---

## Output Artifacts & Publication Figures

Simulations save comprehensive structured measurements and vector figures:

1. **`sensor_data.npz`**: Uncompressed NumPy archive:
   - `times`: Discrete simulation time vector from $t = 0$ to $t = T$.
   - `sensor_x`, `sensor_y`, `sensor_coords`: Exact sensor coordinates along the array.
   - `sensor_data_x`: In-plane longitudinal displacement histories $u_1(x_i, t)$.
   - `sensor_data_y`: Plate-transverse displacement histories $u_2(x_i, t)$.
   - Provenance fields: Material models, wave speeds, layer summaries, and source locations.
2. **`simulation_config.yaml`**: Effective simulation settings and complete runtime metadata:
   - Damping values, actual time step, duration, and step count.
   - Mesh dimensions, DOFs, MPI rank count, and operator spectral bounds.
   - Environment versions (Python, NumPy, DOLFINx, PETSc) and module SHA-256 source hashes.
3. **`quick_results.png`**: Multi-panel visualization containing the excitation wavelet, displacement waterfall, and B-scan overview.
4. **`full_analysis.{png,pdf,svg,eps}`**: Publication-ready 2×2 composite figure (300 DPI):
   - **Panel (a)**: Selected $u_2$ waveforms at the first, middle, and last sensors.
   - **Panel (b)**: Spatio-temporal $u_2$ B-scan with theoretical wave arrival slopes $t_0 + x/c$.
   - **Panel (c)**: Forward-propagating 2D FFT dispersion spectrum for horizontal motion $u_1$.
   - **Panel (d)**: Forward-propagating 2D FFT dispersion spectrum for transverse motion $u_2$.
5. **`guided_wave.bp`** *(optional)*: ADIOS2 VTX displacement snapshots for 2D/3D visualization in ParaView.

---

## Code Quality & Developer Workflow

The repository includes an automated testing and verification suite evaluated against analytical references:

```bash
# Run complete test suite (21 unit, regression, MPI, and CLI tests)
conda run -n fenicsx-waves python -m pytest -v

# Run distributed MPI tests (2 ranks)
conda run -n fenicsx-waves mpiexec -n 2 python -m pytest -q tests/test_numerics.py -k exact_point

# Code style and linting
conda run -n fenicsx-waves python -m ruff check src tests

# Whitespace integrity check
git diff --check
```

### Automated Verification Matrix

| Area | Verification Target | Test Method |
| :--- | :--- | :--- |
| **Damped Time Integration** | 4th-order temporal convergence | Analytical matrix-exponential oscillator solution $\exp(\mathbf{A}t)$ |
| **FE Sensor Evaluation** | Quadratic off-node exact evaluation | Distributed bounding-box tree and collision cells |
| **Spectral Analysis** | Exact frequency and wavenumber peaks | Monochromatic traveling wave with exact $f_0$ and $k_0$ |
| **Stability Limits** | Assembled operator bounds $B_K$ and $B_C$ | SciPy direct generalized eigensolution $\mathbf{K}\mathbf{v} = \lambda \mathbf{M}_L \mathbf{v}$ |
| **Material Energy** | Positive definiteness enforcement | Voigt tensor and viscosity eigenvalue validation |
| **Distributed Solver Agreement** | Serial vs. 2-rank MPI consistency | Coupled fluid–solid waveguide regression with relative error $< 10^{-8}$ |
| **CLI & Orchestration** | Template generation, dry-run, and postprocessing | Automated CLI test suite (`tests/test_cli.py`) |
| **Configuration Schemas** | All 13 shipped configurations | Automated schema parsing and normalization check |

---

## Version Compatibility & Numerical Changes

### Version 0.2 Highlights
- **Integrator:** Replaced first-order damped Verlet with classical 4th-order explicit RK4 with lumped mass.
- **Stability:** Replaced fixed-iteration power method with conservative Gershgorin operator row-sum bounds.
- **Sensors:** Replaced nearest-node sensing with exact finite element point evaluation (`PointSensors`).
- **Dispersion:** Added 2D separable Hann tapering, non-folded forward-wave spectral filtering, and half-pixel B-scan alignment.
- **Modularity:** Restructured codebase into a pure `src/guidedwavefem` Python package.

See [`CHANGELOG.md`](CHANGELOG.md) for detailed version history.

---

## Citation & License

This project is licensed under the [MIT License](LICENSE).

If you use `GuidedWaveFEM` in academic research or industrial publications, please cite the package using the metadata in [`CITATION.cff`](CITATION.cff) or the following BibTeX entry:

```bibtex
@software{Le_GuidedWaveFEM_2026,
  author       = {Le, Duc-Tho},
  title        = {{GuidedWaveFEM: Unified Guided \& Surface Wave Finite Element Framework}},
  year         = {2026},
  version      = {0.2.0},
  publisher    = {GitHub},
  url          = {https://github.com/ductho-le/GuidedWaveFEM}
}
```
