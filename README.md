# GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.10.0-orange.svg)](https://fenicsproject.org/)
[![MPI Parallel](https://img.shields.io/badge/MPI-MPI4Py%20%7C%20PETSc-green.svg)](https://petsc.org/)
[![CI](https://github.com/ductho-le/GuidedWaveFEM/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ductho-le/GuidedWaveFEM/actions/workflows/tests.yml)
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
  - [7. Excitation Waveforms & Physical Scales](#7-excitation-waveforms--physical-scales)
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
│       ├── __init__.py         # Public API (lazy exports; importing does not load DOLFINx)
│       ├── __main__.py         # Direct execution entry point (`python -m guidedwavefem`)
│       ├── config.py           # Multi-format loaders, strict schema, defaults, CLI overrides, template
│       ├── materials.py        # Material/structure normalization, Rayleigh speed, mesh layout, preflight
│       ├── numerics.py         # RK4 reference step, operator bounds, exact PointSensors, 2D FFT
│       ├── sources.py          # Excitation waveforms (Ricker wavelet, Hann tone burst)
│       ├── units.py            # Scale-aware display units and unit-string parsing
│       ├── solvers.py          # Mesh, sponges, assembly, Newmark-β and explicit RK4 drivers, outputs
│       ├── solvers_gpu.py      # Single-process array RK4 engine (SciPy / CuPy / PyTorch backends)
│       ├── device.py           # Accelerator discovery and device-request validation
│       ├── analysis.py         # Band-pass filtering, dispersion figures, quick-look plot
│       ├── reporting.py        # `guidedwavefem` logger, verbosity, banners and structure tables
│       └── cli.py              # `guided-waves` argument parsing and orchestration
├── configs/                    # One documented example per structure mode
├── tests/                      # Unit, regression, physics, MPI, and CLI tests
├── tools/clean_junk.py         # Safe cache/build/OS-junk cleaner (used by pre-commit)
├── .github/workflows/tests.yml # CI: lint, tests (+MPI, coverage), distribution build
├── .pre-commit-config.yaml     # Junk cleanup/guard, hygiene, and lint/format hooks
├── environment.yml             # Conda environment (DOLFINx/PETSc/MPI stack)
├── pyproject.toml              # Build backend, package metadata, dependencies, and tools
├── CHANGELOG.md                # Version release notes
├── CITATION.cff                # Software citation metadata
└── LICENSE                     # MIT open-source license
```

### Module Responsibilities

| Module | Core Responsibilities |
| :--- | :--- |
| [`guidedwavefem.config`](src/guidedwavefem/config.py) | Multi-format parsing (YAML/TOML/JSON), hierarchical-to-flat normalization, strict schema (unknown keys rejected with suggestions), typed validation, single table of defaults, CLI overrides, and template generation. |
| [`guidedwavefem.materials`](src/guidedwavefem/materials.py) | Material normalization (isotropic, anisotropic Voigt, viscoelastic, regularized fluid) with per-model field checks, exact secular Rayleigh speed, mesh layout shared by solver and preflight, and preflight resource estimation. |
| [`guidedwavefem.numerics`](src/guidedwavefem/numerics.py) | Classical Runge–Kutta reference step (`rk4_step`), vectorized Gershgorin operator bounds (`operator_bound`), exact point sensors with `Gatherv` collection (`PointSensors`), and forward-wave dispersion extraction (`dispersion_spectrum`). |
| [`guidedwavefem.sources`](src/guidedwavefem/sources.py) | Excitation time functions: broadband Ricker wavelet and Hann-windowed $N$-cycle tone burst, with pulse delay/duration helpers. |
| [`guidedwavefem.units`](src/guidedwavefem/units.py) | Scale-aware display units (s/ms/μs, m/mm, Hz/kHz/MHz, rad/m or rad/mm) for logs and figures, and parsing of quantities such as `"2 MHz"` or `"3 rad/mm"`. |
| [`guidedwavefem.solvers`](src/guidedwavefem/solvers.py) | Quadrilateral mesh creation, interface-conforming layering, 5th-order ALID sponges, assembly, implicit Newmark-$\beta$ and explicit lumped-mass RK4 drivers, provenance, and output files. |
| [`guidedwavefem.solvers_gpu`](src/guidedwavefem/solvers_gpu.py) | Backend-agnostic, allocation-free RK4 engine for single-process runs on CPU (SciPy) or CUDA (CuPy/PyTorch); exact sparse sensor operator; device-resident sensor buffering; non-finite detection. |
| [`guidedwavefem.device`](src/guidedwavefem/device.py) | Discovery of usable accelerators and strict validation of device requests (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). |
| [`guidedwavefem.analysis`](src/guidedwavefem/analysis.py) | Zero-phase Butterworth filtering, separable 2D Hann windowing, forward-wave 2D FFT, B-scan with half-pixel alignment, and figure export (`.png`, `.pdf`, `.svg`, `.eps`) via the object-oriented Matplotlib API with scoped style settings. |
| [`guidedwavefem.reporting`](src/guidedwavefem/reporting.py) | Package logger (`guidedwavefem`), verbosity control, banner and structure tables. |
| [`guidedwavefem.cli`](src/guidedwavefem/cli.py) | Command-line parsing, fail-fast MPI wrapper, and run orchestration. |

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

> [!WARNING]
> **Limitations of the regularized fluid.** A real inviscid fluid permits tangential slip at a solid interface; here the tangential displacement is continuous, and the fluid carries a (tiny) shear stiffness. Spurious low-speed shear modes with $c_T = c_f\sqrt{\epsilon_f}$ exist in the fluid, and fluid-loaded interface waves (Scholte/Stoneley) and leaky-wave attenuation are reproduced only approximately. Treat fluid-loaded results as qualitative unless validated against an acoustic–elastic reference solution for your case.

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

By default the peak parameters scale with the centre frequency,

$$
\alpha_{\mathrm{peak}} = 10^7 \, \text{s}^{-1} \, \frac{f_0}{1\,\text{MHz}}, \qquad \beta_{\mathrm{peak}} = 10^{-9} \, \text{s} \, \frac{1\,\text{MHz}}{f_0},
$$

so the damping ratio $\zeta(f_0) = \alpha/(4\pi f_0) + \pi f_0 \beta \approx 0.80 + 0.003$ is the same from seismic (Hz) to ultrasonic (MHz) problems. The mass-proportional term does the absorbing; its spatial attenuation $\approx \alpha / (2c)$ is nearly frequency independent. The stiffness-proportional term is kept small because $\beta\,\omega^2$ grows with the highest mesh frequency and would otherwise limit the explicit time step (the solver warns when damping, rather than wave speed, sets $\Delta t$; this can still happen in high-contrast media, see example 04). Explicit `absorbing_boundaries.alpha/beta` values override the defaults. Horizontal and vertical sponge indicators are combined additively in halfspace corners.

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

Because Gershgorin row sums can only *over*-estimate the spectral radius, the resulting step is guaranteed stable but conservative — typically somewhat smaller than the true RK4 limit. The bound includes the assembled sponge and viscous damping, so strongly damped sponges can reduce $\Delta t$.

##### (e) Execution Engines
- **Single process** (`mpiexec -n 1` or no MPI): the array engine in [`solvers_gpu.py`](src/guidedwavefem/solvers_gpu.py) advances the stacked state $[\mathbf{u};\mathbf{v}]$ with one sparse product $[\mathbf{K}\,|\,\mathbf{C}]\,[\mathbf{u};\mathbf{v}]$ per stage, preallocated buffers, and a bounded on-device sensor buffer flushed to the host in chunks. Backends: SciPy (CPU), CuPy or PyTorch (CUDA).
- **Multiple MPI ranks**: the distributed PETSc engine applies the identical RK4 scheme with parallel PETSc operators.

Both engines are verified to agree to a relative error below $10^{-8}$ (see `tests/test_mpi.py`).

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
f = \frac{c_{\mathrm{phase}}}{2\pi} \, k \quad [f \text{ in Hz}, \, k \text{ in rad/m}, \, c_{\mathrm{phase}} \text{ in m/s}]
$$

3. **Independent Component Processing:** In-plane longitudinal $u_1$ and transverse out-of-plane $u_2$ spectra are normalized and displayed separately to preserve modal polarity information.
4. **Band-pass:** Records are filtered with a zero-phase Butterworth band $[f_0/5, \, 3 f_0]$ that scales with the excitation frequency.

---

### 7. Excitation Waveforms & Physical Scales

The surface (or interface) traction is $\mathbf{t}(x, t) = T_{\max} \, g(x) \, s(t) \, \hat{\mathbf{d}}$ with a Gaussian strip $g(x)$ of width `excitation_width`, direction $\hat{\mathbf{d}} \propto$ (`source_fx`, `source_fy`), and a time function selected by `excitation.source_type`:

| `source_type` | $s(t)$ | Use |
| :--- | :--- | :--- |
| `ricker` (default) | $(1 - 2a)\,e^{-a}$, $a = [\pi f_0 (t - 1.5/f_0)]^2$ | Broadband pulse; best for extracting dispersion curves with the 2D FFT |
| `tone_burst` | $\tfrac12\left[1 - \cos(2\pi f_0 t / N)\right] \sin(2\pi f_0 t)$ on $[0, N/f_0]$ | Narrowband, mode-selective NDE/SHM excitation; `n_cycles` $= N$ (default 5, $-6$ dB bandwidth $\approx 2f_0/N$) |

Unless `solver.duration` is given, the simulated time is `time_factor` $\times$ `sensing_length` / $c_{\mathrm{slow}}$ plus the pulse length ($3/f_0$ for the Ricker wavelet, $N/f_0$ for the tone burst).

All inputs are SI, and nothing in the solver assumes ultrasonic scales: logs, the preflight box and figures choose units from the data (μs/mm/MHz for an aluminium plate, ms/m/Hz for a soil profile). Dispersion plot limits accept SI numbers or unit strings (`"2 MHz"`, `"40 Hz"`, `"3 rad/mm"`, `"1.2 rad/m"`), and the default source position keeps the load at least one `excitation_width` away from the sponge.

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

# 3. Install GuidedWaveFEM in editable mode (dependencies come from conda)
python -m pip install --no-deps -e .

# 4. (Optional) developer tooling
python -m pip install -e ".[dev]" && pre-commit install
```

> [!IMPORTANT]
> DOLFINx, PETSc, petsc4py and mpi4py are runtime requirements but are not declared as pip dependencies because they are not reliably pip-installable; install them from conda-forge via `environment.yml`.

#### Optional GPU Acceleration (explicit solver, single process)

```bash
# NVIDIA CUDA via CuPy (recommended)
python -m pip install -e ".[cuda]"      # installs cupy-cuda12x
# or NVIDIA CUDA via PyTorch
python -m pip install -e ".[torch]"
```

Apple-Silicon MPS is detected but currently unusable: PyTorch does not implement sparse CSR mat-vec on Metal, so `--device mps` reports the device as unavailable and `auto` falls back to the CPU engine.

To archive an exact platform lockfile for reproducibility:
```bash
conda list --explicit > environment-explicit.txt
```

---

## Usage Guide & Command-Line Interface

All simulations require an explicit configuration file via `-c` / `--config`. Simulations can be launched using either the installed CLI `guided-waves` or module invocation `python -m guidedwavefem`.

### 1. Running a Simulation

```bash
guided-waves -c configs/01_plate_aluminum_lamb.yaml
# Or alternatively:
python -m guidedwavefem -c configs/01_plate_aluminum_lamb.yaml
```

### 2. Parallel MPI Execution

Distribute simulation work across processes (uses the distributed PETSc engines):

```bash
mpiexec -n 4 guided-waves -c configs/05_plate_in_halfspaces_bone_qus.yaml
# Or:
mpiexec -n 4 python -m guidedwavefem -c configs/05_plate_in_halfspaces_bone_qus.yaml
```

### 3. Preflight Resource Estimation (`--dry-run`)

Preview mesh sizing, DOF counts, time step limits, and memory requirements upfront without allocating FEM structures:

```bash
guided-waves -c configs/01_plate_aluminum_lamb.yaml --dry-run
```

Output preview:
```text
╔══════════════════════════════════════════════════════════════════╗
║  PREFLIGHT NUMERICAL ESTIMATE (EXPLICIT)                         ║
║  ──────────────────────────────────────────────────────────────  ║
║  Excitation:            Ricker wavelet, f₀ = 1 MHz               ║
║  Mesh frequency:        1 MHz (λ_min = 3.122 mm)                 ║
║  Target element size h: 390.2 μm (actual min h = 166.7 μm)       ║
║  Grid resolution:       1538 × 6 Q2 elements (9,228 total)       ║
║  Estimated global DOFs: 80,002                                   ║
║  Simulation duration:   118.3 μs                                 ║
║  Time step:             ~6.723 ns (heuristic, CFL=0.5)           ║
║  Estimated step count:  ~17,599 steps                            ║
║  Estimated memory RAM:  ~143.4 MB (single state vector: 0.6 MB)  ║
║  Explicit Δt is finalized from the assembled operator bound.     ║
╚══════════════════════════════════════════════════════════════════╝
```

Units follow the problem scale (the seismic example reports metres, milliseconds and hertz). The preflight uses the same mesh-layout code as the solver, so mesh sizes and DOF counts match the actual run. The implicit step is exact; the explicit step shown is a pre-assembly heuristic (the solver derives the final value from the assembled operators).

> [!NOTE]
> `elems_per_wavelength` counts **elements** per shortest physical wavelength, independent of the polynomial degree; a Q2 element already carries two node spacings, so Q2 meshes are effectively twice as fine in nodes per wavelength as Q1 meshes with the same setting.

### 4. Runtime CLI Overrides

Override configuration settings directly from the terminal:

```bash
# Select implicit Newmark time integration
guided-waves -c configs/01_plate_aluminum_lamb.yaml --solver implicit

# Use an accelerator if one is available, otherwise the CPU engine
guided-waves -c configs/01_plate_aluminum_lamb.yaml --device auto

# Require a specific CUDA GPU (fails with exit code 3 if unavailable)
guided-waves -c configs/01_plate_aluminum_lamb.yaml --device cuda:0

# Single precision (float32) halves memory traffic on the array engine
guided-waves -c configs/01_plate_aluminum_lamb.yaml --device auto --precision single

# Specify custom output directory
guided-waves -c configs/03_halfspace_steel_rayleigh.yaml --output results_rayleigh

# Throttle numerical controls (duration, mesh frequency, CFL, dt)
guided-waves -c configs/01_plate_aluminum_lamb.yaml --duration 5.0e-5 --max-frequency 1.0e6 --cfl 0.4

# Set dispersion display limits: SI numbers (Hz, rad/m) or strings with units
guided-waves -c configs/01_plate_aluminum_lamb.yaml --f-max "3 MHz" --k-max "4 rad/mm"

# Disable postprocessing and plot generation
guided-waves -c configs/01_plate_aluminum_lamb.yaml --no-postprocess

# Quiet (warnings only) or verbose (debug) console output
guided-waves -c configs/01_plate_aluminum_lamb.yaml -q
```

#### Device Selection Semantics

| Request | Single process | `mpiexec -n >1` | Implicit solver |
| :--- | :--- | :--- | :--- |
| `auto` (default) | First usable CUDA device, else CPU array engine | Distributed PETSc engine | PETSc (CPU) |
| `cpu` | CPU array engine (SciPy) | Distributed PETSc engine | PETSc (CPU) |
| `cuda`, `cuda:N` | That GPU, or **error** if unavailable | **Error** (GPU engine is single-process) | Ignored with a warning |
| `mps` | **Error** (no sparse SpMV on Metal) | **Error** | Ignored with a warning |

`precision: single` applies to the single-process array engine only; the PETSc engines always use float64 and emit a warning if single precision was requested. The resolved device, backend, and precision are recorded in `simulation_config.yaml` under `runtime`.

Console output goes through the standard `logging` logger named `guidedwavefem`; library users can silence it with `guidedwavefem.set_verbosity("WARNING")` or attach their own handlers.

#### Exit Status

| Code | Meaning |
| :--- | :--- |
| `0` | Success |
| `1` | No configuration file given for a simulation run |
| `2` | Usage or input error: invalid/unreadable configuration, `--dry-run` without `-c`, missing sensor archive |
| `3` | Explicitly requested compute device is unavailable |

Input errors print a single `guided-waves: error: ...` line; add `-v` to see the full traceback.

### 5. Standalone Signal Post-Processing

Reprocess existing measurement archives without rerunning the FEM solver:

```bash
guided-waves --postprocess results_01_plate_aluminum_lamb/sensor_data.npz --output reprocessed_figures

# Optionally apply output settings (figure_formats, usetex, plot limits) from a config file
guided-waves -c my_plot_settings.yaml --postprocess results_01_plate_aluminum_lamb/sensor_data.npz
```

Without `-c`, the plot limits stored in the archive are used unless overridden with `--f-min/--f-max/--k-min/--k-max` (SI numbers or unit strings such as `--f-max 40Hz`). Archives written by earlier versions (limits in MHz and rad/mm) are still understood.

### 6. Generating Configuration Templates

Generate a fully annotated template YAML file:

```bash
guided-waves --generate-config my_custom_waveguide.yaml
```

### 7. Configuration Parameter Hierarchy

```yaml
solver:
  type: explicit           # "explicit" (lumped RK4) or "implicit" (Newmark-beta)
  device: auto             # auto | cpu | cuda | cuda:N | mps (explicit solver only)
  precision: double        # double | single (single-process explicit engine only)
  CFL: 0.5                 # Courant-Friedrichs-Lewy stability factor
  time_factor: 1.2         # Duration multiplier: T = time_factor * L / c_R (or slowest wave speed)
  dt: null                 # Maximum requested time step [s] (null = auto)
  duration: null           # Physical duration [s] (overrides time_factor)
  beta: 0.25               # Newmark-beta parameter (implicit only)
  gamma: 0.5               # Newmark-gamma parameter (implicit only)

geometry:
  sensing_length: 0.30     # [m] Pristine observation region length
  absorb_length: 0.15      # [m] Absorbing sponge layer thickness on lateral sides

absorbing_boundaries:
  alpha: null              # [1/s] Peak mass damping (null = 1e7 * f_center / 1 MHz)
  beta: null               # [s] Peak stiffness damping (null = 1e-9 * 1 MHz / f_center)

excitation:
  source_type: ricker      # ricker (broadband) | tone_burst (Hann-windowed, narrowband)
  f_center: 1.0e6          # [Hz] Centre frequency
  n_cycles: null           # tone_burst only (null = 5)
  traction_max: 1.0e6      # [Pa] Peak traction
  excitation_width: 2.0e-3 # [m] Gaussian strip width
  source_x: null           # [m] null = absorb_length + max(5 mm, excitation_width)
  source_fx: 1.0           # Tangential traction component
  source_fy: 1.0           # Normal traction component

mesh:
  elems_per_wavelength: 10 # Elements (not nodes) per shortest wavelength
  min_elems_x: 100         # Longitudinal element floor
  min_elems_y: 4           # Transverse element floor through thickness
  degree: 2                # Polynomial order (Q2 quadrilateral elements)
  max_frequency: null      # [Hz] Mesh sizing frequency (defaults to 3 * f_center)

postprocess:
  run_postprocess: true    # 2D FFT dispersion analysis and figures
  f_min: null              # Plot limits: SI numbers or unit strings ("2 MHz", "3 rad/mm")
  f_max: null              # null = 2 * f_center
  k_min: null              # null = 0
  k_max: null              # null = 2*pi*f_max / slowest speed (capped at pi / pitch)

output:
  output_dir: null         # Result directory (null = auto: results_guided_waves_{solver})
  write_vtx: false         # ADIOS2 VTX (.bp) Paraview export
  target_snapshots: 200    # Target snapshot count
  quick_plot: true         # Generate fast overview waterfall/B-scan plot
  compress_output: false   # Write sensor_data.npz with zlib compression
  figure_formats: [png, pdf, svg, eps]  # Any subset
  usetex: auto             # auto (LaTeX if installed) | true | false
```

The schema is **strict**: unknown keys, misspelled section fields, unknown material fields, constitutive fields the chosen `material_model` would ignore (e.g. `Cv` on an `isotropic` layer), parameters defined twice, and non-boolean values for boolean switches are all rejected with a descriptive error (including a "did you mean" suggestion where possible). Run `guided-waves --generate-config template.yaml` for the full, commented schema.

---

## Pre-Configured Examples & Benchmark Setups

[`configs/`](configs/) holds **one example per supported structure mode**, each built from a real application with documented materials and references (see [`configs/README.md`](configs/README.md)):

| Config file | Mode | Physical system | Application | Excitation |
| :--- | :--- | :--- | :--- | :--- |
| [`01_plate_aluminum_lamb.yaml`](configs/01_plate_aluminum_lamb.yaml) | `1_layer` | 1 mm aluminium plate in vacuum | Lamb-wave NDE / SHM ($S_0$, $A_0$, $A_1$) | Ricker, 1 MHz |
| [`02_layers_bonded_joint.yaml`](configs/02_layers_bonded_joint.yaml) | `N_layers` | Al / 0.2 mm viscoelastic epoxy / Al | Adhesive bond inspection (aerospace) | 5-cycle tone burst, 300 kHz; implicit solver |
| [`03_halfspace_steel_rayleigh.yaml`](configs/03_halfspace_steel_rayleigh.yaml) | `halfspace` | Rail steel, truncated halfspace | Rayleigh-wave surface-crack NDE ($c_R \approx 2.98$ km/s) | Ricker, 1 MHz |
| [`04_plate_on_halfspace_soil_bedrock.yaml`](configs/04_plate_on_halfspace_soil_bedrock.yaml) | `plate_on_halfspace` | 10 m soft soil on bedrock | Seismic MASW / site characterisation | Ricker, 15 Hz |
| [`05_plate_in_halfspaces_bone_qus.yaml`](configs/05_plate_in_halfspaces_bone_qus.yaml) | `plate_in_halfspaces` | Anisotropic viscoelastic cortical bone between soft tissue and marrow | Axial-transmission bone QUS | Ricker, 500 kHz |

Together the examples exercise every material model (isotropic, anisotropic, viscoelastic, fluid), both excitation types, both time integrators, and the full frequency range from 15 Hz to 1 MHz. Example 05 uses the regularized-elastic fluid model; see the limitations in [Constitutive Material Models](#d-acoustic-fluid-regularized-small-shear-elasticity).

> [!TIP]
> Run `guided-waves -c <config> --dry-run` to inspect the computational footprint first; example 05 (≈ 0.66 M DOFs) benefits from `mpiexec -n 4`.

---

## Output Artifacts & Publication Figures

Simulations save comprehensive structured measurements and vector figures:

1. **`sensor_data.npz`**: NumPy archive (zlib-compressed when `compress_output: true`):
   - `times`: Discrete simulation time vector from $t = 0$ to $t = T$.
   - `sensor_x`, `sensor_y`, `sensor_coords`: Exact sensor coordinates along the array.
   - `sensor_data_x`: In-plane longitudinal displacement histories $u_1(x_i, t)$.
   - `sensor_data_y`: Plate-transverse displacement histories $u_2(x_i, t)$.
   - Provenance fields: Material models, wave speeds, layer summaries, and source locations.
2. **`simulation_config.yaml`**: Effective simulation settings and complete runtime metadata (reloadable as a configuration):
   - Damping values, actual time step, duration, and step count.
   - Mesh dimensions, DOFs, MPI rank count, operator spectral bounds, and the engine/device/precision used.
   - Package and environment versions (GuidedWaveFEM, Python, NumPy, DOLFINx, PETSc) and module SHA-256 source hashes.
3. **`quick_results.png`**: Multi-panel visualization containing the excitation wavelet, displacement waterfall, and B-scan overview.
4. **`full_analysis.{png,pdf,svg,eps}`** (formats per `figure_formats`): 2×2 composite figure (300 DPI):
   - **Panel (a)**: Selected $u_2$ waveforms at the first, middle, and last sensors.
   - **Panel (b)**: Spatio-temporal $u_2$ B-scan with theoretical wave arrival slopes $t_0 + x/c$.
   - **Panel (c)**: Forward-propagating 2D FFT dispersion spectrum for horizontal motion $u_1$.
   - **Panel (d)**: Forward-propagating 2D FFT dispersion spectrum for transverse motion $u_2$.
5. **`guided_wave.bp`** *(optional)*: ADIOS2 VTX displacement snapshots for 2D/3D visualization in ParaView.

---

## Code Quality & Developer Workflow

The repository includes an automated testing and verification suite evaluated against analytical references:

```bash
# Complete serial test suite (unit, regression, physics, MPI-subprocess, and CLI tests)
conda run -n fenicsx-waves python -m pytest

# Skip the longer end-to-end physics verification runs
conda run -n fenicsx-waves python -m pytest -m "not slow"

# Coverage report
conda run -n fenicsx-waves python -m pytest --cov

# Distributed tests (2 ranks)
conda run -n fenicsx-waves mpiexec -n 2 python -m pytest -q -p no:cacheprovider tests/test_numerics.py -k "exact_point or gather"

# Lint and format (also run by pre-commit and CI)
conda run -n fenicsx-waves ruff check src tests tools
conda run -n fenicsx-waves ruff format --check src tests tools

# All pre-commit hooks on the whole repository
conda run -n fenicsx-waves pre-commit run --all-files

# Remove junk files and folders (preview first with --dry-run)
conda run -n fenicsx-waves python tools/clean_junk.py --dry-run
conda run -n fenicsx-waves python tools/clean_junk.py
```

#### Pre-commit Hooks

Hooks run in the order listed:

| Hook | Purpose |
| :--- | :--- |
| `forbid-junk` | Blocks commits that stage junk files (e.g. via `git add -f`); runs first to fail fast |
| `pre-commit-hooks` | YAML/TOML/JSON syntax, end-of-file and trailing whitespace, large files, merge and case conflicts |
| `ruff-check`, `ruff-format` | Lint (with autofix) and format, pinned to the CI version |
| `clean-junk` | Runs last, so it also removes caches created by earlier hooks. Deletes regenerable junk: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.ipynb_checkpoints/`, `htmlcov/`, coverage files, `*.egg-info/`, root `build/`/`dist/`, `.DS_Store`, `._*`, `Thumbs.db`, editor swap/backup files |

> [!NOTE]
> `clean-junk` never touches simulation results (`results_*/`, `*.npz`, `*.bp`), virtual environments, the `.dev/` scratch area, or any Git-tracked file.

### Automated Verification Matrix

| Area | Verification Target | Test Method |
| :--- | :--- | :--- |
| **Physics** | Low-frequency $S_0$ speed equals the plate velocity $\sqrt{E/(\rho(1-\nu^2))}$ | End-to-end explicit and implicit runs, cross-correlation arrival time (agreement ≈ 0.1 %) |
| **Damped Time Integration** | 4th-order temporal convergence | Analytical matrix-exponential oscillator solution $\exp(\mathbf{A}t)$ |
| **Array RK4 Engine** | Bit-level agreement with the reference RK4 step | Random damped system, incl. tiny sensor ring buffers, instability detection, optional Torch/CuPy backends |
| **FE Sensor Evaluation** | Exact off-node evaluation | Distributed bounding-box search; sparse sensor operator vs. DOLFINx `eval` (shared/adjacent cells) |
| **Spectral Analysis** | Exact frequency and wavenumber peaks | Monochromatic traveling wave with exact $f_0$ and $k_0$ |
| **Stability Limits** | Assembled operator bounds $B_K$ and $B_C$ | SciPy generalized eigensolution $\mathbf{K}\mathbf{v} = \lambda \mathbf{M}_L \mathbf{v}$ |
| **Material Energy** | Positive definiteness enforcement | Voigt tensor and viscosity eigenvalue validation |
| **Distributed Solver Agreement** | Single-process array engine vs. 2-rank PETSc engine | Coupled fluid–solid waveguide regression with relative error $< 10^{-8}$ |
| **Configuration Schema** | Strict, typed validation; one valid shipped example per structure mode | Dedicated schema tests (`tests/test_config.py`), example coverage test |
| **Excitation & Units** | Tone-burst window, spectrum and timing; unit picking and parsing | `tests/test_units_sources.py` |
| **Figures** | Portable labels, no global-state leakage, selectable formats, seismic-scale units | `tests/test_analysis.py` |
| **CLI & Orchestration** | Template, dry-run, postprocessing, verbosity, error paths | `tests/test_cli.py` |

---

## Version Compatibility & Numerical Changes

### Version 0.3 Highlights
- **Explicit engine:** One backend-agnostic, allocation-free RK4 engine (SciPy/CuPy/PyTorch) with a stacked $[\mathbf{K}|\mathbf{C}]$ operator, device-resident sensor buffering, and non-finite detection; verified against the distributed PETSc engine.
- **Device handling:** Strict device requests (explicit GPU requests fail instead of silently falling back), MPI-aware selection, and provenance in `simulation_config.yaml`.
- **Configuration:** Strict schema with suggestions, typed validation, a single table of defaults, `solver.device`/`solver.precision` sections, and new output options (`compress_output`, `figure_formats`, `usetex`).
- **Materials:** Fields ignored by the chosen material model are rejected (e.g. `Cv` on a non-`viscoelastic` layer is an error instead of being silently dropped).
- **Figures:** Labels render identically with and without LaTeX; no global Matplotlib/PATH side effects.
- **Excitation:** Hann-windowed $N$-cycle tone burst (`source_type: tone_burst`) next to the Ricker wavelet.
- **Scale awareness:** Units in logs, preflight and figures follow the problem scale (seismic to ultrasonic); dispersion limits are SI or unit strings; sponge damping defaults scale with $f_0$.
- **Examples:** One documented example per structure mode (NDE, aerospace, rail, seismic, biomedical).
- **Engineering:** `logging`-based output with `-q/-v`, lazy imports, vectorized operator bounds, `Gatherv` sensor collection, physics verification test, hardened CI, pre-commit.

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
  version      = {0.3.0},
  publisher    = {GitHub},
  url          = {https://github.com/ductho-le/GuidedWaveFEM}
}
```
