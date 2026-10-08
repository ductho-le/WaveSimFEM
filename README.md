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
| [`guidedwavefem.solvers_gpu`](src/guidedwavefem/solvers_gpu.py) | Backend-agnostic RK4 engine with preallocated stage vectors for single-process runs on CPU (SciPy) or CUDA (CuPy/PyTorch); exact sparse sensor operator; device-resident sensor buffering; non-finite detection. |
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

Bulk wave speed bounds are derived from the eigenvalues of the orthonormal Kelvin-scaled tensor $\mathbf{S} \mathbf{C} \mathbf{S}$, where $\mathbf{S} = \mathrm{diag}(1, 1, \sqrt{2})$. Guided modes can have shorter wavelengths and slower packet speeds. Choose and validate explicit mesh counts for anisotropic structures; `mesh.min_wavelength` constrains automatic source, sensor and display defaults, while `solver.min_group_velocity` constrains the arrival-time estimate.

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
\Delta t_{\mathrm{crit}} = \frac{R}{\max\left(\sqrt{B_K}, B_C\right)}, \qquad R = 2.6, \qquad \Delta t = \mathrm{CFL} \cdot \Delta t_{\mathrm{crit}} \quad (0 < \mathrm{CFL} \le 1)
$$

The integer step count is fitted exactly: $N_{\mathrm{steps}} = \lceil T_{\mathrm{total}} / \Delta t \rceil, \, \Delta t_{\mathrm{actual}} = T_{\mathrm{total}} / N_{\mathrm{steps}}$ (then rounded up to a multiple of the sensor stride, see below).

**Why $R = 2.6$.** For a mode with frequency $\omega$ and damping rate $\zeta$ (both nonnegative, $\zeta \le B_C$, $\omega^2 \le B_K$), the eigenvalues of the first-order system satisfy $|\lambda| \le \max(\omega, \zeta) \le \max(\sqrt{B_K}, B_C)$ and $\mathrm{Re}\,\lambda \le 0$. Every $\lambda\,\Delta t$ therefore lies in the closed left half-disk of radius $\Delta t \max(\sqrt{B_K}, B_C)$. The largest such half-disk inside the RK4 stability region $|1 + z + z^2/2 + z^3/6 + z^4/24| \le 1$ has radius $2.6156$ (computed by [`rk4_half_disk_radius`](src/guidedwavefem/numerics.py) and checked in the tests); the constant [`RK4_STABILITY_RADIUS`](src/guidedwavefem/numerics.py) $= 2.6$ keeps a small margin.

Gershgorin row sums can only *over*-estimate the spectral radius, so the step is guaranteed stable. Supply `solver.CFL` explicitly in `(0, 1]`; there is no implicit CFL default. The mesh uses the exact element counts in the configuration. The bound includes the assembled sponge and viscous damping, so strongly damped sponges or Kelvin–Voigt layers can reduce $\Delta t$ (a warning is printed when damping is the limiting term). The `--dry-run` preflight evaluates the same bound element-by-element before meshing; that value is never larger than the solver's.

##### (e) Execution Engines
- **Single process** (`mpiexec -n 1` or no MPI): the array engine in [`solvers_gpu.py`](src/guidedwavefem/solvers_gpu.py) advances the stacked state $[\mathbf{u};\mathbf{v}]$ with one sparse product $[\mathbf{K}\,|\,\mathbf{C}]\,[\mathbf{u};\mathbf{v}]$ per stage, preallocated buffers, and a bounded on-device sensor buffer flushed to the host in chunks. Backends: SciPy (CPU), CuPy or PyTorch (CUDA).
- **Multiple MPI ranks**: the distributed PETSc engine ([`PetscRK4`](src/guidedwavefem/numerics.py)) applies the identical RK4 scheme with parallel PETSc operators, preallocated stage vectors, and fused vector kernels (no per-step allocations). Damping matrices are stored without their structural zeros, so the damping product costs only the sponge and viscoelastic volume.

Both engines are verified to agree to a relative error below $10^{-8}$ (see `tests/test_mpi.py`).

##### (f) Sensor Sampling
The solver step is usually far finer than the signal bandwidth requires, so sensor traces are stored every `stride` steps. By default the archive is sampled at 20 samples per period of `max_frequency`; `sensors.sample_rate` (e.g. `"20 MHz"`) overrides this. The stride is always reduced so that the Nyquist frequency stays above $1.25\times$ both the dispersion plot range and the $3 f_0$ band-pass edge, and so that at least 32 samples are kept. The step count is a multiple of the stride, so `times` is uniform and ends exactly at $T_{\mathrm{total}}$. The stride is recorded as `runtime.sample_stride` in `simulation_config.yaml`.

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

The effective matrix and $\mathbf{M}$, $\mathbf{C}$ are assembled once; each step forms $\mathbf{F}_{\mathrm{eff}}$ with two sparse matrix-vector products and solves one linear system.

##### (c) Time Step (Accuracy-Driven)
Trapezoidal Newmark ($\beta = 1/4$, $\gamma = 1/2$) is unconditionally stable and non-dissipative; its only time error is a period elongation $\Delta T / T \approx (\omega \Delta t)^2 / 12$. The step is therefore set by accuracy at the selected analysis frequency $f_{\max}$ (`max_frequency`), not by the smallest element:

$$
\Delta t = \frac{1}{N_{\mathrm{ppp}} \, f_{\max}}, \qquad \frac{\Delta T}{T} \approx \frac{\pi^2}{3 N_{\mathrm{ppp}}^2}
$$

with $N_{\mathrm{ppp}}$ = `solver.steps_per_period` (default 40, i.e. a 0.2 % phase error per period at $f_{\max}$; minimum 4). Thin adhesive or coating layers no longer shrink the implicit step.

##### (d) PETSc Solvers
Parallel MUMPS direct LU for moderate sizes; above the iterative threshold, Conjugate Gradients with algebraic multigrid (GAMG). GAMG receives the three rigid-body modes (two translations and the in-plane rotation) as its near-nullspace, which it needs to build effective coarse spaces for elasticity.

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

The surface (or interface) traction is $\mathbf{t}(x, t) = T_{\max} \, g(x) \, s(t) \, \hat{\mathbf{d}}$ with a Gaussian strip $g(x) = \exp[-(x - x_s)^2 / (2\sigma^2)]$, $\sigma$ = `excitation_width` / 3, direction $\hat{\mathbf{d}} \propto$ (`source_fx`, `source_fy`), and a time function selected by `excitation.source_type`:

| `source_type` | $s(t)$ | Use |
| :--- | :--- | :--- |
| `ricker` (default) | $(1 - 2a)\,e^{-a}$, $a = [\pi f_0 (t - 1.5/f_0)]^2$ | Broadband pulse; best for extracting dispersion curves with the 2D FFT |
| `tone_burst` | $\tfrac12\left[1 - \cos(2\pi f_0 t / N)\right] \sin(2\pi f_0 t)$ on $[0, N/f_0]$ | Narrowband, mode-selective NDE/SHM excitation; `n_cycles` $= N$ (default 5, $-6$ dB bandwidth $\approx 2f_0/N$) |

Unless `solver.duration` is given, the simulated time is `time_factor` $\times D/c_{\mathrm{packet}}$ plus the pulse length ($3/f_0$ for the Ricker wavelet, $N/f_0$ for the tone burst). Here $D$ is the larger of `sensing_length` and the absolute source-to-receiver distances at either end of the array. The packet-speed estimate starts from the bulk/Rayleigh reference. For a free homogeneous isotropic plate, it also includes A0 group velocity sampled over the arrival-sizing band (`mesh.min_frequency`, default $f_0/5$, through `mesh.max_frequency`, default $3f_0$). These settings do not filter the traces. Higher Lamb modes can travel more slowly near their cutoffs; preflight warns when the band reaches their first cutoff and no arrival target is supplied. Set `solver.min_group_velocity` for the slowest packets of interest, or specify `solver.duration` directly.

All inputs are SI, and nothing in the solver assumes ultrasonic scales: logs, the preflight box and figures choose units from the data (μs/mm/MHz for an aluminium plate, ms/m/Hz for a soil profile). Dispersion plot limits accept SI numbers or unit strings (`"2 MHz"`, `"40 Hz"`, `"3 rad/mm"`, `"1.2 rad/m"`).

Source and sensor geometry left as `null` scales with the sizing wavelength $\lambda_{\mathrm{ref}} = \lambda_{\mathrm{size}}(f_0)$. At frequency $f$, $\lambda_{\mathrm{size}}(f)$ is the smaller of the reference-speed wavelength ($c_R/f$ for a halfspace, otherwise the slowest physical bulk speed divided by $f$), the A0 wavelength for a free homogeneous isotropic plate, and `mesh.min_wavelength` when supplied. Other guided structures need independently validated bounds. With $L$ = `sensing_length`, $w$ = `excitation_width`, and $\lambda_{\mathrm{band}} = \lambda_{\mathrm{size}}(\texttt{max_frequency})$:

| Key | Default |
| :--- | :--- |
| `excitation_width` | $\min(\lambda_{\mathrm{ref}}/2, \, L/10)$ |
| `source_x` | `absorb_length` $+ \max\left[\min(\max(\lambda_{\mathrm{ref}}, w), \, L/8), \, w/2\right]$ |
| `sensor_offset` | $\min(2\lambda_{\mathrm{ref}}, \, L/4)$ |
| `sensor_pitch` | $\min(\lambda_{\mathrm{ref}}/8, \, \lambda_{\mathrm{band}}/4, \, L/100)$ |
| `n_sensors` | fill the rest of the sensing region (at most 1001) |

The domain-fraction caps keep fully automatic layouts inside the sensing region even when $\lambda_{\mathrm{ref}}$ is comparable to the domain. An explicit load width also enforces a half-width source clearance; incompatible layouts are rejected. The resolved values are written to `simulation_config.yaml`.

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

All simulations require an explicit configuration file via `-c` / `--config`, including exact mesh counts (`mesh.nx`, `mesh.ny_per_region`). Explicit runs also require a user-selected `solver.CFL`; the solver then calculates the time step from the assembled operators. Simulations can be launched using either the installed CLI `guided-waves` or module invocation `python -m guidedwavefem`.

### 1. Running a Simulation

```bash
guided-waves -c configs/01_plate_aluminum_lamb.yaml
# Or alternatively:
python -m guidedwavefem -c configs/01_plate_aluminum_lamb.yaml
```

### 2. Parallel MPI Execution

Distribute simulation work across processes (uses the distributed PETSc engines):

```bash
mpiexec -n 4 guided-waves -c configs/05_layers_bone_qus.yaml
# Or:
mpiexec -n 4 python -m guidedwavefem -c configs/05_layers_bone_qus.yaml
```

**Threads per rank.** Parallelism comes from MPI (one rank per core). Threaded BLAS/OpenMP libraries would otherwise start one thread per core *in every rank* and oversubscribe the machine. When launched through `guided-waves` or `python -m guidedwavefem` under `mpiexec`/`srun`, each rank is therefore capped at `cpus // ranks_per_node` threads (Open MPI, MPICH/Intel MPI, MVAPICH2 and Slurm launch variables are recognised). The setting is reported in the run banner (`Threads: ...`). It never overrides a value you set yourself, e.g. `OMP_NUM_THREADS=1 mpiexec -n 8 guided-waves ...`. Library users calling `run_simulation` from their own scripts can call `guidedwavefem.limit_threads_per_rank()` before importing NumPy.

### 3. Preflight Resource Estimation (`--dry-run`)

Preview the supplied mesh counts, actual element sizes, DOF counts, time step limits, and memory requirements without allocating FEM structures:

```bash
guided-waves -c configs/01_plate_aluminum_lamb.yaml --dry-run
```

Units follow the problem scale (the seismic example reports metres, milliseconds and hertz). The preflight uses the same mesh-layout and sampled-time-grid code as the solver. The implicit step follows the same accuracy rule as the solver. The automatic explicit step uses a conservative element bound. A requested explicit `dt` beyond that certificate produces a warning and conservative resource estimates; only the assembled-operator check can accept or reject that request. Final step counts reflect rounding to the sensor stride.

> [!NOTE]
> Supply `mesh.nx` and `mesh.ny_per_region` as exact positive element counts. A Q2 element carries two node spacings. Changing material properties, frequency, or polynomial degree never changes these counts; validate spatial convergence by explicitly refining them.

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

# Override duration, analysis frequency, and the user-supplied CFL factor
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

The overview reports the applied filter, propagation direction and unpadded FFT
spacing. Display limits stop at the stored sampling Nyquist limits. When no
wavenumber limit is archived or supplied, the entire spatial band is shown so
slow guided waves are not cropped by a bulk-wave speed assumption. Arrays wholly
left of the source use the `-x` spectrum; other arrays use `+x`. The opposite
direction is omitted from this overview. Receiver coordinates take precedence
over legacy pitch/offset metadata. Displacement colours saturate at the robust
display limits shown on flat-ended colourbars; each dispersion component has its
own power reference. Display saturation does not change the recorded measurements.

Structure-aware reports are enabled by default (`postprocess.structure_analysis:
true`). Their model and material references come from the **saved archive**, so
reprocessing with a plotting-only config cannot change the interpreted structure.
Each run has an organized `wave_physics/` directory with two four-panel figures:

- `dispersion_and_modes`: signed frequency–wavenumber spectrum,
  phase-velocity coherence image, qualified
  measured peaks, and a diagnostic selected for the structure.
- `propagation_and_components`: horizontal/vertical wavefields on one physical
  amplitude scale, component bandwidth and balance on one spectral reference,
  and counter-propagating spectral content.

| Structure | Physical references and selected diagnostic |
|---|---|
| Free isotropic plate | A0/S0 Rayleigh–Lamb curves and coherent vertical/horizontal displacement ratio. |
| Bonded or viscoelastic layers | Constituent bulk references and frequency-dependent apparent spatial decay, with exponential-fit screening. |
| Solid isotropic halfspace | Rayleigh/P/SV speeds and surface Rayleigh V/H comparison when receivers are on the free surface. |
| Soil or other layers on a solid halfspace | Constituent speed comparisons and measured wavelength relative to finite-layer thickness. |
| Bone/tissue or other solid–fluid layers | Coupled guided-wave context, axial elastic quasi-bulk references where applicable, and fluid sound-line comparison. |
| Fluid-loaded/immersed plate | Fluid sound lines and measured phase-speed/sound-speed ratio, with subsonic/evanescent and propagating-field context. |

Only proven free isotropic plates receive A0/S0 theory overlays. Constituent
speeds in other structures are comparison lines, not solutions of the complete
layered dispersion problem. Plane-strain reports do not introduce SH/Love modes,
and artificial fluid shear stiffness is never displayed as a physical wave speed.

The phase image uses amplitude-weighted phase alignment, related to
[Park, Miller and Xia's phase-shift imaging method](https://www.parkseismic.com/wp-content/uploads/2021/06/ParkEtAl1998.pdf).
Its absolute 0–1 coherence scale does not brighten each frequency independently.
Support requires sufficient relative power, at least four effective receivers,
and wavelengths between the spatial alias limit and array aperture. Measured
peaks additionally require an isolated interior maximum, adequate coherence and
two wavelengths across the amplitude-weighted aperture. These are diagnostic
screening rules, not confidence intervals or automatic mode identification.
Phase velocity is distinct from packet/group velocity. For arrays straddling
the source, phase analysis selects the larger one-sided aperture and records
the receiver indices; the propagation panels retain the whole array.

`report.json` records structure provenance, references, methods and exclusions;
four CSV tables retain phase-velocity, polarization, decay and directionality
measurements and their acceptance status. `spectral_arrays.npz` preserves the
coherence image, support mask and selected receivers. Missing or unsupported
measurements remain unavailable. Spatial-decay fits use untapered full-record
Fourier amplitudes to avoid arrival-dependent taper bias; modal interference,
spreading and record truncation still prevent interpreting them as intrinsic
material loss. Directional power is squared displacement, not energy flux or a
reflection coefficient. Wavelength/thickness is not an inferred probing depth.

For quantitative receiver measurements, enable the packet report in a small
post-processing config:

```yaml
postprocess:
  wave_characterisation:
    enabled: true
    component: y
    frequency_band: ["0.3 MHz", "0.8 MHz"]
    time_window: null # seconds; select an interval containing one complete packet
    min_relative_amplitude: 0.01
output:
  figure_formats: [png, pdf]
  usetex: false
```

Choose the frequency band for your record; the values above are an ultrasonic
example. This writes four figures and `characterisation/metrics.json` plus
`receivers.csv`:

| Output | Interpretation |
|---|---|
| `arrival_velocity` | Envelope peaks, accepted/excluded receivers, fitted travel time and residuals against one stored sample. |
| `pulse_evolution` | Normalised packet shapes, amplitude decay relative to the array peak, and envelope-energy RMS widths. |
| `particle_motion` | Time-coloured displacement trajectories with equal physical axes and common limits. |
| `time_frequency` | Hann-window spectrograms with one shared power reference and explicit window length/frequency spacing. |

Packet measurements remove the complete-record mean before optional filtering.
Three-point interpolation refines envelope-peak times. Screening excludes weak
signals, packets with a gate-edge envelope above 10% of the peak, and a second
resolved peak at least 50% as high. Velocity requires at least three accepted
receivers, arrival-fit R² ≥ 0.9 and a fitted travel-time span of at least two
stored samples. An amplitude-fit R² below 0.8 suppresses apparent attenuation,
except for effectively constant amplitude (log-amplitude range ≤ 0.01).
The JSON records these criteria, preprocessing, fit status and residuals; the
CSV gives each receiver's exclusion reason, peak, centroid, RMS width and FWHM.
These checks are diagnostic heuristics, not statistical confidence intervals.
Ungated multimode records can remain ambiguous: choose a complete isolated
packet before interpreting speed or spreading. Apparent attenuation includes
interference, spreading and numerical effects, and is not intrinsic material loss.

For a free, single isotropic plate with archived model provenance,
`postprocess.lamb_modes: true` additionally writes `lamb_modes/modal_resolution`,
`modal_separation` and `modal_waveforms`. The report shows both propagation
directions, theory branches, usable frequency intervals and sampling margins.
Input, A0, S0 and residual maps share one displacement scale. Its schema-v2
`diagnostics.json` and `modal_signals.npz` include reconstruction evidence for
both displacement components. Unresolved content stays in the residual; RMS
ratios are not additive modal energy fractions. Packet and Lamb JSON reports
use schema version 2; existing measurement keys remain available.

Every exported figure has a **190 mm canvas width** and **10-point text**,
including panel titles, axis labels, tick numbers, legends and colourbars.
Every panel has a complete four-sided box. Colourbars have flat ends, with
saturation behaviour documented in the accompanying README. Panels with the same
colour scale share a single colourbar; distinct quantities such as spectral power
and coherence retain separate bars. Compact panel gaps,
label padding and margins keep the plotting area prominent at the final size.
Heights follow the panel arrangement. Figures contain the panels, short panel
titles and essential axis/legend labels; figure-level titles, schematics and
explanatory prose are kept in accompanying README files. Each report directory
has its own README with provenance, processing, panel descriptions and limits.
The standard/quick figures write `full_analysis_README.md` and
`quick_overview_README.md` beside their outputs.

`output.usetex: false` is the default: text uses Matplotlib's bundled serif fonts
and mathtext, with no external LaTeX installation needed. Set `output.usetex: true`
to render text with LaTeX; a missing toolchain then raises an actionable error.
The legacy `output.usetex: auto` option uses LaTeX when the required tools are
available and built-in rendering otherwise. Both modes retain the same 190 mm
width, 10-point base font size and panel styling.

For external LaTeX rendering, `latex`, `dvipng` and Ghostscript must be on `PATH`.
On Debian/Ubuntu, install `texlive-latex-base texlive-latex-recommended
texlive-latex-extra cm-super dvipng ghostscript`; on macOS, a TeX distribution
such as MacTeX plus Ghostscript supplies the tools.

Rasters use 300 dpi by default; PDF/SVG/EPS retain vector text and lines while
large heatmaps are rasterised to keep files practical. Fixed-canvas export avoids
`tight` cropping, which would change the physical width. Raster dimensions round
to whole pixels (2244 pixels at 300 dpi); vector canvas width is exactly 190 mm.
Place the figure at its native width to preserve the 10-point text size.
Each export replaces its destination atomically after rendering.
To inspect reproducible analytical examples without a FEM solve:

```bash
python tools/preview_postprocessing.py --output results_postprocess_preview
python tools/preview_structure_reports.py --output results_structure_preview
```

The preview contains a packet with known speed/decay and a Rayleigh–Lamb A0/S0
superposition. These are synthetic verification signals, not simulation results.
The structure preview renders all six shipped structures with PNG/PDF reports
and a provenance manifest. Its layered/fluid-loaded packets have explicitly
prescribed illustrative dispersion, not independently solved physical modes.

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
  CFL: 0.5                 # Required for explicit: fraction (0, 1] of assembled RK4 stability limit
  steps_per_period: 40     # Implicit only: Newmark steps per period at max_frequency
  time_factor: 1.2         # T = time_factor * travel_distance / packet_speed + pulse length
  min_group_velocity: null # [m/s] Conservative guided-packet speed bound
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
  excitation_width: null   # [m] sigma = width / 3; auto wavelength and domain caps above
  source_x: null           # [m] Automatic load clearance from sponge (see Physical Scales)
  source_fx: 0.0           # Tangential traction component
  source_fy: 1.0           # Normal traction component (default: normal load)

mesh:
  nx: 1200                 # Required: exact elements along the full domain, including sponges
  ny_per_region: [12]       # Required: exact elements per region, bottom to top (one plate here)
  degree: 2                # Polynomial order (Q2 quadrilateral elements)
  max_frequency: null      # [Hz] Analysis/time-accuracy/sampling band (default 3 * f_center)
  min_frequency: null      # [Hz] Arrival-sizing band edge (defaults to f_center / 5)
  min_wavelength: null     # [m] Source/sensor/display default bound; never changes mesh counts

sensors:
  n_sensors: null          # null = fill the sensing region (at most 1001)
  sensor_offset: null      # [m] Source to first sensor (null = min(2 lambda_ref, L / 4))
  sensor_pitch: null       # [m] min(lambda_ref/8, lambda_band/4, L/100)
  sensor_y: null           # [m] Sensor line height (null = loaded surface/interface)
  sample_rate: null        # [Hz] Stored sampling rate (null = 20 * max_frequency)

postprocess:
  run_postprocess: true    # 2D FFT dispersion analysis and figures
  f_min: null              # Plot limits: SI numbers or unit strings ("2 MHz", "3 rad/mm")
  f_max: null              # null = 2 * f_center
  k_min: null              # null = 0
  k_max: null              # Auto sizing wavelength at f_max, limited by spatial Nyquist

output:
  output_dir: null         # Result directory (null = auto: results_guided_waves_{solver})
  write_vtx: false         # ADIOS2 VTX (.bp) Paraview export
  target_snapshots: 200    # Target snapshot count
  quick_plot: true         # Generate fast overview waterfall/B-scan plot
  compress_output: false   # Write sensor_data.npz with zlib compression
  history_storage: memory  # memory | disk (temporary maps; FFT postprocessing still needs RAM)
  figure_formats: [png, pdf, svg, eps]  # Any subset
  usetex: false            # true enables external LaTeX; default uses built-in fonts
```

`ny_per_region` follows every normalized region from bottom to top: a single plate
or halfspace has one entry; `N_layers` follows the layer list;
`plate_on_halfspace` starts with the substrate and then the layers;
`plate_in_halfspaces` starts with the substrate, then the layers, then the
superstrate. Sponge depths belong to their existing halfspace regions and do not
add entries. Counts remain fixed when material properties, frequencies or
polynomial degree change.

**Migrating existing configurations:** replace `mesh.elems_per_wavelength`,
`mesh.min_elems_x`, and `mesh.min_elems_y` with explicit `mesh.nx` and
`mesh.ny_per_region`, and remove every region's `min_elems_y`. The old fields now
raise a migration error instead of silently influencing resolution. To preserve
an old run's mesh, copy its actual `nx` and `ny_per_region` counts from the prior
preflight or runtime metadata. Explicit configurations must also supply
`solver.CFL`; `solver.dt` remains an optional checked override. The bundled
examples retain their previous resolved element counts and configured CFLs.

The schema is **strict**: unknown keys, misspelled section fields, unknown material fields, constitutive fields the chosen `material_model` would ignore (e.g. `Cv` on an `isotropic` layer), parameters defined twice, and non-boolean values for boolean switches are all rejected with a descriptive error (including a "did you mean" suggestion where possible). Run `guided-waves --generate-config template.yaml` for the full, commented schema.

---

## Pre-Configured Examples & Benchmark Setups

[`configs/`](configs/) holds **runnable benchmark examples covering every supported structure mode**, each built from a real application with documented materials and references (see [`configs/README.md`](configs/README.md)):

| Config file | Mode | Physical system | Application | Excitation |
| :--- | :--- | :--- | :--- | :--- |
| [`01_plate_aluminum_lamb.yaml`](configs/01_plate_aluminum_lamb.yaml) | `1_layer` | 3 mm aluminium plate in vacuum | Lamb-wave NDE / SHM ($S_0$, $A_0$, $A_1$) | Ricker, 1 MHz |
| [`02_layers_bonded_joint.yaml`](configs/02_layers_bonded_joint.yaml) | `N_layers` | Al / 0.2 mm viscoelastic epoxy / Al | Adhesive bond inspection (aerospace) | 5-cycle tone burst, 1 MHz; implicit solver |
| [`03_halfspace_steel_rayleigh.yaml`](configs/03_halfspace_steel_rayleigh.yaml) | `halfspace` | Rail steel, truncated halfspace | Rayleigh-wave surface-crack NDE ($c_R \approx 2.98$ km/s) | Ricker, 1 MHz |
| [`04_plate_on_halfspace_soil_bedrock.yaml`](configs/04_plate_on_halfspace_soil_bedrock.yaml) | `plate_on_halfspace` | 10 m soft soil on bedrock | Seismic MASW / site characterisation | Ricker, 15 Hz |
| [`05_layers_bone_qus.yaml`](configs/05_layers_bone_qus.yaml) | `N_layers` | Stratified solid-fluid plate: bone marrow / cortical bone / soft tissue | Axial-transmission bone QUS (Le et al., 2025) | Ricker, 500 kHz |
| [`06_plate_in_halfspaces_immersed_water.yaml`](configs/06_plate_in_halfspaces_immersed_water.yaml) | `plate_in_halfspaces` | Aluminium plate immersed in water | Ultrasonic immersion NDE (leaky Lamb & Scholte waves) | Ricker, 500 kHz |

Together the examples exercise the `isotropic`, `viscoelastic` and `fluid` material models, anisotropic stiffness (the full plane-strain tensor of the cortical bone in example 05, given through its `viscoelastic` layer), both excitation types, both time integrators, and the full frequency range from 15 Hz to 1 MHz. A purely elastic anisotropic layer uses `material_model: anisotropic` with the same `C` tensor and no `Cv`. Examples 05 and 06 use the regularized-elastic fluid model; see the limitations in [Constitutive Material Models](#d-acoustic-fluid-regularized-small-shear-elasticity).

> [!TIP]
> Run `guided-waves -c <config> --dry-run` to inspect the computational footprint first; example 06 (≈ 0.46 M DOFs) benefits from `mpiexec -n 4`.

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
| **Physics** | Rayleigh speed on a truncated halfspace; sponge residual | End-to-end run vs. closed-form $c_R$ (within 1 %); late-window residual after the direct pulse < 5 % |
| **Dispersive Physics** | A0 phase convergence with mesh and time refinement | Explicit and implicit FEM vs. an independent Rayleigh–Lamb determinant; finest phase error < 0.3 % |
| **Anisotropic / Viscous Physics** | Orthotropic extensional speed and Kelvin–Voigt attenuation | Long-wave analytical complex wavenumber; phase error < 0.2 %, nonzero attenuation error < 2 % |
| **Examples** | Every shipped configuration (all five structure modes) runs end to end | Shortened CLI runs (`tests/test_examples.py`, slow) and 2-rank CI smoke runs |
| **Damped Time Integration** | 4th-order temporal convergence | Analytical matrix-exponential oscillator solution $\exp(\mathbf{A}t)$ |
| **RK4 Step Limit** | Half-disk radius $R$ and stability at the rigorous limit | Numerical radius 2.6156 ≥ $R$ = 2.6; bounded energy at $0.999\,R/\max(\sqrt{B_K}, B_C)$ for wave- and damping-limited FE systems (detects instability at 3×) |
| **PETSc RK4** | Agreement with the functional reference step | `PetscRK4` vs. `rk4_step` on a damped system (rtol $10^{-12}$) |
| **Array RK4 Engine** | Bit-level agreement with the reference RK4 step | Random damped system, incl. tiny sensor ring buffers, decimated sampling, instability detection, optional Torch/CuPy backends |
| **Backend Accuracy** | Both precisions against an independent matrix exponential | SciPy and Torch CPU in CI; explicitly required CuPy/Torch CUDA on provisioned hardware; streaming, partial flushes and sink failures |
| **FE Sensor Evaluation** | Exact off-node evaluation | Distributed bounding-box search; sparse sensor operator vs. DOLFINx `eval` (shared/adjacent cells) |
| **Spectral Analysis** | Exact frequency and wavenumber peaks | Monochromatic traveling wave with exact $f_0$ and $k_0$ |
| **Stability Limits** | Assembled operator bounds $B_K$ and $B_C$; pre-mesh element bounds | SciPy generalized eigensolution $\mathbf{K}\mathbf{v} = \lambda \mathbf{M}_L \mathbf{v}$; element bounds ≥ assembled bounds (Q1, Q2) |
| **Implicit Solver Helpers** | GAMG near-nullspace; zero-filtered damping | Rigid-body modes orthonormal with $\mathbf{K}\mathbf{r} \approx 0$; filtered matrix reproduces the mat-vec |
| **Thread Policy** | Per-rank BLAS/OpenMP cap | Launcher variables of Open MPI, MPICH, MVAPICH2, Slurm; user settings respected; no heavy imports |
| **Material Energy** | Positive definiteness enforcement | Voigt tensor and viscosity eigenvalue validation |
| **Distributed Solver Agreement** | Serial vs. 2- and 4-rank PETSc runs, explicit and implicit | Coupled fluid–solid waveguide regression with relative error $< 10^{-8}$; memory and disk histories |
| **MPI Failure Handling** | Rank-local errors terminate peers | Subprocess timeouts for configuration errors, unsuccessful exits and interrupts; help/version remain successful |
| **Output Integrity** | Failed writes preserve the previous archive | Injected serialization/write/replace failures; committed archive embeds matching configuration; scratch cleanup |
| **Configuration Schema** | Strict, typed validation; one valid shipped example per structure mode | Dedicated schema tests (`tests/test_config.py`), example coverage test |
| **Excitation & Units** | Tone-burst window, spectrum and timing; unit picking and parsing | `tests/test_units_sources.py` |
| **Figures** | Portable labels, no global-state leakage, selectable formats, seismic-scale units | `tests/test_analysis.py` |
| **CLI & Orchestration** | Template, dry-run, postprocessing, verbosity, error paths | `tests/test_cli.py` |

---

## Version Compatibility & Numerical Changes

### Version 0.3 Highlights
- **Time stepping:** Explicit RK4 uses the rigorous half-disk radius $R = 2.6$ (2.6× larger steps, still provably stable); implicit Newmark is accuracy-driven (`steps_per_period`), so thin layers no longer force tiny steps; the preflight explicit estimate uses the same bound per element and never overestimates the step.
- **Performance:** Implicit right-hand side from two SpMVs (no per-step assembly), allocation-free `PetscRK4`, zero-filtered damping matrices, decimated sensor sampling (`sensors.sample_rate`), GAMG rigid-body near-nullspace, and per-rank BLAS/OpenMP thread capping under MPI. Historical benchmarks on four MPI ranks with earlier example meshes reduced example 03 from 296 s to 101 s and completed example 02 in 41 s instead of an estimated 11.5 min. Runtime depends on the configuration and hardware.
- **Explicit engine:** One backend-agnostic RK4 engine (SciPy/CuPy/PyTorch) with preallocated stage vectors, a stacked $[\mathbf{K}|\mathbf{C}]$ operator, device-resident sensor buffering, and non-finite detection; verified against the distributed PETSc engine. Sparse products allocate temporary results.
- **Device handling:** Strict device requests (explicit GPU requests fail instead of silently falling back), MPI-aware selection, and provenance in `simulation_config.yaml`.
- **Configuration:** Strict schema with suggestions, typed validation, a single table of defaults, `solver.device`/`solver.precision` sections, and new output options (`compress_output`, `figure_formats`, `usetex`).
- **Materials:** Fields ignored by the chosen material model are rejected (e.g. `Cv` on a non-`viscoelastic` layer is an error instead of being silently dropped).
- **Figures:** All exports use 190 mm width and 10-point text, with external LaTeX optional and off by default; no global Matplotlib/PATH side effects.
- **Excitation:** Hann-windowed $N$-cycle tone burst (`source_type: tone_burst`) next to the Ricker wavelet.
- **Scale awareness:** Units in logs, preflight and figures follow the problem scale (seismic to ultrasonic); dispersion limits are SI or unit strings; sponge damping and source/sensor geometry defaults scale with $f_0$ and the reference wavelength.
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
  author       = {Le, Ductho},
  title        = {{GuidedWaveFEM: Unified Guided \& Surface Wave Finite Element Framework}},
  year         = {2026},
  version      = {0.3.0},
  publisher    = {GitHub},
  url          = {https://github.com/ductho-le/GuidedWaveFEM}
}
```

## Dispersive defaults and conservative controls

Mesh resolution is always supplied explicitly through `mesh.nx` and
`mesh.ny_per_region`. For a free, homogeneous isotropic plate, automatic source,
sensor and display defaults include the fundamental A0 Rayleigh–Lamb branch. Recording duration uses A0
group velocity sampled over the arrival-sizing band (by default `f_center / 5` through
`max_frequency`, which defaults to `3 * f_center`) and the farthest receiver.
The lower band edge is finite: no finite duration captures arbitrarily slow
zero-frequency flexural waves or every higher-mode packet near a cutoff. A0
sizing does not bound those higher-mode arrival times. Preflight reports its
duration basis and warns when additional arrival targets are needed. Validate
convergence for the observable, frequency band and modes of interest.

Layered, anisotropic, viscoelastic, and fluid-loaded structures require explicit
mesh-convergence studies and arrival bounds from an independent dispersion
calculation or convergence study. Their bulk
speeds alone do not bound guided-mode wavelengths or arrival times. Conservative
controls, in SI units, are available for every structure:

```yaml
mesh:
  min_frequency: 20000       # arrival-sizing lower edge [Hz]; does not filter traces
  min_wavelength: 0.0005     # source/sensor/display default bound [m]; no mesh refinement
solver:
  min_group_velocity: 80    # slowest relevant packet speed [m/s]
```

These controls tighten automatic geometry, display and duration estimates;
they never alter mesh counts. Explicitly specified sensor pitch,
duration, or plot limits remain user choices. `min_wavelength` also constrains
automatic sensor pitch and expands the automatic wavenumber range, subject to
spatial Nyquist. Arrival guides use absolute source–receiver distances; dispersion
maps deliberately select right-going waves, including for arrays left of the source.

Preflight RAM includes recorded histories, collection copies, transfer buffers,
and estimated solver workspace. Sparse factorization fill and MPI imbalance can
exceed this estimate. Set `output.history_storage: disk` to record both solvers'
histories in temporary memory-mapped files and gather MPI measurements in bounded
chunks. Reserve scratch disk for roughly twice the measurement payload, plus the
staged archive and any previous output. Scratch files are removed on normal return
or a caught exception; a killed process or MPI abort can leave scratch directories.
OS page cache, timestamps, solver workspace and FFT postprocessing still consume
RAM, so disk mode does not guarantee lower peak RSS. For long runs, consider
`--no-postprocess` and analyze the saved archive separately with sufficient memory.

The completed sensor archive is atomically replaced and embeds its matching
configuration as `simulation_config_yaml`, with a shared `run_id`. A failure before
archive replacement preserves the previous archive and configuration. The YAML
sidecar is a convenience copy: an interruption after archive replacement can leave
it stale, while the embedded configuration remains authoritative. Read the
`simulation_config_yaml` scalar from `sensor_data.npz` to recover that exact
configuration. `tools/benchmark.py` records reproducible time and memory
measurements; actual CUDA and multi-node validation requires suitable hardware.

For custom array-engine workflows,
`integrate_explicit_rk4(history_callback=..., retain_history=False)` streams
chronological chunks without retaining full histories in RAM. Zero-sensor operators
still emit every sampled timestamp with a zero-column measurement array. The array
RK4 engine reuses stage buffers, but sparse products allocate temporary results.

Configuration files reject duplicate YAML/JSON keys, including nested material
keys and YAML merge-source mappings. YAML inheritance allows explicit overrides
and reused anchors; each mapping permits one merge directive. Fluid definitions
must use exactly one representation: a sound-speed alias,
a bulk-modulus alias, or the pair `E` and `nu`. SI prefix case is significant:
`mHz` means millihertz and `MHz` means megahertz.

## Resolution and computational cost

Choose `mesh.nx`, `mesh.ny_per_region`, and (for the explicit solver)
`solver.CFL` in the configuration. The solver calculates the explicit time step
from the assembled stiffness, damping and lumped mass, multiplied by that CFL;
the final step is rounded down to fit the duration and sensor sampling grid.
Stability does not certify spatial or temporal accuracy. Increase the supplied
mesh counts and reduce CFL (or implicit `dt`) until the observable of interest
converges.

`tools/compare_resolution.py` compares the configured counts with a specified
refinement factor while holding geometry, source and receiver pitch fixed. Its
three variants separate mesh refinement from time-step refinement. For example:

```bash
python tools/compare_resolution.py \
    --config configs/01_plate_aluminum_lamb.yaml \
    --output resolution_check --duration 40e-6 \
    --refinement-factor 1.5 --step-factor 0.5
```

The shorter recording and first nine receivers make this an early-arrival
sensitivity check. Validate the complete recording window and modes separately
before treating a configuration as converged.
