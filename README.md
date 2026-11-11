<p align="center">
  <img alt="WaveSimFEM" src="assets/logo-banner.svg" width="680">
</p>

<p align="center">
  <strong>A GPU-accelerated finite element framework for acoustic and elastic wave propagation in layered media</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://fenicsproject.org/"><img src="https://img.shields.io/badge/DOLFINx-0.10.0-orange.svg" alt="DOLFINx 0.10.0"></a>
  <a href="https://petsc.org/"><img src="https://img.shields.io/badge/MPI-PETSc-green.svg" alt="MPI / PETSc"></a>
  <a href="https://github.com/ductho-le/WaveSimFEM/actions/workflows/tests.yml"><img src="https://github.com/ductho-le/WaveSimFEM/actions/workflows/tests.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

**WaveSimFEM** is an open-source, user-friendly Python framework for simulating acoustic and elastic wave propagation in layered media. Built on [DOLFINx](https://fenicsproject.org/) and [PETSc](https://petsc.org/), with optional GPU acceleration via [CuPy](https://cupy.dev/), it provides a seamless workflow: **configure in YAML → simulate → record virtual sensors → analyze dispersion**.

Whether you are studying guided Lamb waves in aerospace composites, Rayleigh waves in seismology, ultrasound in cortical bone, or acoustic waves in fluid–solid channels, WaveSimFEM makes complex multi-physics finite element wave modeling accessible in minutes.

---

## ✨ Key Features

- ⚡ **High Performance & GPU Acceleration**: Stage-consistent explicit Runge–Kutta 4 (RK4) integration with GPU acceleration via CuPy and distributed multi-core scaling via MPI/PETSc.
- 🌊 **Coupled Acoustic–Structure Interaction**: True pressure–displacement formulation enforcing normal traction and velocity continuity while naturally allowing tangential fluid slip.
- 🧱 **Rich Material Library**: Supports isotropic elastic, anisotropic elastic, and Kelvin–Voigt viscoelastic solids, coupled to inviscid acoustic fluids.
- 📐 **Zero-Boilerplate Configuration**: Define models, meshes, source excitations, and sensor arrays in simple, human-readable YAML/TOML/JSON files.
- 📊 **Built-in Acquisition & Spectral Analysis**: Virtual point sensor arrays automatically record histories (displacement, pressure, velocity) and generate B-scans and 2D FFT frequency–wavenumber ($f$–$k$) dispersion spectra.
- 🛡️ **Guaranteed Stability**: Built-in Gershgorin–Rayleigh spectral bound automatically ensures stable time steps $\Delta t$ for explicit integration.

---

## 🚀 Quick Start (in 3 Minutes)

### 1. Installation

Set up the environment using Conda (Linux & macOS):

```bash
git clone https://github.com/ductho-le/WaveSimFEM.git
cd WaveSimFEM
conda env create -f environment.yml
conda activate wavesim
pip install --no-deps -e .
```

*(Optional for NVIDIA GPU users)*:
```bash
pip install -e ".[cuda]"
```

<details>
<summary>💡 Note for macOS users</summary>

If Xcode linker issues arise (`unknown architecture arm64e`), prefix the run command with Apple's compiler:
```bash
CC=/usr/bin/clang LDSHARED="/usr/bin/clang -bundle -undefined dynamic_lookup" wavesim -c configs/01_plate_aluminum_lamb.yaml
```
</details>

---

### 2. Run Your First Simulation

Run the aluminum plate guided-wave example (1 MHz Lamb waves):

```bash
# Preview model mesh and memory without running (dry run)
wavesim -c configs/01_plate_aluminum_lamb.yaml --dry-run

# Run simulation on CPU and produce waveforms & dispersion plots
# (Add --no-usetex if external LaTeX/dvipng/Ghostscript are not installed)
wavesim -c configs/01_plate_aluminum_lamb.yaml
```

The results (sensor waveforms, B-scan image, and $f$–$k$ dispersion spectra) will be saved in `results_01_plate_aluminum_lamb/`.

---

### 3. Inspect or Reprocess the Data

Re-generate figures or adjust frequency/wavenumber display limits without re-running the physics:

```bash
wavesim --postprocess results_01_plate_aluminum_lamb/sensor_data.npz --f-max "2 MHz"
```

Or load the recorded waveforms directly in Python:

```python
import numpy as np

data = np.load("results_01_plate_aluminum_lamb/sensor_data.npz")
times = data["times"]  # Time vector [s]
sensors_x = data["sensor_x"]  # Sensor positions along propagation axis [m]
u_y = data["sensor_data_y"]  # Vertical displacement histories [m]

print(f"Recorded {u_y.shape[1]} sensors over {len(times)} time steps.")
```

---

## 💻 Execution: CPU, GPU, or MPI

You can switch computation backends with a simple CLI flag:

```bash
# Single CPU core (SciPy)
wavesim -c configs/01_plate_aluminum_lamb.yaml --device cpu

# NVIDIA GPU (CuPy CUDA)
wavesim -c configs/01_plate_aluminum_lamb.yaml --device cuda

# Multi-core CPU distributed with MPI (4 processes)
mpiexec -n 4 wavesim -c configs/01_plate_aluminum_lamb.yaml --device cpu

# Implicit Newmark-β solver (monolithic consistent-mass via PETSc)
mpiexec -n 4 wavesim -c configs/01_plate_aluminum_lamb.yaml --solver implicit
```

---

## 📚 Flagship Examples

WaveSimFEM includes 8 research-ready configurations under [`configs/`](configs/):

| Configuration | Structure | Physics & Application | Excitation |
| :--- | :--- | :--- | :--- |
| [`01 · Aluminum plate`](configs/01_plate_aluminum_lamb.yaml) | 3 mm isotropic plate | Lamb waves & NDT inspection | 1 MHz Ricker |
| [`02 · CFRP composite`](configs/02_plate_cfrp.yaml) | 8-ply `[0/90/0/90]_s` laminate | Anisotropic guided-wave dispersion | 1 MHz Ricker |
| [`03 · Steel halfspace`](configs/03_halfspace_steel_rayleigh.yaml) | Truncated elastic halfspace | Rayleigh surface waves | 1 MHz Tone burst |
| [`04 · Soil over bedrock`](configs/04_plate_on_halfspace_soil_bedrock.yaml) | 10 m soil layer on bedrock | Seismic surface-wave dispersion | 15 Hz Ricker |
| [`05 · Cortical bone`](configs/05_layers_bone_qus.yaml) | Soft tissue / bone / marrow | Axial-transmission ultrasound | 500 kHz Ricker |
| [`06 · Immersed plate`](configs/06_plate_in_halfspaces_immersed_water.yaml) | Aluminum plate in water | Leaky Lamb & Scholte waves | 500 kHz Ricker |
| [`07 · Water channel`](configs/07_fluid_acoustic.yaml) | Acoustic water cavity | Pure acoustic pressure waves | 250 kHz Ricker |
| [`08 · Coupled fluid–solid`](configs/08_fluid_solid_acoustic.yaml) | Aluminum plate on water | Fluid–structure interaction | 200 kHz Ricker |

---

## 🛠️ Configuration at a Glance

A WaveSimFEM configuration is self-explanatory. You can generate a starter template with:

```bash
wavesim --generate-config my_waveguide.yaml
```

Here is an example of how intuitive a model configuration is:

```yaml
solver:
  type: explicit
  CFL: 0.8

structure:
  mode: 1_layer
  single_layer:
    name: aluminum
    thickness: 3.0e-3
    rho: 2700.0
    E: 70.0e9
    nu: 0.33

geometry:
  sensing_length: 0.10       # 100 mm sensing zone
  absorb_length: 0.03        # 30 mm lateral absorbing sponges

mesh:
  nx: 320
  ny_per_region: [12]
  degree: 2                  # Quadratic (Q2) quadrilateral elements

excitation:
  source_type: ricker
  f_center: 1.0e6            # 1 MHz pulse
  traction_max: 1.0e6
  source_fy: -1.0

sensors:
  n_sensors: 64
  sensor_offset: 5.0e-3
  sensor_pitch: 1.0e-3       # 1 mm sensor spacing
```

*See [configs/template.yaml](configs/template.yaml) for a complete, annotated reference of all available parameters.*

---

## 📁 Output Artifacts

Every simulation outputs structured, reproducible archives:

- **`sensor_data.npz`**: Compressed NumPy archive storing timestamps, sensor positions, recorded physical observables (`sensor_data_x`, `sensor_data_y`, `sensor_data_pressure`), units, and run metadata.
- **`simulation_config.yaml`**: Full resolved configuration and runtime provenance (mesh details, realized $\Delta t$, versions, hashes).
- **`full_analysis.png`** / **`pdf`**: Multi-panel visualization containing time-domain waveforms, B-scan spatio-temporal waterfall plots, and frequency–wavenumber ($f$–$k$) dispersion spectra.
- **`guided_wave.bp`** / **`solid_displacement.bp`** / **`acoustic_pressure.bp`** *(optional)*: 2D wavefield snapshots viewable in ParaView (`output.write_vtx: true`).

---

## 🧪 Testing & Verification

WaveSimFEM is verified against analytical dispersion relations (Lamb, Rayleigh, Scholte modes), acoustic cavity resonances, and transmission benchmarks:

```bash
# Run test suite
pytest -m "not slow"

# Run acoustic verification suite
python tools/verify_acoustics.py --case all
```

---

## 📖 Citation

If you use WaveSimFEM in your research, please cite:

```bibtex
@misc{le2026wavesimfem,
  author       = {Le, Ductho},
  title        = {{WaveSimFEM}: A {GPU}-accelerated finite element framework for acoustic and elastic wave propagation in layered media},
  year         = {2026},
  howpublished = {\url{https://github.com/ductho-le/WaveSimFEM}},
  note         = {Manuscript in preparation}
}
```

*Software metadata is also available in [`CITATION.cff`](CITATION.cff).*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
