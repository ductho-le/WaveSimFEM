# Guided Wave FEM Simulation Configurations

This directory contains standardized, version-controlled simulation configuration files (`.yaml`, `.toml`, and `.json`) for the `guided_waves.py` unified solver.

---

## Available Pre-Configured Cases

| Config File | Structure Mode | Physical System & Material | Solvers | Key Phenomena |
| :--- | :--- | :--- | :--- | :--- |
| [`01_single_layer_aluminum.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/01_single_layer_aluminum.yaml) | `1_layer` | 1 mm Isotropic Aluminum plate in vacuum | Explicit / Implicit | Fundamental Lamb waves ($S_0$, $A_0$) |
| [`01_single_layer_aluminum.toml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/01_single_layer_aluminum.toml) | `1_layer` | 1 mm Isotropic Aluminum plate (TOML format) | Explicit | Demonstrates Python 3.11 standard library `tomllib` |
| [`01_single_layer_aluminum.json`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/01_single_layer_aluminum.json) | `1_layer` | 1 mm Isotropic Aluminum plate (JSON format) | Explicit | Standard JSON schema for API / automated workflows |
| [`02_single_layer_cortical_bone_viscoelastic.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/02_single_layer_cortical_bone_viscoelastic.yaml) | `1_layer` | 3 mm Anisotropic Cortical Bone (Kelvin-Voigt) | Implicit | Anisotropy, material damping, attenuation |
| [`03_multilayer_bone_aluminum.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/03_multilayer_bone_aluminum.yaml) | `N_layers` | 2 mm Bone Substrate + 1 mm Aluminum Coating | Explicit | Heterogeneous laminate, mode conversion |
| [`04_halfspace_rayleigh_aluminum.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/04_halfspace_rayleigh_aluminum.yaml) | `halfspace` | 40 mm Aluminum Halfspace (25 mm sponge) | Explicit | Non-dispersive Rayleigh waves ($c_R \approx 2906\text{ m/s}$) |
| [`05_plate_on_halfspace_bone_marrow.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/05_plate_on_halfspace_bone_marrow.yaml) | `plate_on_halfspace` | 2 mm Bone plate on 12 mm Marrow substrate | Explicit | Leaky Lamb waves, substrate radiation |
| [`06_plate_in_halfspaces_bone_tissue_marrow.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/06_plate_in_halfspaces_bone_tissue_marrow.yaml) | `plate_in_halfspaces` | 2 mm Bone embedded in Tissue & Marrow | Explicit | Interface / Stoneley waves, dual sponges |
| [`07_plate_on_water_halfspace.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/07_plate_on_water_halfspace.yaml) | `plate_on_halfspace` | 1.5 mm Aluminum plate on Water substrate | Explicit | Scholte interface waves ($c_S \approx 1470\text{ m/s}$), leaky Lamb radiation |
| [`08_plate_immersed_in_water.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/08_plate_immersed_in_water.yaml) | `plate_in_halfspaces` | 1.5 mm Aluminum plate immersed between dual Water halfspaces | Explicit | Bilateral fluid loading (immersion UT), symmetric leaky Lamb & dual Scholte modes |
| [`09_bonedc_case1_cortical_bone.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/09_bonedc_case1_cortical_bone.yaml) | `1_layer` | 4.0 mm Cortical Bone plate (BoneDC Case 1) | Explicit | Direct SCM validation benchmark, Lamb modes ($S_0, A_0, S_1, A_1$) |
| [`10_bonedc_case2_cb_st_fluid.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/10_bonedc_case2_cb_st_fluid.yaml) | `N_layers` | 4 mm Bone + 1 mm Soft Tissue fluid (BoneDC Case 2) | Explicit | SCM validation with acoustic fluid coating, coupled solid-fluid modes |
| [`11_bonedc_case3_st_cb_mr_fluid.yaml`](file:///Users/ductho/Library/CloudStorage/OneDrive-Personal/04%20CODE/GitHub/GuidedWaveFEM/configs/11_bonedc_case3_st_cb_mr_fluid.yaml) | `N_layers` | 1 mm Marrow fluid + 4 mm Bone + 1 mm Soft Tissue fluid (BoneDC Case 3) | Explicit | Full tri-layer biological waveguide (MR+CB+ST) benchmark against SCM |

---

## Execution Examples

### 1. Serial Run with a Specified Configuration File
```bash
python guided_waves.py --config configs/01_single_layer_aluminum.yaml
```

### 2. Parallel MPI Execution
```bash
mpirun -np 4 python guided_waves.py -c configs/03_multilayer_bone_aluminum.yaml
```

### 3. Command-Line Overrides
Command-line arguments take precedence over values specified inside the configuration file:
```bash
# Override solver scheme to implicit Newmark-beta
python guided_waves.py -c configs/01_single_layer_aluminum.yaml --solver implicit

# Override output directory
python guided_waves.py -c configs/04_halfspace_rayleigh_aluminum.yaml --output my_rayleigh_test
```

### 4. Generate a Blank / Default Template
```bash
python guided_waves.py --generate-config my_custom_case.yaml
```

---

## Configuration Schema & Unit Reference

All quantities are in SI units ($\text{m}$, $\text{s}$, $\text{kg}$, $\text{Pa}$, $\text{Hz}$):

```yaml
solver:
  type: "explicit"         # "explicit" (Velocity Verlet) or "implicit" (Newmark-beta)
  CFL: 0.5                 # Courant-Friedrichs-Lewy stability factor
  time_factor: 1.2         # Total simulated time factor: T_total = time_factor * L / c_T
  beta: 0.25               # Newmark-beta parameter (implicit solver only)
  gamma: 0.5               # Newmark-gamma parameter (implicit solver only)

geometry:
  sensing_length: 0.30     # [m] Pristine observation region length
  absorb_length: 0.15      # [m] 5th-order absorbing sponge width on lateral boundaries

structure:
  mode: "1_layer"          # "1_layer", "N_layers", "halfspace", "plate_on_halfspace", "plate_in_halfspaces"

  # Single layer definition (mode: "1_layer")
  single_layer:
    name: "Material Name"
    thickness: 1.0e-3      # [m] Layer thickness
    material_model: "isotropic" # "isotropic", "anisotropic", "viscoelastic", or "fluid"
    rho: 2700.0            # [kg/m³] Density

    # Options for isotropic solid:
    E: 70.0e9              # [Pa] Young's modulus
    nu: 0.33               # [-] Poisson's ratio

    # Options for acoustic fluid (material_model: "fluid"):
    c: 1480.0              # [m/s] Acoustic sound speed (or bulk modulus K = rho * c²)
    fluid_shear_reg: 1.0e-8 # [-] Shear modulus regularizer (mu_f = reg * K_f, default: 1e-8)

    # Options for anisotropic solid:
    C: [...]               # 6 Voigt stiffness coefficients [C11, C12, C16, C22, C26, C66]
    Cv: [...]              # 6 Voigt damping coefficients (viscoelastic only)
    min_elems_y: 6         # Minimum elements through thickness

  # Multilayer laminate definition (mode: "N_layers", "plate_on_halfspace", "plate_in_halfspaces")
  layers:
    - name: "Layer 1 (Bottom)"
      thickness: 2.0e-3
      ...
    - name: "Layer 2 (Top)"
      thickness: 1.0e-3
      ...

  # Pristine Halfspace (mode: "halfspace")
  halfspace:
    name: "Halfspace"
    thickness: 40.0e-3     # [m] Total truncation depth
    absorb_depth: 25.0e-3  # [m] Bottom absorbing sponge thickness
    ...

  # Substrate and Superstrate (modes: "plate_on_halfspace", "plate_in_halfspaces")
  substrate:
    name: "Bottom Substrate"
    thickness: 12.0e-3
    absorb_depth: 8.0e-3
    ...

  superstrate:
    name: "Top Superstrate"
    thickness: 12.0e-3
    absorb_depth: 8.0e-3
    ...

absorbing_boundaries:
  alpha: null              # [1/s] Peak mass-proportional damping (null = auto)
  beta: null               # [s] Peak stiffness-proportional damping (null = auto)

excitation:
  f_center: 1.0e6          # [Hz] Center frequency of Ricker wavelet
  traction_max: 1.0e6      # [Pa] Surface load amplitude
  excitation_width: 2.0e-3 # [m] Spatial Gaussian width
  source_x: null           # [m] Source x-position (null = absorb_length + 5 mm)
  source_fx: 1.0           # Horizontal traction vector component
  source_fy: 1.0           # Vertical traction vector component

mesh:
  elems_per_wavelength: 10 # Spatial discretization density
  min_elems_x: 100         # Longitudinal element floor
  min_elems_y: 4           # Transverse element floor
  degree: 2                # Polynomial order (Q2 quadrilateral elements)

sensors:
  n_sensors: 401           # Measurement points
  sensor_offset: 0.05      # [m] Distance from excitation to 1st sensor
  sensor_pitch: 0.5e-3     # [m] Distance between sensors

postprocess:
  run_postprocess: true    # Compute 2D FFT dispersion analysis
  f_min_disp_MHz: 0.0      # [MHz] Min frequency on dispersion axis
  f_max_disp_MHz: 2.0      # [MHz] Max frequency on dispersion axis
  k_min_disp: 0.0          # [rad/mm] Min wavenumber on dispersion axis
  k_max_disp: 3.0          # [rad/mm] Max wavenumber on dispersion axis

output:
  output_dir: null         # Result directory (null = auto: results_guided_waves_{solver})
  write_vtx: false         # ADIOS2 VTX (.bp) Paraview export
  target_snapshots: 200    # Target snapshot count
```
