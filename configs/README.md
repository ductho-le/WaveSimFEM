# Example configurations

These are runnable examples demonstrating different waveguide physics, boundary conditions, and material models. They are categorized into three computational tiers based on their resolution and memory requirements.

Use `--dry-run` to preview expected mesh resolution, DOF counts, step counts, and memory requirements before running:

```sh
python guided_waves.py -c configs/01_single_layer_aluminum.yaml --dry-run
```

---

## Configuration Tiers & Computational Profiles

### Tier 1: Quick Verification & Tutorials (Workstation / Laptop Friendly)
*Fast, lightweight verification cases designed to run in under a minute on a personal computer.*

| Case | System | Model | Typical Runtime | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [01](01_single_layer_aluminum.yaml) | Single aluminum plate | `1_layer` (explicit) | ~30–45 s | Fundamental Lamb waves ($S_0, A_0$). Also in [TOML](01_single_layer_aluminum.toml) & [JSON](01_single_layer_aluminum.json). |
| [09](09_bonedc_case1_cortical_bone.yaml) | BoneDC cortical bone plate | `1_layer` (explicit) | ~1–2 min | 4 mm plate benchmark with $f_0 = 0.5$ MHz. |

### Tier 2: Multilayer Benchmarks & Viscoelastic Laminates
*Moderate scale (~0.5M–4.5M DOFs). Recommended for multi-core MPI (`mpiexec -n 4`).*

| Case | System | Model | Solvers | Key Phenomena |
| :--- | :--- | :--- | :--- | :--- |
| [02](02_single_layer_cortical_bone_viscoelastic.yaml) | Cortical bone (Kelvin–Voigt) | `1_layer` | Implicit | Viscoelastic attenuation, PETSc CG+GAMG |
| [03](03_multilayer_bone_aluminum.yaml) | Bone/aluminum laminate | `N_layers` | Explicit | Heterogeneous laminate, interfacial mode conversion |
| [10](10_bonedc_case2_cb_st_fluid.yaml) | Bone + soft tissue fluid | `N_layers` | Explicit | Coupled solid-fluid modes |
| [11](11_bonedc_case3_st_cb_mr_fluid.yaml) | Marrow + bone + soft tissue | `N_layers` | Explicit | Tri-layer biological waveguide |

### Tier 3: Truncated Semi-Infinite Halfspace Models (HPC / Cluster Recommended)
*Large-scale models ($\sim 17\text{M}$ to $>150\text{M}$ DOFs). Require multi-node distributed MPI and $>32$ GB RAM.*

| Case | System | Substrate / Superstrate Depth | Key Phenomena |
| :--- | :--- | :--- | :--- |
| [04](04_halfspace_rayleigh_aluminum.yaml) | Truncated aluminum halfspace | 40 mm aluminum (25 mm sponge) | Rayleigh surface waves ($c_R \approx 2906$ m/s) |
| [05](05_plate_on_halfspace_bone_marrow.yaml) | Bone plate on marrow substrate | 12 mm marrow (8 mm sponge) | Leaky Lamb waves & substrate radiation |
| [06](06_plate_in_halfspaces_bone_tissue_marrow.yaml) | Bone between tissue and marrow | 12 mm substrate + 12 mm superstrate | Stoneley/Scholte interface waves |
| [07](07_plate_on_water_halfspace.yaml) | Aluminum on regularized water | 12 mm water (8 mm sponge) | Scholte waves & leaky Lamb radiation |
| [08](08_plate_immersed_in_water.yaml) | Aluminum immersed in water | Dual 12 mm water halfspaces | Bilateral fluid loading |

---

## Execution Examples & CLI Overrides

### 1. Preflight Estimation (Inspect before execution)
```sh
python guided_waves.py -c configs/01_single_layer_aluminum.yaml --dry-run
python guided_waves.py -c configs/05_plate_on_halfspace_bone_marrow.yaml --dry-run
```

### 2. Fast Tutorial Run (Case 01)
```sh
python guided_waves.py -c configs/01_single_layer_aluminum.yaml
```

### 3. Parallel Execution with CLI Overrides
You can throttle frequency or duration directly from the CLI without editing configuration files:
```sh
# Override mesh sizing frequency and duration for quick evaluation
mpiexec -n 4 python guided_waves.py -c configs/03_multilayer_bone_aluminum.yaml \
    --max-frequency 1.0e6 --duration 30e-6
```

---

## Version 0.2 Numerical Notes

1. **Explicit Solver:** Uses 4th-order classical RK4 with lumped mass, replacing the older damped Velocity Verlet.
2. **Conservative Operator Bounds:** Critical time steps are determined by absolute row sums of assembled stiffness and damping operators ($\Delta t_{\text{crit}} = 1.0 / \max(\omega_{\max}, \text{damping\_bound})$).
3. **Mesh Sizing:** By default, automatic mesh sizing resolves wavelengths down to `mesh.max_frequency` (default `3 * f_center`). For large domains, specify `--max-frequency <f0>` or set `max_frequency` explicitly in the configuration.
4. **Fluid Formulation:** The `fluid` model is a regularized small-shear elastic approximation ($\mu_f = 10^{-8} K_f$) with full tangential displacement continuity, not mixed inviscid acoustics.
