"""Small standalone regression driver, invoked by the subprocess tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wavesimfem import run_simulation

run_simulation(
    cfg={
        "solver_type": sys.argv[2],
        "history_storage": sys.argv[3] if len(sys.argv) > 3 else "memory",
        "output_dir": sys.argv[1],
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "source_x": 0.004,
        "f_center": 5e5,
        "sensor_offset": 0.00113,
        "sensor_pitch": 0.00107,
        "n_sensors": 4,
        "duration": 3e-6,
        "max_frequency": 5e5,
        "nx": 11,
        "ny_per_region": [2, 2, 2],
        "CFL": 0.8,
        "run_postprocess": False,
        "structure_mode": "plate_in_halfspaces",
        "single_layer": {
            "thickness": 0.001,
            "rho": 2700.0,
            "E": 70e9,
            "nu": 0.33,
            "material_model": "isotropic",
        },
        "bottom_halfspace": {
            "thickness": 0.001,
            "rho": 1000.0,
            "c": 1480.0,
            "material_model": "fluid",
            "absorb_depth": 0.0005,
        },
        "top_halfspace": {
            "thickness": 0.001,
            "rho": 1000.0,
            "c": 1480.0,
            "material_model": "fluid",
            "absorb_depth": 0.0005,
        },
    }
)
