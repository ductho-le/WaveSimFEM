"""Small standalone regression driver, invoked by the subprocess tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_waves import run_simulation

run_simulation(cfg={
    'solver_type': sys.argv[2], 'output_dir': sys.argv[1],
    'sensing_length': .01, 'absorb_length': .003, 'source_x': .004,
    'f_center': 5e5, 'sensor_offset': .00113, 'sensor_pitch': .00107,
    'n_sensors': 4, 'duration': 3e-6, 'max_frequency': 5e5,
    'min_elems_x': 10, 'min_elems_y': 2, 'elems_per_wavelength': 2,
    'run_postprocess': False, 'structure_mode': 'plate_in_halfspaces',
    'single_layer': {'thickness': .001, 'rho': 2700., 'E': 70e9,
                     'nu': .33, 'material_model': 'isotropic'},
    'bottom_halfspace': {'thickness': .001, 'rho': 1000., 'c': 1480.,
                         'material_model': 'fluid', 'absorb_depth': .0005},
    'top_halfspace': {'thickness': .001, 'rho': 1000., 'c': 1480.,
                      'material_model': 'fluid', 'absorb_depth': .0005},
})
