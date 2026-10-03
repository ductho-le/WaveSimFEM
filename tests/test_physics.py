"""Physics verification against closed-form wave speeds."""

from __future__ import annotations

import numpy as np
import pytest

from guidedwavefem import run_simulation

E, NU, RHO, H = 70e9, 0.33, 2700.0, 1e-3


def _arrival_lag(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Time lag of ``b`` relative to ``a`` from the cross-correlation peak
    (parabolic sub-sample refinement)."""
    xc = np.correlate(b, a, mode="full")
    i = int(np.argmax(xc))
    y0, y1, y2 = xc[i - 1], xc[i], xc[i + 1]
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (i - (len(a) - 1) + frac) * dt


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_s0_plate_velocity_low_frequency(tmp_path, solver):
    """The fundamental symmetric Lamb mode at low frequency-thickness travels at
    the plate velocity c_p = sqrt(E / (rho (1 - nu^2))) (≈ 5.39 km/s for Al).

    An in-plane surface load excites S0 and A0; sampling u_1 on the mid-plane
    removes A0 (its u_1 is antisymmetric through the thickness).
    """
    c_plate = np.sqrt(E / (RHO * (1.0 - NU**2)))
    f0 = 100e3  # f·d = 0.1 MHz·mm: S0 is essentially non-dispersive
    pitch, n_sensors = 2e-3, 31
    path, _ = run_simulation(
        cfg={
            "solver": {"type": solver, "duration": 50e-6},
            "geometry": {"sensing_length": 0.12, "absorb_length": 0.05},
            "structure": {
                "mode": "1_layer",
                "single_layer": {"thickness": H, "rho": RHO, "E": E, "nu": NU},
            },
            "excitation": {"f_center": f0, "source_fx": 1.0, "source_fy": 0.0},
            "mesh": {"elems_per_wavelength": 8, "min_elems_y": 2, "degree": 2},
            "sensors": {
                "n_sensors": n_sensors,
                "sensor_offset": 0.02,
                "sensor_pitch": pitch,
                "sensor_y": H / 2,
            },
            "postprocess": {"run_postprocess": False},
            "output": {"output_dir": str(tmp_path)},
        }
    )
    with np.load(path) as data:
        times, ux = data["times"], data["sensor_data_x"]
    lag = _arrival_lag(ux[:, 0], ux[:, -1], times[1] - times[0])
    c_measured = (n_sensors - 1) * pitch / lag
    assert c_measured == pytest.approx(c_plate, rel=0.01), (c_measured, c_plate)
