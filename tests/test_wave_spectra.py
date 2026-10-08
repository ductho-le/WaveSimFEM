"""Analytical-wave checks for measured dispersion and polarization diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from guidedwavefem.wave_spectra import phase_velocity_spectrum, polarization_spectrum


def _wave(*, direction=1, speed=2000.0, frequency=125_000.0):
    times = np.arange(1024) * 1e-7
    positions = np.arange(64) * 0.001
    phase = 2 * np.pi * frequency * (times[:, None] - direction * positions[None, :] / speed)
    return times, positions, np.cos(phase), phase


def _measure(times, positions, values, **kwargs):
    return phase_velocity_spectrum(times, positions, values, np.linspace(500, 5000, 451), **kwargs)


def test_known_phase_speed_and_physical_sampling_limits():
    times, positions, values, _ = _wave()
    result = _measure(times, positions, values)
    dominant = np.argmax(result["relative_power_db"])
    # Input frequency is between physical FFT bins; phase speed at that bin
    # scales with f/f0 because a monochromatic field carries only one k.
    expected = 2000 * result["frequencies"][dominant] / 125_000
    assert result["ridge_valid"][dominant]
    assert result["ridge_velocity"][dominant] == pytest.approx(expected, abs=10)
    assert result["peak_coherence"][dominant] > 0.995
    assert result["diagnostics"]["frequency_bin_hz"] == pytest.approx(1 / (1024e-7))
    np.testing.assert_allclose(result["alias_min_velocity"], 2 * result["frequencies"] * 0.001)
    np.testing.assert_allclose(result["aperture_max_velocity"], result["frequencies"] * 0.063)
    assert np.max(result["coherence"]) <= 1
    json.dumps(result["diagnostics"], allow_nan=False)


@pytest.mark.parametrize("scale", [1e-20, -3.5, 1e15])
def test_velocity_image_invariant_to_displacement_amplitude(scale):
    times, positions, values, _ = _wave()
    reference = _measure(times, positions, values)
    scaled = _measure(times, positions, values * scale)
    np.testing.assert_allclose(scaled["coherence"], reference["coherence"], atol=1e-6)
    np.testing.assert_allclose(
        scaled["ridge_velocity"], reference["ridge_velocity"], equal_nan=True
    )


def test_reversed_wave_and_coordinate_origin():
    times, positions, values, _ = _wave(direction=-1)
    reverse = _measure(times, positions + 42.0, values, direction=-1)
    wrong = _measure(times, positions, values, direction=1)
    dominant = np.argmax(reverse["relative_power_db"])
    assert reverse["ridge_valid"][dominant]
    assert reverse["peak_coherence"][dominant] > 0.99
    assert not wrong["ridge_valid"][dominant]


def test_two_resolved_modes_do_not_produce_confident_single_branch():
    times, positions, first, _ = _wave(speed=1100)
    _, _, second, _ = _wave(speed=2300)
    result = _measure(times, positions, first + second)
    dominant = np.argmax(result["relative_power_db"])
    assert not result["ridge_valid"][dominant]
    assert result["ridge_reason"][dominant] == "competing velocity peaks"
    assert np.isnan(result["ridge_velocity"][dominant])


def test_too_short_aperture_rejects_apparently_coherent_speed():
    times, _, _, _ = _wave()
    positions = np.arange(16) * 0.001
    values = np.cos(2 * np.pi * 125_000 * (times[:, None] - positions[None, :] / 1800))
    result = _measure(times, positions, values)
    dominant = np.argmax(result["relative_power_db"])
    assert not result["ridge_valid"][dominant]
    assert result["ridge_reason"][dominant] in (
        "fewer than two wavelengths across effective aperture",
        "peak at scan or sampling boundary",
    )


def test_frequency_reduction_preserves_narrow_carrier_and_band():
    times, positions, values, _ = _wave(frequency=781_250)
    result = _measure(times, positions, values, max_frequencies=12, frequency_band=(20_000, 2e6))
    assert len(result["frequencies"]) == 12
    assert result["frequencies"].min() >= 20_000
    assert result["frequencies"].max() <= 2e6
    assert result["frequencies"][np.argmax(result["relative_power_db"])] == pytest.approx(781_250)
    assert result["coherence"].shape == result["support_mask"].shape == (12, 451)


def test_one_dominant_receiver_cannot_qualify_spatial_measurement():
    times, positions, values, _ = _wave()
    values[:, 1:] *= 1e-8
    result = _measure(times, positions, values)
    assert not result["ridge_valid"].any()
    assert not result["support_mask"].any()


def test_nonuniform_array_uses_conservative_maximum_gap():
    times, positions, _, _ = _wave()
    positions[20:] += 0.0007
    values = np.cos(2 * np.pi * 125_000 * (times[:, None] - positions[None, :] / 2000))
    result = _measure(times, positions, values)
    dominant = np.argmax(result["relative_power_db"])
    assert result["ridge_valid"][dominant]
    assert result["diagnostics"]["maximum_receiver_gap_m"] == pytest.approx(0.0017)


def test_spectrally_flat_coherent_wave_is_not_mistaken_for_noise():
    times = np.arange(512) * 1e-6
    positions = np.arange(64) * 0.001
    values = np.zeros((len(times), len(positions)))
    values[64 + np.arange(64), np.arange(64)] = 1
    result = _measure(times, positions, values)
    row = np.argmin(abs(result["frequencies"] - 125_000))
    assert result["spectral_contrast_db"][row] < 6
    assert result["ridge_valid"][row]
    assert result["ridge_velocity"][row] == pytest.approx(1000)
    polarization = polarization_spectrum(times, values, 2 * values)
    assert polarization["valid"].any()
    np.testing.assert_allclose(polarization["vertical_horizontal_ratio"][polarization["valid"]], 2)


@pytest.mark.parametrize("constant", [0.0, 42.0])
def test_constant_signal_returns_explicit_no_measurement(constant):
    times, positions, values, _ = _wave()
    values[:] = constant
    result = _measure(times, positions, values)
    assert not result["ridge_valid"].any()
    assert not result["support_mask"].any()
    assert not result["coherence"].any()
    assert np.isfinite(result["relative_power_db"]).all()
    polarization = polarization_spectrum(times, values, values)
    assert not polarization["valid"].any()
    assert np.isnan(polarization["vertical_horizontal_ratio"]).all()
    json.dumps(polarization["diagnostics"], allow_nan=False)


def test_polarization_elliptic_ratio_cross_phase_and_amplitude_invariance():
    times, _, horizontal, phase = _wave()
    vertical = 2.5 * np.sin(phase)
    result = polarization_spectrum(times, horizontal, vertical)
    dominant = np.argmax(result["relative_power_db"])
    assert result["valid"][dominant]
    assert result["vertical_horizontal_ratio"][dominant] == pytest.approx(2.5, rel=0.002)
    assert result["cross_phase_deg"][dominant] == pytest.approx(-90, abs=0.1)
    assert result["coherence"][dominant] > 0.999
    scaled = polarization_spectrum(times, horizontal * 1e-12, vertical * 1e-12)
    np.testing.assert_allclose(
        result["vertical_horizontal_ratio"], scaled["vertical_horizontal_ratio"], equal_nan=True
    )
    assert result["diagnostics"]["segment_count"] >= 3
    json.dumps(result["diagnostics"], allow_nan=False)


@pytest.mark.parametrize("component", ["x", "y"])
def test_pure_component_and_negligible_cross_component_suppress_polarization(component):
    times, _, values, _ = _wave()
    for fraction in (0, 1e-8):
        ux, uy = (values, values * fraction) if component == "x" else (values * fraction, values)
        result = polarization_spectrum(times, ux, uy)
        assert not result["valid"].any()
        assert np.isnan(result["vertical_horizontal_ratio"]).all()
        assert np.isnan(result["cross_phase_deg"]).all()


def test_incoherent_components_do_not_create_ellipticity():
    times, _, horizontal, phase = _wave()
    rng = np.random.default_rng(71)
    vertical = np.cos(phase + rng.uniform(-np.pi, np.pi, horizontal.shape[1]))
    result = polarization_spectrum(times, horizontal, vertical)
    dominant = np.argmax(result["relative_power_db"])
    assert result["coherence"][dominant] < 0.1
    assert not result["valid"][dominant]


@pytest.mark.parametrize(
    "problem", ["short", "nan", "time", "shape", "positions", "velocities", "direction", "band"]
)
def test_invalid_inputs_fail_clearly(problem):
    times, positions, values, _ = _wave()
    velocities = np.linspace(500, 5000, 100)
    options = {}
    if problem == "short":
        times, values = times[:16], values[:16]
    elif problem == "nan":
        values[0, 0] = np.nan
    elif problem == "time":
        times[1] *= 1.5
    elif problem == "shape":
        values = values.T
    elif problem == "positions":
        positions = positions[::-1]
    elif problem == "velocities":
        velocities[0] = 0
    elif problem == "direction":
        options["direction"] = 0
    elif problem == "band":
        options["frequency_band"] = (1e8, 1e9)
    with pytest.raises(ValueError):
        phase_velocity_spectrum(times, positions, values, velocities, **options)


def test_polarization_requires_matching_shapes_and_valid_time():
    times, _, values, _ = _wave()
    with pytest.raises(ValueError, match="matching shapes"):
        polarization_spectrum(times, values, values[:, :-1])
    with pytest.raises(ValueError, match="32 time"):
        polarization_spectrum(times[:8], values[:8], values[:8])
