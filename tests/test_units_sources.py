"""Scale-aware units and excitation waveforms."""

from __future__ import annotations

import numpy as np
import pytest

from wavesimfem import sources, units

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "picker,value,label",
    [
        (units.time_unit, 2.0, "s"),
        (units.time_unit, 0.5, "ms"),
        (units.time_unit, 120e-6, "μs"),
        (units.time_unit, 3e-10, "ns"),
        (units.length_unit, 150.0, "m"),
        (units.length_unit, 2500.0, "km"),
        (units.length_unit, 0.06, "mm"),
        (units.length_unit, 5e-7, "μm"),
        (units.frequency_unit, 15.0, "Hz"),
        (units.frequency_unit, 250e3, "kHz"),
        (units.frequency_unit, 2e6, "MHz"),
        (units.displacement_unit, 3e-11, "pm"),
        (units.displacement_unit, 2e-9, "nm"),
    ],
)
def test_unit_picking(picker, value, label):
    assert picker(value).label == label


@pytest.mark.parametrize("value", [0.0, np.inf, np.nan])
def test_degenerate_magnitudes_fall_back_to_si(value):
    assert units.time_unit(value).size == 1.0
    assert units.frequency_unit(value).label == "Hz"


def test_unit_conversion_and_wavenumber_pairing():
    mm = units.length_unit(0.01)
    k = units.wavenumber_unit(mm)
    assert k.label == "rad/mm" and k.size == pytest.approx(1e3)
    assert k(3000.0) == pytest.approx(3.0)
    np.testing.assert_allclose(mm(np.array([0.001, 0.002])), [1.0, 2.0])


def test_formatters():
    assert units.fmt_time(1.2e-4) == "120 μs"
    assert units.fmt_length(0.002) == "2 mm"
    assert units.fmt_frequency(5e5) == "500 kHz"
    assert units.fmt_frequency(15.0) == "15 Hz"


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        (2e6, "frequency", 2e6),
        (15, "frequency", 15.0),
        ("2 MHz", "frequency", 2e6),
        ("250kHz", "frequency", 2.5e5),
        ("40 hz", "frequency", 40.0),
        ("1.5e3", "frequency", 1.5e3),
        ("3 rad/mm", "wavenumber", 3e3),
        ("1.2 rad/m", "wavenumber", 1.2),
        ("0.5 rad/cm", "wavenumber", 50.0),
        (np.float32(4.0), "wavenumber", 4.0),
    ],
)
def test_parse_quantity(value, kind, expected):
    assert units.parse_quantity(value, kind) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,kind,match",
    [
        (True, "frequency", "boolean"),
        ("fast", "frequency", "cannot parse"),
        ("2 MHz extra", "frequency", "cannot parse"),
        ("3 rad/mm", "frequency", "unknown frequency unit"),
        ("2 MHz", "wavenumber", "unknown wavenumber unit"),
        (float("inf"), "frequency", "finite"),
        ([1.0], "frequency", "number or string"),
    ],
)
def test_parse_quantity_errors(value, kind, match):
    with pytest.raises(ValueError, match=match):
        units.parse_quantity(value, kind)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
F0 = 250e3


@pytest.mark.parametrize("n_cycles", [1, 3, 5, 10])
def test_tone_burst_window_and_peak(n_cycles):
    t_end = n_cycles / F0
    t = np.linspace(-0.5 * t_end, 1.5 * t_end, 20001)
    s = sources.tone_burst(t, F0, n_cycles)
    assert np.all(s[(t < 0) | (t > t_end)] == 0.0)
    assert sources.tone_burst(0.0, F0, n_cycles) == 0.0
    assert abs(sources.tone_burst(t_end, F0, n_cycles)) < 1e-12
    assert 0.5 < np.max(np.abs(s)) <= 1.0
    if n_cycles >= 5:  # sine crest lands near the window centre
        assert np.max(np.abs(s)) == pytest.approx(1.0, abs=0.05)


def test_tone_burst_spectrum_is_centred_on_f0():
    fs, n = 50e6, 1 << 16
    t = np.arange(n) / fs
    spectrum = np.abs(np.fft.rfft(sources.tone_burst(t, F0, 5)))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    assert freqs[np.argmax(spectrum)] == pytest.approx(F0, rel=0.02)


@pytest.mark.parametrize("n_cycles", [0, -1, True, np.bool_(True), 1.5, np.nan, np.inf, "3"])
def test_tone_burst_rejects_invalid_cycles(n_cycles):
    with pytest.raises(ValueError, match="n_cycles"):
        sources.tone_burst(0.0, F0, n_cycles)


def test_scalar_input_returns_float():
    assert isinstance(sources.tone_burst(1e-6, F0, 5), float)
    assert isinstance(sources.ricker(1e-6, F0, 0.0), float)


def test_source_dispatch_delay_and_duration():
    t = np.linspace(0, 40e-6, 101)
    np.testing.assert_allclose(
        sources.source_signal(t, "ricker", F0), sources.ricker(t, F0, sources.ricker_delay(F0))
    )
    np.testing.assert_allclose(
        sources.source_signal(t, "tone_burst", F0, 3), sources.tone_burst(t, F0, 3)
    )
    assert sources.source_delay("ricker", F0) == pytest.approx(1.5 / F0)
    assert sources.source_delay("tone_burst", F0, 4) == pytest.approx(2.0 / F0)
    assert sources.source_duration("tone_burst", F0, 4) == pytest.approx(4.0 / F0)
    assert sources.source_duration("tone_burst", F0) == pytest.approx(sources.DEFAULT_N_CYCLES / F0)
    assert sources.describe_source("tone_burst", 3) == "3-cycle Hann tone burst"
    assert sources.describe_source("ricker") == "Ricker wavelet"


@pytest.mark.parametrize(
    "func", [sources.source_signal, sources.source_delay, sources.describe_source]
)
def test_unknown_source_type(func):
    args = (0.0, "chirp", F0) if func is sources.source_signal else ("chirp",)
    if func is sources.source_delay:
        args = ("chirp", F0)
    with pytest.raises(ValueError, match="Unknown source_type"):
        func(*args)
