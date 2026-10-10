"""Post-processing figure tests: portable labels, scoped styling, output formats."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

from wavesimfem import analysis


def _archive(path, n_t=256, n_x=16):
    t = np.arange(n_t) * 1e-7
    x = np.arange(n_x) * 0.001
    sig = np.sin(2 * np.pi * 5e5 * t[:, None] - 1000 * x[None, :])
    np.savez(
        path,
        times=t,
        sensor_x=x,
        sensor_data_y=sig,
        sensor_data_x=sig,
        f_center=5e5,
        c_L=6000.0,
        c_T=3000.0,
    )
    return str(path)


def _capture_titles(monkeypatch):
    titles: list[str] = []
    usetex_flags: list[bool] = []

    def fake_save(fig, *args, **kwargs):
        usetex_flags.append(bool(matplotlib.rcParams["text.usetex"]))
        titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_save)
    return titles, usetex_flags


def test_full_analysis_has_equal_square_panels(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def check_geometry(fig, *args, **kwargs):
        FigureCanvasAgg(fig).draw()
        assert fig.get_figwidth() * 25.4 == pytest.approx(190)
        panels = [ax for ax in fig.axes if ax.get_title()]
        assert len(panels) == 4
        sizes = []
        for ax in panels:
            bounds = ax.get_window_extent()
            assert bounds.width == pytest.approx(bounds.height)
            sizes.append(bounds.width)
        assert sizes == pytest.approx([sizes[0]] * 4)
        return [str(tmp_path / "full_analysis.png")]

    monkeypatch.setattr(analysis, "_save", check_geometry)
    analysis.plot_full_analysis(_archive(tmp_path / "square.npz"), str(tmp_path))


def test_relative_power_is_amplitude_invariant(tmp_path, monkeypatch):
    path = _archive(tmp_path / "original.npz")
    with np.load(path) as archive:
        data = dict(archive)
    images = []

    def capture(fig, *args, **kwargs):
        images.append(
            [np.asarray(ax.images[0].get_array()).copy() for ax in fig.axes if ax.images][-1]
        )

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture)
    for scale in (1.0, 1e-20, 0.0):
        np.savez(
            tmp_path / "scaled.npz",
            **dict(
                data,
                sensor_data_x=data["sensor_data_x"] * scale,
                sensor_data_y=data["sensor_data_y"] * scale,
            ),
        )
        analysis.plot_full_analysis(
            str(tmp_path / "scaled.npz"), str(tmp_path), usetex=False, formats=["png"]
        )
    np.testing.assert_allclose(images[0], images[1], atol=1e-5)
    assert np.max(images[1]) == pytest.approx(0.0)
    assert np.isfinite(images[2]).all() and np.max(images[2]) < -40


def test_left_receiver_arrival_guides(tmp_path, monkeypatch):
    path = _archive(tmp_path / "left.npz")
    with np.load(path) as archive:
        data = dict(archive)
    data["sensor_offset"] = -0.011
    np.savez(path, **data)
    arrivals = []

    def capture(fig, *args, **kwargs):
        arrivals.extend(
            line.get_xdata()[0] for line in fig.axes[2].lines if "c_L" in line.get_label()
        )

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture)
    analysis.plot_full_analysis(path, str(tmp_path), usetex=False, formats=["png"])
    # Ricker center 3 us + 11 mm / 6000 m/s.
    assert arrivals == pytest.approx([4.833333333333])


def test_concurrent_plotting_restores_rc(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    path = _archive(tmp_path / "threaded.npz")
    before = dict(matplotlib.rcParams)
    entered, release, second_started = Event(), Event(), Event()
    flags = []
    monkeypatch.setattr(analysis, "latex_available", lambda: True)

    def capture(fig, *args, **kwargs):
        flags.append(matplotlib.rcParams["text.usetex"])
        if len(flags) == 1:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture)

    def plot(tex):
        if not tex:
            second_started.set()
        return analysis.plot_full_analysis(path, str(tmp_path), usetex=tex, formats=["png"])

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(plot, True)
        assert entered.wait(10)
        second = pool.submit(plot, False)
        assert second_started.wait(10)
        release.set()
        first.result()
        second.result()
    assert flags == [True, False]
    assert dict(matplotlib.rcParams) == before


def test_default_plotting_uses_bundled_fonts(tmp_path, monkeypatch):
    def unexpected_check():
        pytest.fail("Default plotting must not require an external TeX installation")

    monkeypatch.setattr(analysis, "latex_available", unexpected_check)
    titles, usetex_flags = _capture_titles(monkeypatch)
    analysis.plot_full_analysis(_archive(tmp_path / "s.npz"), str(tmp_path))
    assert titles and usetex_flags == [False] * len(usetex_flags)
    assert all(not t.startswith("\\textbf{") for t in titles)
    assert any("Dispersion ($u_1$)" in t for t in titles)


def test_usetex_titles_are_bold(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "latex_available", lambda: True)
    titles, usetex_flags = _capture_titles(monkeypatch)
    analysis.plot_full_analysis(
        _archive(tmp_path / "s.npz"), str(tmp_path), usetex=True, formats=["png"]
    )
    assert usetex_flags == [True]
    assert all(t.startswith("\\textbf{") for t in titles)


def test_missing_latex_is_explicit_and_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "latex_available", lambda: False)
    with pytest.raises(RuntimeError, match="LaTeX"):
        analysis.plot_full_analysis(
            _archive(tmp_path / "s.npz"), str(tmp_path), usetex=True, formats=["png"]
        )
    assert not list(tmp_path.glob("full_analysis.*"))


def test_global_state_is_not_modified(tmp_path):
    rc_before = dict(matplotlib.rcParams)
    path_before = os.environ.get("PATH")
    analysis.plot_full_analysis(
        _archive(tmp_path / "s.npz"), str(tmp_path), usetex=False, formats=["png"], dpi=50
    )
    assert os.environ.get("PATH") == path_before
    changed = {k for k, v in matplotlib.rcParams.items() if rc_before.get(k) != v}
    assert not changed


def test_formats_parameter(tmp_path):
    analysis.plot_full_analysis(
        _archive(tmp_path / "s.npz"), str(tmp_path), usetex=False, formats=["png", "svg"], dpi=50
    )
    assert sorted(p.name for p in tmp_path.glob("full_analysis.*")) == [
        "full_analysis.png",
        "full_analysis.svg",
    ]
    with pytest.raises(ValueError, match="Unsupported figure format"):
        analysis.plot_full_analysis(
            _archive(tmp_path / "t.npz"), str(tmp_path), usetex=False, formats=["bmp"]
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(f_min=2e6, f_max=1e6), "Dispersion limits"),
        (dict(f_min="2 MHz", f_max="1 MHz"), "Dispersion limits"),
        (dict(k_min=-1.0), "Dispersion limits"),
        (dict(f_max="2 rad/mm"), "frequency"),
    ],
)
def test_invalid_limits(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        analysis.plot_full_analysis(_archive(tmp_path / "s.npz"), str(tmp_path), **kwargs)


def _capture_axes(monkeypatch):
    captured: dict[str, list[str]] = {"xlabels": [], "ylabels": [], "ylim": []}

    def fake_save(fig, *args, **kwargs):
        for ax in fig.axes:
            captured["xlabels"].append(ax.get_xlabel())
            captured["ylabels"].append(ax.get_ylabel())
            if "Dispersion" in ax.get_title():
                captured["ylim"].append(ax.get_ylim())

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_save)
    return captured


def test_unit_string_limits_are_honoured(tmp_path, monkeypatch):
    captured = _capture_axes(monkeypatch)
    analysis.plot_full_analysis(
        _archive(tmp_path / "s.npz"), str(tmp_path), f_max="800 kHz", usetex=False, formats=["png"]
    )
    assert "Frequency [kHz]" in captured["ylabels"]
    assert captured["ylim"][0][1] == pytest.approx(800.0)


def test_legacy_archive_limits_are_honoured(tmp_path, monkeypatch):
    path = tmp_path / "s.npz"
    _archive(path)
    with np.load(path) as archive:
        data = dict(archive)
    np.savez(path, **data, f_min_disp_MHz=0.0, f_max_disp_MHz=0.8, k_min_disp=0.0, k_max_disp=2.0)
    captured = _capture_axes(monkeypatch)
    analysis.plot_full_analysis(str(path), str(tmp_path), usetex=False, formats=["png"])
    assert captured["ylim"][0][1] == pytest.approx(800.0)  # kHz
    assert "Wavenumber [rad/mm]" in captured["xlabels"]


def test_seismic_scale_archive_uses_natural_units(tmp_path, monkeypatch):
    """A 15 Hz record over ~100 m is labelled in s, m, Hz and rad/m."""
    t = np.arange(1024) * 1e-3
    x = 10.0 + np.arange(48) * 2.0
    sig = 1e-6 * np.sin(2 * np.pi * 15.0 * t[:, None] - 0.4 * x[None, :])
    path = tmp_path / "seis.npz"
    np.savez(
        path,
        times=t,
        sensor_x=x,
        sensor_data_y=sig,
        sensor_data_x=sig,
        f_center=15.0,
        c_L=400.0,
        c_T=200.0,
        source_x=0.0,
        source_type="tone_burst",
        n_cycles=3,
    )
    captured = _capture_axes(monkeypatch)
    analysis.plot_full_analysis(str(path), str(tmp_path), usetex=False, formats=["png"])
    assert "Time [s]" in captured["xlabels"]
    assert "Offset from source [m]" in captured["ylabels"]
    assert "Frequency [Hz]" in captured["ylabels"]
    assert "Wavenumber [rad/m]" in captured["xlabels"]
    assert captured["ylim"][0][1] == pytest.approx(30.0)  # automatic 2 f0


def test_short_record_rejected(tmp_path):
    with pytest.raises(ValueError, match="32 time samples"):
        analysis.plot_full_analysis(_archive(tmp_path / "s.npz", n_t=16), str(tmp_path))


def test_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        analysis.plot_full_analysis(str(tmp_path / "nope.npz"))


def test_bandpass_filter_passes_centre_and_rejects_dc():
    fs, f0 = 10e6, 5e5
    t = np.arange(4096) / fs
    signal = np.sin(2 * np.pi * f0 * t)[:, None] + 1.0  # tone + DC offset
    filtered, band = analysis.bandpass_filter(signal, fs, f0)
    assert band is not None and band[0] < f0 < band[1]
    core = filtered[512:-504, 0]  # 3080 samples = 154 whole periods
    assert abs(core.mean()) < 1e-3
    assert np.std(core) == pytest.approx(np.sqrt(0.5), rel=0.02)


def test_bandpass_filter_short_record_is_passthrough():
    sig = np.ones((10, 2))
    out, band = analysis.bandpass_filter(sig, 10e6, 5e5)
    assert band is None and out is sig


def test_quick_overview(tmp_path):
    t = np.arange(200) * 1e-7
    x = 0.01 + np.arange(8) * 1e-3
    data = np.sin(2 * np.pi * 5e5 * t[:, None] - 500 * x[None, :])
    out = analysis.plot_quick_overview(t, x, data, 5e5, 0.005, str(tmp_path / "quick.png"))
    assert os.path.getsize(out) > 0


@pytest.mark.parametrize("usetex", [False, True])
def test_quick_overview_respects_latex_toggle(tmp_path, monkeypatch, usetex):
    monkeypatch.setattr(analysis, "latex_available", lambda: True)
    titles, usetex_flags = _capture_titles(monkeypatch)
    t = np.arange(200) * 1e-7
    x = 0.01 + np.arange(8) * 1e-3
    data = np.sin(2 * np.pi * 5e5 * t[:, None] - 500 * x[None, :])
    analysis.plot_quick_overview(t, x, data, 5e5, 0.005, str(tmp_path / "quick.png"), usetex=usetex)
    assert usetex_flags == [usetex]
    assert titles and all(title.startswith(r"\textbf{") == usetex for title in titles)


def test_quick_overview_tone_burst(tmp_path):
    t = np.arange(400) * 2e-7
    x = 0.01 + np.arange(8) * 1e-3
    data = np.sin(2 * np.pi * 3e5 * t[:, None] - 500 * x[None, :])
    out = analysis.plot_quick_overview(
        t,
        x,
        data,
        3e5,
        0.005,
        str(tmp_path / "quick.png"),
        source_type="tone_burst",
        n_cycles=5,
    )
    assert os.path.getsize(out) > 0


def _legend_texts(monkeypatch):
    texts: list[str] = []

    def fake_save(fig, *args, **kwargs):
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                texts.extend(t.get_text() for t in legend.get_texts())

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_save)
    return texts


@pytest.mark.parametrize(
    "c_l,c_t,fluid",
    [
        (120.0, 40.0, False),  # very soft soil: slow, but a solid (c_T/c_L = 1/3)
        (1500.0, 1.5, True),  # regularized fluid: c_T/c_L = 1e-3
    ],
)
def test_legacy_fluid_detection_is_relative(tmp_path, monkeypatch, c_l, c_t, fluid):
    """Archives without ``is_fluid_top`` classify the top region by c_T/c_L, not by
    an absolute speed, so slow solids keep their c_L/c_T guide lines."""
    t = np.arange(512) * 1e-3
    x = 2.0 + np.arange(32) * 0.5
    sig = 1e-6 * np.sin(2 * np.pi * 15.0 * t[:, None] - 2.0 * x[None, :])
    path = tmp_path / "soft.npz"
    np.savez(
        path, times=t, sensor_x=x, sensor_data_y=sig, sensor_data_x=sig, f_center=15.0,
        c_L=c_l, c_T=c_t, source_x=0.0,
    )  # fmt: skip
    texts = _legend_texts(monkeypatch)
    analysis.plot_full_analysis(str(path), str(tmp_path), usetex=False, formats=["png"])
    joined = " ".join(texts)
    assert ("fluid" in joined) is fluid
    assert ("c_T" in joined) is not fluid
