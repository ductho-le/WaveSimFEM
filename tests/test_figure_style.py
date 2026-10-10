"""Physical export dimensions and typography shared by every figure family."""

import re
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
import pytest
from matplotlib.texmanager import TexManager
from matplotlib.text import Text
from PIL import Image

from wavesimfem import analysis


@pytest.mark.parametrize("usetex", [False, True])
def test_all_export_formats_preserve_190_mm_canvas(tmp_path, usetex):
    if usetex and not analysis.latex_available():
        pytest.skip("LaTeX, dvipng and Ghostscript are required for external TeX export")
    with matplotlib.rc_context(analysis._rc(usetex)):
        fig = analysis._figure(100, layout="constrained")
        ax = fig.subplots()
        image = ax.imshow(np.arange(9).reshape(3, 3))
        colorbar = fig.colorbar(image, ax=ax, label="Power [dB]")
        ax.plot([0, 2], [0, 2], label="50% reference_gain")
        ax.set(title="(a) Measurement", xlabel=r"Time [$\mu$s]", ylabel="Position [mm]")
        ax.title.set_fontsize(18)
        ax.legend(fontsize=7)

    # Saving after leaving the figure's style context must retain its mode,
    # regardless of the user's global text or tight-cropping preferences.
    with matplotlib.rc_context({"savefig.bbox": "tight", "text.usetex": not usetex}):
        analysis._save(fig, str(tmp_path / "figure"), analysis.FIGURE_FORMATS, 300)
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert colorbar.extend == "neither"
    for artist in fig.findobj(match=Text):
        if artist.get_text():
            assert artist.get_fontsize() == 10
            assert artist.get_usetex() is usetex

    pdf = (tmp_path / "figure.pdf").read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    assert box is not None
    assert float(box[1]) * 25.4 / 72 == pytest.approx(190, abs=1e-6)
    assert float(box[2]) * 25.4 / 72 == pytest.approx(100, abs=1e-6)
    svg = ET.parse(tmp_path / "figure.svg").getroot()
    assert float(svg.attrib["width"].removesuffix("pt")) * 25.4 / 72 == pytest.approx(190, abs=1e-5)
    eps = (tmp_path / "figure.eps").read_text(errors="replace")
    bounds = re.search(r"%%HiResBoundingBox: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", eps)
    assert bounds is not None
    assert (float(bounds[3]) - float(bounds[1])) * 25.4 / 72 == pytest.approx(190, abs=0.001)
    with Image.open(tmp_path / "figure.png") as raster:
        assert raster.width == int(190 / 25.4 * 300)
        assert raster.width / raster.info["dpi"][0] * 25.4 == pytest.approx(190, abs=25.4 / 300)


def test_disabled_latex_never_checks_external_tools(monkeypatch):
    def unexpected_check():
        pytest.fail("Disabled LaTeX must not inspect the external toolchain")

    monkeypatch.setattr(analysis, "latex_available", unexpected_check)
    assert analysis.resolve_usetex() is False
    assert analysis.resolve_usetex(False) is False
    assert analysis.resolve_usetex(np.bool_(False)) is False


@pytest.mark.parametrize("available", [False, True])
def test_auto_latex_uses_toolchain_availability(monkeypatch, available):
    monkeypatch.setattr(analysis, "latex_available", lambda: available)
    assert analysis.resolve_usetex("auto") is available


def test_enabled_latex_requires_external_tools(monkeypatch):
    monkeypatch.setattr(analysis, "latex_available", lambda: False)
    with pytest.raises(RuntimeError, match="LaTeX"):
        analysis.resolve_usetex(True)
    monkeypatch.setattr(analysis, "latex_available", lambda: True)
    assert analysis.resolve_usetex(True) is True


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true", "invalid"])
def test_invalid_latex_toggle_is_rejected(invalid):
    with pytest.raises(ValueError, match="usetex"):
        analysis.resolve_usetex(invalid)


def test_default_export_does_not_use_external_tex(tmp_path, monkeypatch, caplog):
    def unexpected_tex(*args, **kwargs):
        pytest.fail("Default figure exports must not invoke or require external TeX")

    monkeypatch.setattr(analysis, "latex_available", unexpected_tex)
    monkeypatch.setattr(TexManager, "_run_checked_subprocess", unexpected_tex)
    label = r"50% gain_a & b #1, $c_f$ and $\alpha$"
    with matplotlib.rc_context(analysis._rc()):
        fig = analysis._figure(100, layout="constrained")
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1], label=label)
        ax.set(title="(a) Measurement", xlabel=r"Time [$\mu$s]", ylabel="Position [mm]")
        legend = ax.legend()
        analysis._save(fig, str(tmp_path / "portable"), ["png", "pdf"], 72)
    assert legend.get_texts()[0].get_text() == label
    assert ax.get_title() == "(a) Measurement"
    assert all(not artist.get_usetex() for artist in fig.findobj(match=Text))
    assert (tmp_path / "portable.png").stat().st_size > 0
    assert (tmp_path / "portable.pdf").stat().st_size > 0
    assert "findfont:" not in caplog.text


def test_literal_tex_characters_preserve_math():
    text = analysis._latex_text(r"50% gain_a & b #1, $c_f$ and $\alpha$")
    assert text == r"50\% gain\_a \& b \#1, $c_f$ and $\alpha$"
    assert analysis._latex_text(text) == text
