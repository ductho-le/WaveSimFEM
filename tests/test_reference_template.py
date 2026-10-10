"""Keep the full reference template usable and aligned with the public schema."""

from pathlib import Path

from wavesimfem.config import (
    SECTION_KEYS,
    generate_template_config,
    load_config,
    parse_and_resolve_config,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "template.yaml"


def test_reference_template_covers_active_parameters():
    raw = load_config(str(TEMPLATE))
    # Mutually exclusive structure blocks and legacy plot aliases are documented
    # as commented alternatives rather than activated in the runnable plate case.
    alternatives = {
        "structure": {
            "layers",
            "halfspace",
            "substrate",
            "superstrate",
            "bottom_halfspace",
            "top_halfspace",
        },
        "postprocess": {"f_min_disp_MHz", "f_max_disp_MHz", "k_min_disp", "k_max_disp"},
    }
    text = TEMPLATE.read_text(encoding="utf-8")
    for section, keys in SECTION_KEYS.items():
        omitted = alternatives.get(section, set())
        assert set(raw[section]) == set(keys) - omitted
        for key in omitted:
            assert key in text


def test_reference_template_resolves():
    cfg = parse_and_resolve_config(str(TEMPLATE))
    assert cfg["nx"] == 640
    assert cfg["ny_per_region"] == [6]
    assert cfg["CFL"] == 0.8
    assert cfg["usetex"] is False


def test_generated_template_defaults_to_builtin_text(tmp_path):
    path = tmp_path / "config.yaml"
    generate_template_config(str(path))
    assert load_config(str(path))["output"]["usetex"] is False
