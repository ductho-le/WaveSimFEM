"""Tests for the junk cleaner (tools/clean_junk.py) and the ``forbid-junk`` pre-commit guard."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("clean_junk", REPO / "tools" / "clean_junk.py")
clean_junk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean_junk)

JUNK = [
    "__pycache__/x.cpython-311.pyc",
    "src/guidedwavefem/__pycache__/cli.cpython-311.pyc",
    "stray.pyc",
    ".pytest_cache/v/cache/nodeids",
    "tests/.ruff_cache/0.16.9/abc",
    ".mypy_cache/3.11/x.json",
    "notebooks/.ipynb_checkpoints/a-checkpoint.ipynb",
    "htmlcov/index.html",
    "src/guidedwavefem.egg-info/PKG-INFO",
    "build/lib/x.py",
    "dist/guidedwavefem-0.3.0.tar.gz",
    ".DS_Store",
    "configs/.DS_Store",
    "configs/._01_single_layer_aluminum.yaml",
    "Thumbs.db",
    "src/.cli.py.swp",
    "README.md~",
    ".coverage",
    ".coverage.host.1234",
    "coverage.xml",
]
KEEP = [
    "src/guidedwavefem/cli.py",
    "configs/01_single_layer_aluminum.yaml",
    "README.md",
    "tools/clean_junk.py",
    "results_case01/sensor_data.npz",
    "results_case01/__pycache__/keep.pyc",  # results are never touched
    ".dev/scratch/.DS_Store",  # protected scratch area
    ".venv/lib/x.pyc",  # virtual environments are never touched
    "src/build/keep.txt",  # build/ is junk only at the repository root
]


def _hooks() -> list[dict]:
    config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    return [hook for repo in config["repos"] for hook in repo["hooks"]]


def _forbid_regex() -> re.Pattern[str]:
    (hook,) = [hook for hook in _hooks() if hook["id"] == "forbid-junk"]
    return re.compile(hook["files"])


def test_hook_order_guard_first_cleaner_last():
    ids = [hook["id"] for hook in _hooks()]
    assert ids[0] == "forbid-junk"
    assert ids[-1] == "clean-junk"  # must run after hooks that create caches (e.g. ruff)


def _make_tree(root: Path, paths: list[str]) -> None:
    for rel in paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


@pytest.mark.parametrize("path", JUNK)
def test_forbid_junk_regex_blocks_junk(path):
    assert _forbid_regex().search(path)


@pytest.mark.parametrize(
    "path", ["src/guidedwavefem/cli.py", "configs/README.md", "tools/clean_junk.py", "LICENSE"]
)
def test_forbid_junk_regex_allows_sources(path):
    assert not _forbid_regex().search(path)


def test_cleaner_removes_junk_and_keeps_everything_else(tmp_path, capsys):
    _make_tree(tmp_path, JUNK + KEEP)
    assert clean_junk.main(["--root", str(tmp_path)]) == 0
    assert "clean-junk: Removed" in capsys.readouterr().out
    for rel in JUNK:
        assert not (tmp_path / rel).exists(), rel
    for rel in KEEP:
        assert (tmp_path / rel).exists(), rel
    assert not (tmp_path / "src" / "guidedwavefem" / "__pycache__").exists()


def test_cleaner_dry_run_deletes_nothing(tmp_path, capsys):
    _make_tree(tmp_path, JUNK)
    assert clean_junk.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert "Would remove" in capsys.readouterr().out
    for rel in JUNK:
        assert (tmp_path / rel).exists(), rel


def test_cleaner_never_deletes_tracked_files(tmp_path):
    _make_tree(tmp_path, ["tracked/__pycache__/vendored.pyc", "tracked/__pycache__/stray.pyc"])
    found = {
        p.relative_to(tmp_path).as_posix()
        for p in clean_junk.find_junk(tmp_path, frozenset({"tracked/__pycache__/vendored.pyc"}))
    }
    assert found == {"tracked/__pycache__/stray.pyc"}


def test_cleaner_reports_clean_tree(tmp_path, capsys):
    _make_tree(tmp_path, KEEP)
    assert clean_junk.main(["--root", str(tmp_path)]) == 0
    assert "working tree is clean" in capsys.readouterr().out
