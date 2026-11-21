"""Release gate tests: reject mismatched tags, missing notes, or version drift."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_helpers = runpy.run_path(str(REPO / "tools" / "check_release.py"))
check_release = _helpers["check_release"]
release_metadata = _helpers["release_metadata"]


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    pkg = tmp_path / "src" / "wavesimfem"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('raise RuntimeError("must not import")\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "wavesimfem"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: "WaveSimFEM"\nversion: 1.2.3\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\nPending.\n\n"
        "## [1.2.3] — 2026-11-21 — release polish\n"
        "### Added\n- Automated releases.\n\n"
        "## [1.2.2] — 2026-10-10\nOlder notes.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_release_reads_version_without_importing_package(release_tree: Path) -> None:
    assert check_release(release_tree, "v1.2.3") == "### Added\n- Automated releases.\n"


def test_release_rejects_wrong_tag(release_tree: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        check_release(release_tree, "v1.2.4")


def test_release_rejects_citation_version_mismatch(release_tree: Path) -> None:
    (release_tree / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: "WaveSimFEM"\nversion: 1.2.2\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"CITATION\.cff version"):
        check_release(release_tree, "v1.2.3")


def test_release_allows_citation_without_top_level_version(release_tree: Path) -> None:
    (release_tree / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: "WaveSimFEM"\n', encoding="utf-8"
    )
    assert check_release(release_tree, "v1.2.3") == "### Added\n- Automated releases.\n"


@pytest.mark.parametrize(
    "changelog",
    [
        "## [Unreleased]\nPending.\n",
        "## [1.2.3] - 2026-11-21\n\n## [1.2.2]\nOld.\n",
    ],
)
def test_release_rejects_missing_notes(release_tree: Path, changelog: str) -> None:
    (release_tree / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError, match="no release notes"):
        check_release(release_tree, "v1.2.3")


@pytest.mark.parametrize(
    ("version", "prerelease"),
    [
        ("1.2.3a1", True),
        ("1.2.3b1", True),
        ("1.2.3rc1", True),
        ("1.2.3.dev1", True),
        ("1.2.3", False),
        ("1.2.3.post1", False),
        ("1.2.3.post1.dev1", True),
    ],
)
def test_release_prerelease_metadata_matches_python_versions(
    release_tree: Path, version: str, prerelease: bool
) -> None:
    (release_tree / "pyproject.toml").write_text(
        f'[project]\nname = "wavesimfem"\nversion = "{version}"\n', encoding="utf-8"
    )
    (release_tree / "CITATION.cff").write_text(
        f'cff-version: 1.2.0\ntitle: "WaveSimFEM"\nversion: {version}\n', encoding="utf-8"
    )
    (release_tree / "CHANGELOG.md").write_text(
        f"## [{version}]\nRelease notes.\n", encoding="utf-8"
    )
    assert check_release(release_tree, f"v{version}") == "Release notes.\n"
    assert release_metadata(version) == {"version": version, "prerelease": prerelease}


def test_release_rejects_invalid_python_version(release_tree: Path) -> None:
    (release_tree / "pyproject.toml").write_text(
        '[project]\nname = "wavesimfem"\nversion = "invalid"\n', encoding="utf-8"
    )
    (release_tree / "CHANGELOG.md").write_text("## [invalid]\nRelease notes.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid version"):
        check_release(release_tree, "vinvalid")


def test_release_cli_writes_notes_and_metadata(release_tree: Path, tmp_path: Path) -> None:
    script = release_tree / "tools" / "check_release.py"
    script.parent.mkdir()
    script.write_text((REPO / "tools" / "check_release.py").read_text(encoding="utf-8"))
    notes_file = tmp_path / "notes.md"
    metadata_file = tmp_path / "metadata.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tag",
            "v1.2.3",
            "--notes-file",
            str(notes_file),
            "--metadata-file",
            str(metadata_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert notes_file.read_text(encoding="utf-8") == "### Added\n- Automated releases.\n"
    assert json.loads(metadata_file.read_text(encoding="utf-8")) == {
        "version": "1.2.3",
        "prerelease": False,
    }


def test_repository_current_version_passes_release_gate() -> None:
    """Repository pyproject.toml, CITATION.cff, and CHANGELOG.md stay synchronized."""
    import tomllib

    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    notes = check_release(REPO, f"v{version}")
    assert notes.strip()
