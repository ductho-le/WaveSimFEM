#!/usr/bin/env python3
"""Validate release tags and extract changelog notes without importing wavesimfem."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version


def release_metadata(version: str) -> dict[str, str | bool]:
    """Parse a PEP 440 version string and return release metadata."""
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version {version!r}: {exc}") from exc
    return {"version": version, "prerelease": parsed.is_prerelease}


def _read_pyproject_version(root: Path) -> str:
    pyproject_path = root / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Missing [project].version in {pyproject_path}")
    return version.strip()


def _check_citation_version(root: Path, expected_version: str) -> None:
    citation_path = root / "CITATION.cff"
    if not citation_path.is_file():
        return
    data = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data or data["version"] is None:
        return
    citation_version = str(data["version"]).strip()
    if citation_version != expected_version:
        raise ValueError(
            f"CITATION.cff version {citation_version!r} does not match "
            f"pyproject.toml version {expected_version!r}"
        )


def check_release(root: Path | str, tag: str) -> str:
    """Validate ``tag`` against ``pyproject.toml``, ``CITATION.cff``, and ``CHANGELOG.md``.

    Returns the extracted Markdown release notes for ``tag``.
    """
    root = Path(root)
    version = _read_pyproject_version(root)
    release_metadata(version)
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match package version {version}")
    _check_citation_version(root, version)

    changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]:|\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError(f"Version {version} has no release notes in CHANGELOG.md")
    return match.group(1).strip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Git release tag (e.g. v0.4.0)")
    parser.add_argument("--notes-file", type=Path, default=None, help="Output path for notes")
    parser.add_argument(
        "--metadata-file", type=Path, default=None, help="Output path for JSON metadata"
    )
    args = parser.parse_args(argv)
    try:
        notes = check_release(Path(__file__).resolve().parents[1], args.tag)
    except ValueError as error:
        parser.error(str(error))
    if args.notes_file is not None:
        args.notes_file.write_text(notes, encoding="utf-8")
    if args.metadata_file is not None:
        args.metadata_file.write_text(
            json.dumps(release_metadata(args.tag.removeprefix("v"))), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
