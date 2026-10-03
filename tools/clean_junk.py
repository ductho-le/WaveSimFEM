#!/usr/bin/env python3
"""Remove regenerable junk files and folders from the repository working tree.

Targets interpreter/tool caches, build artifacts, coverage output, and OS/editor
litter. It deliberately never touches simulation results (``results_*``, ``*.npz``,
``*.bp``), virtual environments, ``.git``, the ``.dev`` scratch area, or any file
tracked by Git, so it is safe to run as a pre-commit hook or by hand::

    python tools/clean_junk.py            # delete junk
    python tools/clean_junk.py --dry-run  # only list what would be deleted

The script always exits 0 after a cleanup: removing untracked junk never changes
the commit contents. Committing junk is blocked separately by the ``forbid-junk``
hook in ``.pre-commit-config.yaml``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

#: Directory names removed wherever they appear in the tree.
JUNK_DIRS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "htmlcov",
    ".eggs",
    "*.egg-info",
)
#: Directory names removed only at the repository root (``build`` may be legitimate elsewhere).
ROOT_JUNK_DIRS: tuple[str, ...] = ("build", "dist")
#: File-name patterns removed wherever they appear in the tree.
JUNK_FILES: tuple[str, ...] = (
    "*.py[co]",
    ".DS_Store",
    "._*",
    "Thumbs.db",
    "ehthumbs.db",
    "[Dd]esktop.ini",
    "*.sw[po]",
    "*~",
    ".coverage",
    ".coverage.*",
    "coverage.xml",
)
#: Directory names that are never descended into or deleted.
PROTECTED_DIRS: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "env", "ENV", ".dev", "node_modules"}
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def repository_root(start: Path) -> Path:
    """Return the Git top-level directory containing ``start`` (or ``start`` itself)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return start.resolve()
    return Path(out.stdout.strip()).resolve()


def tracked_paths(root: Path) -> frozenset[str]:
    """Return repository-relative POSIX paths of every Git-tracked file (empty outside Git)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    return frozenset(path for path in out.stdout.split("\0") if path)


def find_junk(root: Path, tracked: frozenset[str] = frozenset()) -> Iterator[Path]:
    """Yield junk paths under ``root``, skipping protected and Git-tracked content.

    A junk directory containing tracked files is not yielded as a whole; its untracked
    junk files are still yielded individually.
    """
    tracked_dirs: set[str] = set()
    for path in tracked:
        parent = Path(path).parent
        while parent != Path("."):
            tracked_dirs.add(parent.as_posix())
            parent = parent.parent

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        at_root = rel_dir == "."
        keep: list[str] = []
        for name in sorted(dirnames):
            rel = name if at_root else f"{rel_dir}/{name}"
            if name in PROTECTED_DIRS or (at_root and fnmatch.fnmatchcase(name, "results_*")):
                continue
            is_junk = _matches(name, JUNK_DIRS) or (at_root and name in ROOT_JUNK_DIRS)
            if is_junk and rel not in tracked_dirs:
                yield current / name
            elif not (current / name).is_symlink():
                keep.append(name)
        dirnames[:] = keep

        for name in sorted(filenames):
            rel = name if at_root else f"{rel_dir}/{name}"
            if _matches(name, JUNK_FILES) and rel not in tracked:
                yield current / name


def remove(path: Path) -> int:
    """Delete ``path`` (file, symlink, or directory tree) and return the bytes freed."""
    if path.is_symlink() or not path.is_dir():
        size = path.lstat().st_size
        path.unlink()
        return size
    size = sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() or p.is_symlink())
    shutil.rmtree(path)
    return size


def _human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list junk without deleting it")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    parser.add_argument(
        "--root", type=Path, default=None, help="directory to clean (default: Git top level)"
    )
    args = parser.parse_args(argv)

    root = (args.root or repository_root(Path.cwd())).resolve()
    tracked = tracked_paths(root)
    verb = "Would remove" if args.dry_run else "Removed"

    count = freed = 0
    for path in find_junk(root, tracked):
        rel = path.relative_to(root).as_posix() + ("/" if path.is_dir() else "")
        try:
            freed += 0 if args.dry_run else remove(path)
        except OSError as exc:
            print(f"clean-junk: could not remove {rel}: {exc}", file=sys.stderr)
            continue
        count += 1
        if not args.quiet:
            print(f"  {verb.lower()} {rel}")

    if count:
        suffix = "" if args.dry_run else f" ({_human(freed)} freed)"
        print(f"clean-junk: {verb} {count} junk path(s){suffix}.")
    elif not args.quiet:
        print("clean-junk: working tree is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
