"""
Version Sync Utility
====================
Single source of truth: backend/app/__init__.py (__version__)

Usage:
    # Sync README.md and package.json to match __init__.py
    python backend/sync_version.py

    # Bump to a new version, then sync all files
    python backend/sync_version.py 0.3.0-alpha
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = ROOT / "backend" / "app" / "__init__.py"
README_FILE = ROOT / "README.md"
PACKAGE_JSON = ROOT / "frontend" / "package.json"

VERSION_RE = re.compile(r'^__version__\s*=\s*"(.+?)"', re.MULTILINE)
README_RE = re.compile(r"`v[\d]+\.[\d]+\.[\d]+[^`]*`")


def _read_raw(path: Path) -> tuple[str, str]:
    """Read file preserving line-ending style. Returns (text, newline)."""
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8")
    return text, newline


def _write_raw(path: Path, text: str, newline: str) -> None:
    """Write file preserving original line-ending style."""
    if newline == "\r\n":
        # Normalize to \n first, then convert to \r\n
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    else:
        text = text.replace("\r\n", "\n")
    path.write_bytes(text.encode("utf-8"))


def read_source_version() -> str:
    """Read the current __version__ from the canonical source."""
    text, _ = _read_raw(SOURCE_FILE)
    match = VERSION_RE.search(text)
    if not match:
        print(f"ERROR: Could not find __version__ in {SOURCE_FILE}")
        sys.exit(1)
    return match.group(1)


def write_source_version(version: str) -> None:
    """Update __version__ in the canonical source file."""
    text, newline = _read_raw(SOURCE_FILE)
    new_text = VERSION_RE.sub(f'__version__ = "{version}"', text)
    _write_raw(SOURCE_FILE, new_text, newline)
    print(f"  ✓ {SOURCE_FILE.relative_to(ROOT)} → {version}")


def sync_readme(version: str) -> None:
    """Patch the version badge in README.md."""
    text, newline = _read_raw(README_FILE)
    new_text = README_RE.sub(f"`v{version}`", text, count=1)
    if new_text == text:
        print(f"  ⚠ {README_FILE.relative_to(ROOT)} — no version badge found, skipped")
        return
    _write_raw(README_FILE, new_text, newline)
    print(f"  ✓ {README_FILE.relative_to(ROOT)} → v{version}")


def sync_package_json(version: str) -> None:
    """Patch the version field in frontend/package.json."""
    text, newline = _read_raw(PACKAGE_JSON)
    data = json.loads(text)
    data["version"] = version
    # json.dumps uses \n — we normalize in _write_raw
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _write_raw(PACKAGE_JSON, new_text, newline)
    print(f"  ✓ {PACKAGE_JSON.relative_to(ROOT)} → {version}")


def main() -> None:
    new_version = sys.argv[1] if len(sys.argv) > 1 else None

    if new_version:
        print(f"\n📦 Bumping version to {new_version}\n")
        write_source_version(new_version)
    else:
        print("\n📦 Syncing version from source of truth\n")

    version = read_source_version()
    print(f"  Source: {version}\n")

    sync_readme(version)
    sync_package_json(version)

    print(f"\n✅ All files synced to v{version}\n")


if __name__ == "__main__":
    main()
