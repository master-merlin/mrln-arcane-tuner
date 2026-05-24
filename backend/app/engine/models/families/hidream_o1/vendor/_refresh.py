"""Manual vendor refresh script for HiDream-O1 pipeline code.

Usage (from backend/):
    python -m app.engine.models.families.hidream_o1.vendor._refresh \
        --revision <commit-sha>

Clones HiDream-ai/HiDream-O1 at the specified revision into a temp dir,
copies the pipeline file(s) we use, strips unrelated extras, and writes
the SHA to vendor/REVISION.

Runs manually only. NEVER imported at module-load time.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://github.com/HiDream-ai/HiDream-O1.git"
VENDOR_DIR = Path(__file__).parent

# Files copied from the upstream repo (relative paths from upstream root).
FILES_TO_COPY: dict[str, str] = {
    # "upstream/relative/path.py": "local-name.py"
    # Filled in by Task 2 based on what the upstream repo actually contains.
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Upstream commit SHA")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"Cloning {UPSTREAM_REPO} at {args.revision}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(tmp_path / "src")],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path / "src"), "checkout", args.revision],
            check=True,
        )

        for upstream_rel, local_name in FILES_TO_COPY.items():
            src = tmp_path / "src" / upstream_rel
            dst = VENDOR_DIR / local_name
            if not src.exists():
                print(f"  MISSING upstream file: {upstream_rel}", file=sys.stderr)
                return 1
            shutil.copy2(src, dst)
            print(f"  copied {upstream_rel} -> {local_name}")

    (VENDOR_DIR / "REVISION").write_text(args.revision + "\n")
    print(f"Wrote REVISION: {args.revision}")
    print(
        "NOW: re-apply or forward-port every `# MRLN-PATCH:` marker,\n"
        "then update CHANGELOG.md and open a PR.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
