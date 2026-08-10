"""Manual vendor refresh script for MiniMax-H3 model code.

Usage (from backend/):
    python -m app.engine.models.families.minimax_h3.vendor._refresh \
        --revision 245d78fb48f1c87dfb560a94bea6e191c9f9f1c0

Clones huggingface/diffusers at the specified revision into a temp dir, copies
the vendored Python files, and writes the SHA to vendor/REVISION.

IMPORTANT — TWO upstream sources:
- Python code (this script): https://github.com/huggingface/diffusers, branch
  ``minimax-h3`` (PR #14355). H3 is NOT in any diffusers RELEASE — the installed
  diffusers 0.39.0 contains zero MiniMax code, so these classes exist only here.
  PR #14371 is actively refactoring the same code upstream; pinning is what makes
  that churn a non-event.
- Model weights (HuggingFace): https://huggingface.co/MiniMaxAI/MiniMax-H3
  Weights are NOT tracked via this script — resolved by the definition YAML's
  components.repo.path.

Runs manually only. NEVER imported at module-load time.

After running:
1. Inspect the diff.
2. Re-apply every ``MRLN-PATCH`` marker (see vendor/__init__.py for the list).
3. Update REVISION.
4. Open a PR with the diff. Refreshes are never automatic — they go through review.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://github.com/huggingface/diffusers.git"
# Historical origin of this code (PR #14355). NOT used for cloning below — the
# minimax-h3 branch was deleted upstream after the PR merged into diffusers
# main (confirmed 2026-08-10: `git clone --branch minimax-h3` returns "fatal:
# Remote branch minimax-h3 not found"). Kept here as documentation only.
UPSTREAM_BRANCH = "minimax-h3"
VENDOR_DIR = Path(__file__).parent

# Files copied from the upstream repo (key = path relative to upstream root,
# value = destination path relative to vendor/). Verified against the PR
# #14355 files API on 2026-08-05 — not guessed.
FILES_TO_COPY: dict[str, str] = {
    "src/diffusers/models/transformers/transformer_minimax_h3.py": "transformer_minimax_h3.py",
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py": "autoencoder_kl_minimax_h3.py",
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py": "autoencoder_kl_minimax_h3_audio.py",
    "src/diffusers/schedulers/scheduling_minimax_h3.py": "scheduling_minimax_h3.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Upstream commit SHA")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src_dir = tmp_path / "src"
        print(f"Fetching {UPSTREAM_REPO} @ {args.revision} ...")
        # MRLN-PATCH: fetch the pinned commit directly instead of
        # `git clone --branch UPSTREAM_BRANCH` — the minimax-h3 branch was
        # deleted upstream once PR #14355 merged, so that clone now fails
        # with "Remote branch minimax-h3 not found". GitHub still serves the
        # commit object by SHA even when no branch ref points to it, so
        # fetching by revision is robust to that cleanup (and to any future
        # rename/deletion of the branch) — and it is what we actually pin on.
        subprocess.run(["git", "init", "-q", str(src_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(src_dir), "remote", "add", "origin", UPSTREAM_REPO],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", args.revision],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(src_dir), "checkout", "FETCH_HEAD"],
            check=True,
        )

        for upstream_rel, local_rel in FILES_TO_COPY.items():
            src = src_dir / upstream_rel
            dst = VENDOR_DIR / local_rel
            if not src.exists():
                print(f"  MISSING upstream file: {upstream_rel}", file=sys.stderr)
                return 1
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {upstream_rel} -> vendor/{local_rel}")

    (VENDOR_DIR / "REVISION").write_text(
        "# Vendored Python code source: huggingface/diffusers, branch minimax-h3\n"
        "# (PR #14355 — NOT in any diffusers release as of 2026-08-05.)\n"
        f"DIFFUSERS_MINIMAX_H3_SHA={args.revision}\n"
        "\n"
        "# Model weights source: MiniMaxAI/MiniMax-H3 (HuggingFace)\n"
        "# Licence: MiniMax H3 Community License Agreement (NOT Apache-2.0).\n"
        "# (Not pinned via this file — resolved by the definition YAML's"
        " components.repo.path.)\n"
        "MINIMAX_H3_HF_REPO=https://huggingface.co/MiniMaxAI/MiniMax-H3\n",
        encoding="utf-8",
    )
    print("Wrote REVISION")
    print("NOW inspect the diff and re-apply any MRLN-PATCH markers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
