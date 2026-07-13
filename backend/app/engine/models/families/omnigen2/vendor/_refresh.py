"""Manual vendor refresh script for OmniGen2 model code.

Usage (from backend/):
    python -m app.engine.models.families.omnigen2.vendor._refresh \
        --revision <commit-sha>

Clones VectorSpaceLab/OmniGen2 at the specified revision into a temp dir,
copies the vendored Python files, and writes the SHA to vendor/REVISION.

IMPORTANT — TWO upstream sources:
- Python code (this script): https://github.com/VectorSpaceLab/OmniGen2
  (Apache-2.0). diffusers has never merged OmniGen2 support, so the model
  classes only exist in this repo.
- Model weights (HuggingFace): https://huggingface.co/OmniGen2/OmniGen2
  Weights are NOT tracked via this script — resolved by the definition
  YAML's components.repo.path.

Runs manually only. NEVER imported at module-load time.

After running:
1. Inspect the diff.
2. Re-apply every ``MRLN-PATCH`` marker — the vendored copies STRIP
   flash-attn / triton / TeaCache / TaylorSeer relative to upstream (see
   vendor/__init__.py for the full strip list); a raw upstream copy will
   NOT import in this repo (flash_attn is not a dependency).
3. Update REVISION.
4. Open a PR with the diff. Refreshes are never automatic — they go
   through review.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://github.com/VectorSpaceLab/OmniGen2.git"
VENDOR_DIR = Path(__file__).parent

# Files copied from the upstream repo (key = path relative to upstream root,
# value = destination path relative to vendor/).
FILES_TO_COPY: dict[str, str] = {
    "omnigen2/models/transformers/transformer_omnigen2.py": "models/transformers/transformer_omnigen2.py",
    "omnigen2/models/transformers/repo.py": "models/transformers/repo.py",
    "omnigen2/models/transformers/block_lumina2.py": "models/transformers/block_lumina2.py",
    "omnigen2/models/transformers/components.py": "models/transformers/components.py",
    "omnigen2/models/attention_processor.py": "models/attention_processor.py",
    "omnigen2/models/embeddings.py": "models/embeddings.py",
    "omnigen2/schedulers/scheduling_flow_match_euler_discrete.py": "schedulers/scheduling_flow_match_euler_discrete.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Upstream commit SHA")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"Cloning {UPSTREAM_REPO} at {args.revision}...")
        subprocess.run(
            ["git", "clone", UPSTREAM_REPO, str(tmp_path / "src")],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path / "src"), "checkout", args.revision],
            check=True,
        )

        for upstream_rel, local_rel in FILES_TO_COPY.items():
            src = tmp_path / "src" / upstream_rel
            dst = VENDOR_DIR / local_rel
            if not src.exists():
                print(f"  MISSING upstream file: {upstream_rel}", file=sys.stderr)
                return 1
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {upstream_rel} -> vendor/{local_rel}")

    revision_file = VENDOR_DIR / "REVISION"
    revision_file.write_text(
        "# Vendored Python code source: VectorSpaceLab/OmniGen2 (GitHub, Apache-2.0)\n"
        f"VECTORSPACELAB_OMNIGEN2_SHA={args.revision}\n"
        "\n"
        "# Model weights source: OmniGen2/OmniGen2 (HuggingFace)\n"
        "# (Not pinned via this file — resolved by the definition YAML's components.repo.path.)\n"
        "OMNIGEN2_HF_REPO=https://huggingface.co/OmniGen2/OmniGen2\n",
        encoding="utf-8",
    )
    print(f"Wrote {revision_file}")
    print(
        "NOW RE-APPLY the MRLN-PATCH strips (flash-attn / triton / TeaCache /"
        " TaylorSeer) — see vendor/__init__.py — the raw copies will not"
        " import without them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
