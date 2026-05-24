"""Manual vendor refresh script for HiDream-O1 pipeline code.

Usage (from backend/):
    python -m app.engine.models.families.hidream_o1.vendor._refresh \
        --revision <commit-sha>

Clones Saganaki22/HiDream_O1-ComfyUI at the specified revision into a temp
dir, copies the six Python files we vendor, and writes the SHA to
vendor/REVISION.

IMPORTANT — TWO upstream sources:
- Python code (this script): https://github.com/Saganaki22/HiDream_O1-ComfyUI
  Rationale: HiDream-ai's GitHub repo (HiDream-ai/HiDream-O1-Image) is
  inference-only and does NOT include the custom model class
  (Qwen3VLModelOutputWithPast + x_embedder + final_layer2) required for
  training. Saganaki22's MIT-licensed ComfyUI integration vendors the
  actual checkpoint architecture and re-implements ai-toolkit's May 2026
  LoRA training recipe.
- Model weights (HuggingFace): https://huggingface.co/HiDream-ai/HiDream-O1-Image
  Weights are NOT tracked via this script. They are pinned by the model
  definition YAML's components.unet.revision field.

Runs manually only. NEVER imported at module-load time.

After running:
1. Inspect the diff.
2. Re-apply or forward-port any `# MRLN-PATCH:` markers (currently only the
   relative-import fix in qwen3_vl_transformers.py).
3. Update REVISION and append a row to CHANGELOG.md.
4. Open a PR with the diff. Refreshes are never automatic — they go through review.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://github.com/Saganaki22/HiDream_O1-ComfyUI.git"
VENDOR_DIR = Path(__file__).parent

# Files copied from the upstream repo (key = path relative to upstream root,
# value = destination filename inside vendor/).
FILES_TO_COPY: dict[str, str] = {
    "hidream_o1/models/pipeline.py": "pipeline.py",
    "hidream_o1/models/qwen3_vl_transformers.py": "qwen3_vl_transformers.py",
    "hidream_o1/models/flash_scheduler.py": "flash_scheduler.py",
    "hidream_o1/models/fm_solvers_unipc.py": "fm_solvers_unipc.py",
    "hidream_o1/models/seam_smoothing.py": "seam_smoothing.py",
    "hidream_o1/models/utils.py": "utils.py",
    "hidream_o1/compat.py": "compat.py",
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

    (VENDOR_DIR / "REVISION").write_text(
        f"# Vendored Python code source: Saganaki22/HiDream_O1-ComfyUI (GitHub)\n"
        f"SAGANAKI22_SHA={args.revision}\n\n"
        f"# Model weights source: HiDream-ai/HiDream-O1-Image (HuggingFace)\n"
        f"# (Not pinned via this file — pinned by the definition YAML's components.unet.revision.)\n"
        f"HIDREAM_AI_REPO=https://huggingface.co/HiDream-ai/HiDream-O1-Image\n"
    )
    print(f"Wrote REVISION: {args.revision}")
    print(
        "NOW: re-apply or forward-port every `# MRLN-PATCH:` marker,\n"
        "then update CHANGELOG.md and open a PR.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
