"""Krea-2 arch-capture spike.

Runs ONLY in .agent/workdir/krea2_spike_venv (diffusers-main / 0.39.dev with
Krea2 support layered over the app venv's torch+transformers). NOT imported by
the app. Dumps .agent/workdir/krea2_arch.json — the source of truth for the
krea2 family definitions, LoRA targets, saver keys, and driver forward wiring.

Module names + signatures come from a META-device instantiation of the real
Turbo transformer/config.json (no 26GB weight load needed). Provenance + helper
signatures come from the installed diffusers-main source.

Usage (from repo root, in the spike venv):
    & .agent/workdir/krea2_spike_venv/Scripts/python.exe backend/scripts/krea2_spike.py
"""

import inspect
import json
import os

import torch

TURBO = (
    "D:/AI/huggingface/hub/hub/models--krea--Krea-2-Turbo/snapshots/"
    "1161245028ef398cd0a951101b2bbf486464f841"
)
OUT = ".agent/workdir/krea2_arch.json"


def main() -> None:
    import diffusers
    from diffusers import Krea2Transformer2DModel
    from diffusers.pipelines.krea2 import pipeline_krea2

    rec: dict = {"diffusers_version": diffusers.__version__}

    # --- transformer config + module structure (meta device, no weights) ---
    with open(os.path.join(TURBO, "transformer", "config.json")) as f:
        tf_cfg = json.load(f)
    rec["transformer_config"] = tf_cfg

    with torch.device("meta"):
        model = Krea2Transformer2DModel.from_config(tf_cfg)

    rec["top_level_children"] = [n for n, _ in model.named_children()]
    rec["linear_module_names"] = sorted({
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    })
    # Suffixes (last 1-2 dotted segments) — the LoRA-target form the families use
    suffixes = set()
    for n in rec["linear_module_names"]:
        parts = n.split(".")
        suffixes.add(parts[-1])
        if len(parts) >= 2:
            suffixes.add(".".join(parts[-2:]))
    rec["linear_suffixes"] = sorted(suffixes)

    # --- forward + conditioning helper signatures ---
    rec["transformer_forward_sig"] = str(inspect.signature(model.forward))
    helper_sigs = {}
    for name in dir(pipeline_krea2):
        obj = getattr(pipeline_krea2, name)
        if callable(obj) and name in (
            "Krea2Pipeline",
        ):
            for meth in ("get_text_hidden_states", "prepare_position_ids",
                         "encode_prompt"):
                if hasattr(obj, meth):
                    helper_sigs[meth] = str(
                        inspect.signature(getattr(obj, meth)))
    rec["pipeline_helper_sigs"] = helper_sigs
    rec["pipeline_module_members"] = [
        n for n in dir(pipeline_krea2) if not n.startswith("_")
    ]

    # --- vendor source provenance ---
    import diffusers.models.transformers.transformer_krea2 as tk
    rec["vendor_source_paths"] = {
        "transformer_krea2": tk.__file__,
        "pipeline_krea2": pipeline_krea2.__file__,
    }

    # --- pipeline-level config (model_index) ---
    with open(os.path.join(TURBO, "model_index.json")) as f:
        rec["model_index"] = json.load(f)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", OUT)
    print("diffusers:", rec["diffusers_version"])
    print("top_level_children:", rec["top_level_children"])
    print("n linear modules:", len(rec["linear_module_names"]))
    print("forward sig:", rec["transformer_forward_sig"])


if __name__ == "__main__":
    main()
