"""GPU smoke scenarios for the LTX-2 + WAN 2.x trainer-hardening work.

Builds a known-good training config per scenario and launches the real trainer
(``run_trainer.py``) so each fix can be validated end-to-end on hardware. Every
run uses a ``GPU_smoke_`` lora_name and a tiny ``max_train_steps`` — just enough
to clear the step-0 crashes and confirm a few training steps + a sample.

Usage (from repo root, venv python):
    backend\\venv\\Scripts\\python.exe backend/scripts/gpu_smoke.py <scenario> [--run] [--steps N]

    # print the launch command only (default):
    ... gpu_smoke.py wan21_t2v
    # actually launch it:
    ... gpu_smoke.py wan21_t2v --run

Scenarios:
    wan21_t2v   WAN 2.1 t2v on stills+video  -> proves the 4D-still 5D-lift (C1)
    wan21_i2v   WAN 2.1 i2v                  -> proves the first-frame data path (C2)
    wan22_t2v   WAN 2.2 t2v on stills+video  -> proves the 4D-still 5D-lift (C1)
    ltx2_t2v    LTX-2 t2v, model_shift, rk32 -> quality (A) — was the "weak LoRA" run
    ltx2_i2v    LTX-2 i2v                    -> proves first-frame conditioning (D)

These reproduce the configs that previously crashed at step 0 (same datasets),
now with the hardening fixes + the LTX-recommended recipe (rank/alpha 32,
timestep_sampling=model_shift, LR 1e-4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TRAINER = REPO / "backend" / "run_trainer.py"
PY = REPO / "backend" / "venv" / "Scripts" / "python.exe"

# Mixed stills+video dataset that triggered the WAN crashes.
_DATASETS = [
    {"dataset_name": "Airwolf_video", "num_repeats": 1, "caption_dropout_rate": 0.1},
    {"dataset_name": "Airwolf_images", "num_repeats": 1, "caption_dropout_rate": 0.1},
]

_SAMPLE_PROMPTS = [{"prompt": "Airwolf helicopter flying over a canyon at dusk", "seed": 42}]


def _base(definition_id: str, *, num_frames: int, video_mode: str = "t2v") -> dict:
    """Common smoke config — LTX-recommended LoRA recipe + tiny step budget."""
    return {
        "definition_id": definition_id,
        "datasets": _DATASETS,
        "resolutions": [512],
        "num_frames": num_frames,
        "video_mode": video_mode,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "network_rank": 32,             # LTX-recommended (the weak run used 16)
        "network_alpha": 32,
        "learning_rate": 1e-4,          # matches LTX shipped configs
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "constant",
        "timestep_sampling": "model_shift",   # inference-matched shift (this work)
        "max_train_steps": 20,          # just past the step-0 crashes + a few steps
        "save_every_n_steps": 0,
        # Exercise the sampler/preview path (the user's core concern): the step-0
        # baseline only fires when a sampler exists, and _create_sampler returns
        # None unless sample_every_n_steps > 0. main() keeps this in sync with
        # --steps so a sample also lands at the final step. Sampling failures are
        # non-fatal (logged as warnings), so they never mask the training result.
        "sample_before_training": True,
        "sample_every_n_steps": 20,
        "sample_prompts": _SAMPLE_PROMPTS,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "quantization": "none",
        "cache_latents": True,
        "cache_text_embeddings": True,
        "low_vram": True,
        "output_dir": "./outputs",
    }


def _scenario(name: str) -> tuple[str, dict]:
    if name == "wan21_t2v":
        return "wan2.1-t2v-14b", _base("wan2.1-t2v-14b", num_frames=25)
    if name == "wan22_t2v":
        return "wan2.2-t2v-a14b", _base("wan2.2-t2v-a14b", num_frames=25)
    if name == "wan21_i2v":
        cfg = _base("wan2.1-i2v-14b-720p", num_frames=25, video_mode="i2v")
        return "wan2.1-i2v-14b-720p", cfg
    if name == "ltx2_t2v":
        return "ltx2-3-base", _base("ltx2-3-base", num_frames=25)
    if name == "ltx2_i2v":
        cfg = _base("ltx2-3-base", num_frames=25, video_mode="i2v")
        cfg["first_frame_conditioning_probability"] = 1.0  # always condition, to exercise it
        return "ltx2-3-base", cfg
    raise SystemExit(f"unknown scenario {name!r}; see --help")


def main() -> None:
    p = argparse.ArgumentParser(description="LTX-2 + WAN trainer-hardening GPU smokes")
    p.add_argument("scenario", help="wan21_t2v | wan22_t2v | wan21_i2v | ltx2_t2v | ltx2_i2v")
    p.add_argument("--run", action="store_true", help="actually launch (default: print only)")
    p.add_argument("--steps", type=int, default=0, help="override max_train_steps")
    args = p.parse_args()

    definition_id, cfg = _scenario(args.scenario)
    cfg["lora_name"] = f"GPU_smoke_{args.scenario}"
    if args.steps > 0:
        cfg["max_train_steps"] = args.steps
    # Keep the end-of-run sample aligned with the (possibly overridden) budget so
    # a sampler is always created and a final sample renders.
    cfg["sample_every_n_steps"] = cfg["max_train_steps"]

    config_json = json.dumps(cfg)
    cmd = [str(PY), str(RUN_TRAINER), "--definition_id", definition_id, "--config", config_json]

    print(f"# scenario: {args.scenario}  ->  lora_name={cfg['lora_name']}")
    print(f"# definition: {definition_id}  video_mode={cfg['video_mode']}  steps={cfg['max_train_steps']}")
    if not args.run:
        print("# (dry run — pass --run to launch)\n")
        # Print a copy-pasteable command (config as a single-quoted JSON arg).
        print(f'{PY} {RUN_TRAINER} --definition_id {definition_id} --config \'{config_json}\'')
        return

    print("# launching…\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
