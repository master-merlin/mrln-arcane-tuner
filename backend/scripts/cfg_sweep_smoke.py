"""Render the SAME prompt+seed at several CFG values to pick a guidance scale.

Now that the LTX-2 sampler honours ``guidance_scale`` (classifier-free guidance:
cond + unconditional forward), the in-training preview finally reflects the LoRA.
This smoke loads a trained LoRA via ``resume_from_checkpoint`` and emits one
sample clip per CFG value — same prompt, same seed — so only guidance differs.

Because ``guidance_scale`` is a PER-PROMPT field, the sweep is just N sample
prompts (identical text/seed, different ``guidance_scale``) rendered once via
``sample_before_training``. No training happens beyond a single throwaway step.

Run on a FREE GPU (loads the full LTX-2 stack, ~40 GB):
    backend\\venv\\Scripts\\python.exe backend/scripts/cfg_sweep_smoke.py \\
        --checkpoint outputs/airwolf_audio_stills_ltx2-3-base/<checkpoint-dir> \\
        --cfgs 1,2,3,4,5 --run

Outputs land in ``outputs/CFGSWEEP_ltx2_ltx2-3-base/samples/`` — one clip per
prompt index; the index→CFG mapping is printed below.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TRAINER = REPO / "backend" / "run_trainer.py"
PY = REPO / "backend" / "venv" / "Scripts" / "python.exe"

DEFINITION_ID = "ltx2-3-base"

# Mirrors the "LTX2 - testrun" recipe so the model setup (audio targets, rank)
# matches the trained checkpoint we resume. Only the sampling path is exercised.
BASE_CONFIG: dict = {
    "definition_id": DEFINITION_ID,
    "datasets": [
        {"dataset_name": "Airwolf_video", "num_repeats": 1, "use_captions": True,
         "use_model_aware_captions": True, "num_frames": 0},
        {"dataset_name": "Airwolf_images", "num_repeats": 1, "use_captions": True,
         "use_model_aware_captions": True, "num_frames": 0},
    ],
    "resolutions": [768],
    "still_resolutions": [1920, 1280],
    "resolution_strategy": "mixed",
    "num_frames": 25,
    "video_mode": "t2v",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "network_rank": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "optimizer_type": "AdamW",
    "lr_scheduler": "constant",
    "timestep_sampling": "radc",
    "radc_start": 0.9, "radc_end": 0.3, "radc_width": 0.5,
    "radc_res_influence": 0.15, "radc_seqlen_influence": 0,
    "train_audio": True,
    "audio_loss_weight": 1.0,
    "temporal_coverage": "tiled",
    "max_windows": 10, "window_overlap": 0.3, "frame_stride": 1, "target_fps": 0,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "quantization": "none",
    "low_vram": False,
    "cache_latents": True,
    "cache_text_embeddings": True,
    "global_triggerword": "Airwolf222A",
    # ── Sample-only overrides ──
    "max_train_steps": 1,
    "sample_before_training": True,
    "sample_every_n_steps": 0,
    "save_every_n_steps": 0,
    "sample_num_frames": 25,
    "sample_fps": 24,
    "lora_name": "CFGSWEEP_ltx2",
    "output_dir": "./outputs",
}

DEFAULT_PROMPT = (
    "[triggerword], the black military attack helicopter Airwolf in flight, "
    "side-mounted weapon pods and chin rocket launcher clearly visible, "
    "cinematic, sharp detail"
)


def main() -> None:
    p = argparse.ArgumentParser(description="LTX-2 CFG sweep smoke")
    p.add_argument("--checkpoint", default="", help="resume_from_checkpoint dir (the trained LoRA)")
    p.add_argument("--cfgs", default="1,2,3,4,5", help="comma list of guidance_scale values")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default="", help="sample_negative_prompt (uncond)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--no-audio", action="store_true", help="set train_audio=False")
    p.add_argument("--run", action="store_true", help="actually launch (default: print only)")
    args = p.parse_args()

    cfgs = [float(c) for c in args.cfgs.split(",") if c.strip()]
    cfg = dict(BASE_CONFIG)
    cfg["sample_negative_prompt"] = args.negative
    if args.no_audio:
        cfg["train_audio"] = False

    # resume_from_checkpoint restores global_step (>0), which suppresses
    # sample_before_training AND empties the range(start_step, 1) loop — so we
    # instead run ONE loop step at learning_rate=0 (the LoRA is frozen) with
    # sample_every_n_steps=1, so the resumed weights get sampled at every CFG.
    # No checkpoint → sample the BASE model via sample_before_training.
    if args.checkpoint:
        m = re.search(r"checkpoint-(\d+)", args.checkpoint.replace("\\", "/"))
        if not m:
            raise SystemExit(
                "--checkpoint must be a numbered dir like .../checkpoint-006000 "
                "(the 'final' dir has no step in its name — use its checkpoint-NNNNNN)"
            )
        ckpt_step = int(m.group(1))
        cfg["resume_from_checkpoint"] = args.checkpoint
        cfg["learning_rate"] = 0.0  # freeze: the 1 loop step must not move the LoRA
        cfg["sample_before_training"] = False
        cfg["sample_every_n_steps"] = 1
        cfg["sample_skip_first_n_steps"] = 0
        cfg["max_train_steps"] = ckpt_step + 2  # one iteration at step ckpt_step+1
    else:
        cfg["sample_before_training"] = True
        cfg["sample_every_n_steps"] = 0
        cfg["max_train_steps"] = 1

    cfg["sample_prompts"] = [
        {
            "prompt": args.prompt,
            "seed": args.seed,            # SAME seed → only guidance differs
            "width": args.width,
            "height": args.height,
            "num_inference_steps": args.steps,
            "guidance_scale": g,
        }
        for g in cfgs
    ]

    out_dir = f"./outputs/{cfg['lora_name']}_{DEFINITION_ID}/samples"
    print(f"# CFG sweep: {cfgs}  seed={args.seed}  steps={args.steps}  "
          f"{args.width}x{args.height}  audio={cfg['train_audio']}")
    print(f"# checkpoint: {args.checkpoint or '(none — samples BASE, LoRA not loaded)'}")
    print("# sample index -> CFG mapping:")
    for i, g in enumerate(cfgs):
        print(f"#   sample {i}  ->  CFG {g}")
    print(f"# clips -> {out_dir}")

    if not args.checkpoint:
        print("# WARNING: no --checkpoint → the LoRA is NOT loaded; you'll see the base model.")

    command = [str(PY), str(RUN_TRAINER), "--definition_id", DEFINITION_ID,
               "--config", json.dumps(cfg)]
    if not args.run:
        print("\n# (dry run — pass --run to launch; needs a FREE GPU)")
        return

    print("\n# launching...\n")
    # Run from backend/ so the trainer resolves the warm cache + ./outputs path.
    raise SystemExit(subprocess.call(command, cwd=str(REPO / "backend")))


if __name__ == "__main__":
    main()
