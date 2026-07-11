"""Profile a few LTX-2 training steps with the EXACT "LTX2 - testrun" recipe.

Purpose: apportion the per-step GPU-idle time on the underutilized video run
(GPU ~50%, 48 GB free) — is the idle in the cached-latent load + host→device
copy ("data_prep") or the compute ("forward_loss")? The training loop is
instrumented with a gated ``torch.profiler`` window (see
``PipelineTrainMixin._maybe_init_profiling``); this launcher just supplies the
profile flags + the testrun config and runs ~8 profiled steps.

Run ONLY when the GPU is free (it loads the full LTX-2 + audio stack, ~40 GB):
    backend\\venv\\Scripts\\python.exe backend/scripts/profile_train.py [--run]

Reuses the warm latent/text cache from the real run (same datasets/resolution/
frames), so the profiled steps are cache hits → reflect steady-state training,
not first-touch encode. Writes ``profile_summary.txt`` + ``trace.json`` under
the run's ``profile/`` dir; sampling is disabled so we measure the pure step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TRAINER = REPO / "backend" / "run_trainer.py"
PY = REPO / "backend" / "venv" / "Scripts" / "python.exe"

DEFINITION_ID = "ltx2-3-base"

# Faithful replica of the "LTX2 - testrun" template (numerics as real numbers,
# not the form's strings) + the profile overrides. RADC + audio + tiled, BS=1,
# grad-checkpointing + low_vram ON — i.e. exactly the config that underutilizes.
CONFIG: dict = {
    "definition_id": DEFINITION_ID,
    "datasets": [
        {
            "dataset_name": "Airwolf_video",
            "num_repeats": 1,
            "caption_dropout_rate": 0.1,
            "use_captions": True,
            "use_model_aware_captions": True,
            "num_frames": 0,
        },
        {
            "dataset_name": "Airwolf_images",
            "num_repeats": 1,
            "caption_dropout_rate": 0.1,
            "use_captions": True,
            "use_model_aware_captions": True,
            "num_frames": 0,
        },
    ],
    "resolutions": [768],
    "num_frames": 25,
    "video_mode": "t2v",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "network_rank": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "optimizer_type": "AdamW",
    "lr_scheduler": "constant",
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "min_snr_gamma": 5,
    # RADC timestep sampling (as in the testrun)
    "timestep_sampling": "radc",
    "radc_start": 0.9,
    "radc_end": 0.3,
    "radc_width": 0.5,
    "radc_res_influence": 0.15,
    "radc_seqlen_influence": 0,
    # Audio joint training
    "train_audio": True,
    "audio_loss_weight": 1.0,
    # Temporal sampling
    "temporal_coverage": "tiled",
    "max_windows": 10,
    "window_overlap": 0.3,
    "frame_stride": 1,
    "target_fps": 0,
    # Precision / VRAM (matches the underutilized run)
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "quantization": "none",
    "low_vram": True,
    "cache_latents": True,
    "cache_text_embeddings": True,
    "global_triggerword": "Airwolf222A",
    # ── Profile overrides ──
    "max_train_steps": 12,
    "profile_steps": 8,
    "profile_warmup": 3,
    "sample_before_training": False,
    "sample_every_n_steps": 0,
    "save_every_n_steps": 0,
    "lora_name": "PROFILE_ltx2_audio_radc",
    "output_dir": "./outputs",
}


def main() -> None:
    p = argparse.ArgumentParser(description="LTX-2 RADC+audio training-step profiler")
    p.add_argument("--run", action="store_true", help="actually launch (default: print only)")
    p.add_argument("--steps", type=int, default=0, help="override profile_steps")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.steps > 0:
        cfg["profile_steps"] = args.steps
        cfg["max_train_steps"] = cfg["profile_warmup"] + args.steps + 1

    out_dir = f"./outputs/{cfg['lora_name']}_{DEFINITION_ID}/profile"
    print(f"# profiling {cfg['profile_steps']} steps (warmup {cfg['profile_warmup']}) "
          f"of {DEFINITION_ID}  RADC+audio  BS={cfg['train_batch_size']}")
    print(f"# report -> {out_dir}\\profile_summary.txt  (+ trace.json)")

    cmd = [str(PY), str(RUN_TRAINER), "--definition_id", DEFINITION_ID,
           "--config", json.dumps(cfg)]
    if not args.run:
        print("# (dry run — pass --run to launch; needs a FREE GPU)\n")
        print(f"{PY} {RUN_TRAINER} --definition_id {DEFINITION_ID} --config '{json.dumps(cfg)}'")
        return

    print("# launching...\n")
    # Run from backend/ so the trainer resolves the SAME warm latent/text cache
    # and ./outputs path the real server-spawned run uses (CWD-relative).
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO / "backend")))


if __name__ == "__main__":
    main()
