"""A stand-in for ``run_trainer.py`` in the sampler comparison's tests. Standard library only.

It takes ``--definition_id`` and ``--config`` as the trainer does, asks the backend on ``PORT``
for its dataset, and writes what a run with validation writes: ``training_log.json``,
``validation_split.json`` and ``validation_metrics.jsonl`` (records in the product's shape), and
``run_seed_applied`` / ``step_progress`` lines on stdout. ``--mode`` picks a behaviour; every mode
but ``exit1``, ``hang`` and ``partial`` writes every pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Each arm's loss offset (by its mode) and each seed's: the seed gap within an arm is SEED_STEP.
MODE_OFFSETS = {"uniform": 0.0, "logit_normal": 0.004, "radc": -0.006}
OTHER_MODE_OFFSET = 0.002
SEED_STEP = 0.0005
PICTURES = (".png", ".jpg", ".jpeg", ".webp")
MODES = ("ok", "exit1", "hang", "partial", "wrong_sampler", "other_split", "no_digest", "other_model",
         "no_model", "wrong_split_seed", "wrong_draws", "wrong_fraction", "wrong_bands", "foreign_picture",
         "wrong_seed", "one_bad_record")


def say(event: str, **fields) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


def offset(config: dict) -> float:
    mode = config.get("timestep_sampling", "")
    return MODE_OFFSETS.get(mode, OTHER_MODE_OFFSET) + SEED_STEP * (int(config["seed"]) - 42)


def passes(config: dict) -> list[int]:
    steps, every = int(config["max_train_steps"]), int(config["validation_every_n_steps"])
    return [s for s in range(steps) if s % every == 0 or s == steps - 1]


def run_dir(config: dict, definition_id: str) -> Path:
    model_part = definition_id.split("/")[-1].replace(":", "_")
    return Path(config["output_dir"]) / f"{config['lora_name']}_{model_part}"


def dataset_name(config: dict) -> str:
    entry = config["datasets"][0]
    return entry if isinstance(entry, str) else entry["dataset_name"]


def ask_backend(name: str) -> Path:
    url = f"http://127.0.0.1:{os.environ['PORT']}/api/datasets/{urllib.parse.quote(name)}"
    with urllib.request.urlopen(url, timeout=10) as answer:
        return Path(json.loads(answer.read())["path"])


def split_content(config: dict, definition_id: str, mode: str, folder: Path, name: str) -> dict:
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in PICTURES)
    count = max(1, round(len(files) * float(config["validation_holdout_fraction"])))
    held = files[-count:] if mode == "other_split" else files[:count]
    representatives = []
    for i, file in enumerate(held):
        item = {"key": f"{name}/{file}", "path": str(folder / file), "caption": "", "prefix": "",
                "target_w": 64, "target_h": 64}
        if mode != "no_digest":
            item["sha256"] = hashlib.sha256((folder / file).read_bytes()).hexdigest()
        if mode == "foreign_picture" and i == 0:
            item["sha256"] = "0" * 64
        representatives.append(item)
    model = {"other_model": "flux3-image-klein-base", "no_model": None}.get(mode, definition_id)
    return {
        "holdout": [f"{name}/{file}" for file in held],
        "groups": [[f"{name}/{file}"] for file in files],
        "representatives": representatives,
        "split_seed": 1 if mode == "wrong_split_seed" else int(config["validation_split_seed"]),
        "global_triggerword": config.get("global_triggerword", ""),
        "holdout_fraction": 0.15 if mode == "wrong_fraction" else float(config["validation_holdout_fraction"]),
        "validation_dataset": config.get("validation_dataset") or None,
        "hash_threshold_bits": 6,
        "every_n_steps": int(config["validation_every_n_steps"]),
        "timestep_bands": 4 if mode == "wrong_bands" else int(config["validation_timestep_bands"]),
        "noise_draws": 1 if mode == "wrong_draws" else int(config["validation_noise_draws"]),
        "model": model,
    }


def record(step: int, split: dict, config: dict, draws: int) -> dict:
    """One pass's record, in the shape ``validation_pass._record`` writes."""
    keys = sorted(split["holdout"])
    bands = int(split["timestep_bands"])
    base = 1.0 + offset(config)
    losses = [[[base + 0.01 * i + 0.001 * b + 0.0001 * d for b in range(bands)] for d in range(draws)]
              for i in range(len(keys))]
    band_loss = [sum(sum(image[d][b] for d in range(draws)) / draws for image in losses) / len(keys)
                 for b in range(bands)]
    return {"step": step, "images": len(keys), "band_centres": [(b + 0.5) / bands for b in range(bands)],
            "band_loss": band_loss, "mean": sum(band_loss) / bands, "worst_band": band_loss.index(max(band_loss)),
            "seconds": 0.01, "version": 2, "image_keys": keys, "noise_draws": draws,
            "train_seed": int(config["seed"]), "image_band_loss": losses}


def write_split(config: dict, definition_id: str, mode: str, folder: Path, name: str) -> tuple[Path, dict]:
    out = run_dir(config, definition_id)
    out.mkdir(parents=True, exist_ok=True)
    split = split_content(config, definition_id, mode, folder, name)
    (out / "validation_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return out, split


def write_pass(out: Path, step: int, split: dict, config: dict, mode: str, last: bool) -> None:
    draws = 1 if mode == "one_bad_record" and last else int(split["noise_draws"])
    with open(out / "validation_metrics.jsonl", "a", encoding="utf-8") as metrics:
        metrics.write(json.dumps(record(step, split, config, draws)) + "\n")


def write_training_log(out: Path, config: dict, mode: str) -> None:
    logged = {**config, "timestep_sampling": "radc"} if mode == "wrong_sampler" else config
    (out / "training_log.json").write_text(json.dumps({"config": logged}, indent=2), encoding="utf-8")


def write_run(config: dict, definition_id: str, mode: str, folder: Path, name: str) -> Path:
    """Every file a finished run of ``mode`` writes, at once (for tests that place runs on disk)."""
    out, split = write_split(config, definition_id, mode, folder, name)
    steps = passes(config)
    for i, step in enumerate(steps):
        write_pass(out, step, split, config, mode, last=i == len(steps) - 1)
    write_training_log(out, config, mode)
    return out


def applied_seed_line(config: dict, mode: str) -> str:
    return json.dumps({"event": "run_seed_applied", "seed": 7 if mode == "wrong_seed" else int(config["seed"])})


def _work_files(config: dict) -> tuple[int, bool]:
    """The telemetry file's line count and whether the identity file exists, for this run."""
    logs = Path(config["output_dir"]).parent / "logs"
    lines, identity = -1, False
    for path in logs.glob("*.telemetry.csv"):
        label = path.name[: -len(".telemetry.csv")]
        if config["lora_name"].endswith("_" + label):
            lines = len(path.read_text(encoding="utf-8").splitlines())
            identity = (logs / f"{label}.identity.json").exists()
    return lines, identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition_id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="ok", choices=MODES)
    args = parser.parse_args()
    mode = args.mode
    config = json.loads(args.config)
    lines, identity = _work_files(config)
    say("fake_trainer_started", pid=os.getpid(), port=os.environ.get("PORT"), telemetry_lines=lines,
        identity_found=identity, db=os.environ.get("MRLN_DB_PATH"), offline=os.environ.get("HF_HUB_OFFLINE"))
    if mode == "exit1":
        say("fake_trainer_failed")
        return 1
    name = dataset_name(config)
    folder = ask_backend(name)
    sys.stdout.write(applied_seed_line(config, mode) + "\n")
    sys.stdout.flush()
    out, split = write_split(config, args.definition_id, mode, folder, name)
    if mode == "hang":
        write_training_log(out, config, mode)
        while True:
            time.sleep(1)
    steps = passes(config)
    if mode == "partial":
        steps = steps[:2]
    for i, step in enumerate(steps):
        write_pass(out, step, split, config, mode, last=i == len(steps) - 1)
        say("step_progress", step=step + 1, step_time=0.01)
    write_training_log(out, config, mode)
    say("training_completed_successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
