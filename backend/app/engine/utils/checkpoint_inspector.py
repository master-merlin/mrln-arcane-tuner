"""
Checkpoint Inspector — reads checkpoint metadata and inventories saved
components without loading heavy weight tensors.

Used by the API endpoint ``GET /checkpoints/inspect`` and by the
``validate_compatibility`` check in ``CheckpointManager``.
"""

import json
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def inspect_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Read metadata from a training checkpoint without loading weights.

    Scans the checkpoint directory for:
    - ``training_state.json`` (step, config, timestamp)
    - ``.pt`` files (optimizer, scheduler, scaler, EMA, component weights)
    - PEFT adapter subdirectories (contain ``adapter_config.json``)
    - ``.safetensors`` files (distribution LoRA)

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Dict with ``valid``, ``global_step``, ``config``, ``components``,
        ``adapters``, ``files``, and capability booleans.
    """
    if not os.path.exists(checkpoint_path):
        return {"valid": False, "error": "Path does not exist"}

    if not os.path.isdir(checkpoint_path):
        return {"valid": False, "error": "Path is not a directory"}

    state_file = os.path.join(checkpoint_path, "training_state.json")
    if not os.path.exists(state_file):
        return {"valid": False, "error": "training_state.json not found in checkpoint"}

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("checkpoint_inspect_parse_error", path=state_file, error=str(e))
        return {"valid": False, "error": f"Failed to parse training_state.json: {e}"}

    # Scan filesystem for components
    files: dict[str, int] = {}
    components: list[str] = []
    adapters: list[str] = []

    for entry in os.scandir(checkpoint_path):
        if entry.is_file():
            files[entry.name] = entry.stat().st_size

            # Track .pt component files (exclude known infrastructure files)
            if entry.name.endswith(".pt") and entry.name not in (
                "optimizer.pt", "scheduler.pt", "scaler.pt", "ema_shadow.pt",
                "lora_raw_state.pt",
            ):
                components.append(entry.name.rsplit(".", 1)[0])

        elif entry.is_dir():
            # Check if subdirectory is a PEFT adapter
            adapter_config = os.path.join(entry.path, "adapter_config.json")
            if os.path.exists(adapter_config):
                adapters.append(entry.name)
                # Add adapter files to manifest
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_file():
                        files[f"{entry.name}/{sub_entry.name}"] = sub_entry.stat().st_size

    info: dict[str, Any] = {
        "valid": True,
        "global_step": state.get("global_step", 0),
        "timestamp": state.get("timestamp", 0.0),
        "config": state.get("config", {}),
        "components": components,
        "adapters": adapters,
        "files": files,
        "has_optimizer": "optimizer.pt" in files,
        "has_scheduler": "scheduler.pt" in files,
        "has_scaler": "scaler.pt" in files,
        "has_ema": "ema_shadow.pt" in files,
        "has_te_cache": "te_cache.safetensors" in files,
        "has_cache_manifest": "cache_manifest" in state,
        "total_size_bytes": sum(files.values()),
    }

    logger.info(
        "checkpoint_inspected",
        path=checkpoint_path,
        step=info["global_step"],
        components=components,
        adapters=adapters,
        total_size_mb=round(info["total_size_bytes"] / (1024 * 1024), 2),
    )

    return info
