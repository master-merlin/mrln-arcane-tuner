"""MiniMax-H3 LoRA saver (plan row 2.8) — the ORIGINAL-checkpoint layout.

Production caller: ``MiniMaxH3Driver.get_saver()`` → ``pipeline_optimization``
``CheckpointManager(saver_impl=…)`` → ``save(components, path, metadata)``.

The key layout is a public id (ECOSYSTEM §6, REQUEST-16) and lives in ONE
place, ``lora_keys.py`` — this module only extracts the PEFT adapters, hands
them to the map, attaches the shared metadata and writes the file. Every
metadata key with one correct spelling across savers is IMPORTED from
``app.engine.utils.lora_metadata`` (the ltx2 lesson: a re-typed map drifts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import torch

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.lora_metadata import kohya_config_metadata, trigger_metadata
from app.engine.utils.safe_save import safe_save_file

from .lora_keys import ARTIFACT_LAYOUT_VERSION, adapters_from_peft_model, build_artifact

logger = structlog.get_logger(__name__)

ARCHITECTURE = "minimax-h3"

_SAVE_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class MiniMaxH3Saver(ModelSaver):
    """Write the PEFT adapters as a ``diffusion_model.``-prefixed, native-named,
    scale-folded safetensors file. A save that cannot produce the artifact
    RAISES — the manager logs it and, on the final save, fails the job — never
    a silent no-file success."""

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        model = components.get("unet")
        if model is None:
            model = components.get("transformer")
        if model is None:
            raise ValueError("minimax_h3 saver: no 'unet' / 'transformer' component to save")
        if not hasattr(model, "peft_config"):
            raise ValueError("minimax_h3 saver: the transformer is not a PEFT model — nothing to extract")

        adapters = adapters_from_peft_model(model)
        if not adapters:
            raise ValueError("minimax_h3 saver: the PEFT model carries no LoRA adapters")
        artifact = build_artifact(adapters)

        config = components.get("config")
        if not isinstance(config, dict):
            config = {}
        peft_cfg = next(iter(model.peft_config.values()), None)
        rank = int(getattr(peft_cfg, "r", 0) or 0)

        save_metadata: dict[str, str] = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "modelspec.architecture": ARCHITECTURE,
            "minimax_h3.layout_version": str(ARTIFACT_LAYOUT_VERSION),
        }
        # The scaling is folded into lora_B, so the artifact's effective alpha
        # IS its rank (scale 1): the header describes the file, not the run.
        save_metadata.update(kohya_config_metadata(config, rank=rank, alpha=float(rank)))
        save_metadata.update(trigger_metadata(config))
        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        precision = str(config.get("save_precision", "bf16")).lower()
        dtype = _SAVE_DTYPES.get(precision, torch.bfloat16)
        tensors = {k: v.to(dtype).contiguous() for k, v in artifact.items()}

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_save_file(tensors, str(path), metadata=save_metadata)
        logger.info(
            "minimax_h3_save_lora",
            path=str(path),
            num_tensors=len(tensors),
            adapters=len(adapters),
            rank=rank,
            save_dtype=str(dtype),
        )
