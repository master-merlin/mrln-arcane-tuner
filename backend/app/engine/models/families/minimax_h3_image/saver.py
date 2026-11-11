"""MiniMax-H3-Image LoRA saver: the model's native key layout.

Called through ``MiniMaxH3ImageDriver.get_saver()`` by the checkpoint manager as
``save(components, path, metadata)``.

The key layout lives in one place, ``lora_keys.py``. This module extracts the PEFT adapters,
hands them to that map, attaches the shared metadata (every key with one correct spelling across
savers comes from ``app.engine.utils.lora_metadata``, the trigger word included) and writes the
file.
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

ARCHITECTURE = "minimax-h3-image"

_SAVE_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class MiniMaxH3ImageSaver(ModelSaver):
    """Write the PEFT adapters as a ``diffusion_model.``-prefixed, native-named,
    scale-folded safetensors file. A save that cannot produce the file raises (the
    manager logs it and fails the job on the final save); it never succeeds silently
    with no file."""

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
            raise ValueError("minimax_h3_image saver: no 'unet' / 'transformer' component to save")
        if not hasattr(model, "peft_config"):
            raise ValueError("minimax_h3_image saver: the transformer is not a PEFT model, nothing to extract")

        adapters = adapters_from_peft_model(model)
        if not adapters:
            raise ValueError("minimax_h3_image saver: the PEFT model carries no LoRA adapters")
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
            "minimax_h3_image.layout_version": str(ARTIFACT_LAYOUT_VERSION),
        }
        # The scaling is folded into lora_B, so the file's effective alpha is its
        # rank (scale 1): the header describes the file, not the run.
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
            "minimax_h3_image_save_lora",
            path=str(path),
            num_tensors=len(tensors),
            adapters=len(adapters),
            rank=rank,
            save_dtype=str(dtype),
        )
