"""Background worker that LLM-refines each image's caption into a per-definition suggestion.

Runs on the CPU/background lane (inference is offloaded to the Ollama server). For each
image it resolves the general caption, refines it with the chosen preset, and writes a
pending suggestion (never the live variant — the user accepts via the suggestion routes).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.captioning import caption_suggestions
from app.core.captioning import caption_variants
from app.core.dataset_manager import dataset_manager
from app.core.llm import caption_refine
from app.core.llm.ollama_client import OllamaClient
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


def run_caption_refine_batch(
    task_id: str,
    *,
    dataset_name: str,
    image_rel_paths: list[str],
    definition_id: str,
    preset: str,
    model: str,
    base_url: str = "http://localhost:11434",
) -> None:
    ds = dataset_manager.get_dataset(dataset_name)
    if ds is None:
        task_manager.fail(task_id, f"Dataset '{dataset_name}' not found.")
        return

    ds_path = ds.path
    client = OllamaClient(base_url=base_url)

    async def _run() -> None:
        ok = 0
        failed = 0
        for i, rel in enumerate(image_rel_paths):
            if task_manager.is_cancelled(task_id):
                break
            stem = Path(rel).stem
            source = caption_variants.resolve_caption(ds_path, stem, None)
            try:
                refined = await caption_refine.refine_caption(client, model, source, preset)
                caption_suggestions.write_suggestion(ds_path, definition_id, stem, refined)
                ok += 1
            except Exception:
                logger.exception("caption_refine_failed", rel=rel)
                failed += 1
            task_manager.update(task_id, current=i + 1, item=rel, ok=ok, failed=failed)

    try:
        asyncio.run(_run())
        task_manager.complete(task_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("caption_refine_batch_failed")
        task_manager.fail(task_id, str(e))
