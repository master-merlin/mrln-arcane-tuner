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
from app.engine.core.caption_target import resolve_caption_target

logger = get_logger(__name__)


def _emit_suggestion_written(
    *,
    dataset_name: str,
    stem: str,
    definition_id: str,
    target: str,
    suggestion: str,
) -> None:
    """Broadcast a ``suggestion.written`` event so the frontend review updates
    live (no re-navigation). Mirrors ``caption_batch._emit_caption_written``:
    the worker runs its own ``asyncio.run`` loop on a background thread, so the
    broadcast MUST be scheduled onto the main app loop (where the WS
    connections live) via ``run_coroutine_threadsafe`` — awaiting it on the
    worker's loop would never reach connected clients. No-op if no loop yet."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "stem": stem,
        "definition_id": definition_id,
        "target": target,
        "suggestion": suggestion,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("suggestion.written", payload),
        loop,
    )


def _emit_variant_written(
    *,
    dataset_name: str,
    stem: str,
    definition_id: str,
    target: str,
) -> None:
    """Broadcast a ``variant.written`` event after an auto-accepted refine
    promotes a caption straight to the live variant — so the Browse grid + the
    details editor refresh without re-navigation. Same cross-loop scheduling
    rationale as :func:`_emit_suggestion_written`. No-op if no loop yet."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "stem": stem,
        "definition_id": definition_id,
        "target": target,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("variant.written", payload),
        loop,
    )


def run_caption_refine_batch(
    task_id: str,
    *,
    dataset_name: str,
    image_rel_paths: list[str],
    definition_id: str,
    preset: str,
    model: str,
    base_url: str = "http://localhost:11434",
    target: str = "original",
    style: str = "auto",
    auto_accept: bool = False,
) -> None:
    ds = dataset_manager.get_dataset(dataset_name)
    if ds is None:
        task_manager.fail(task_id, f"Dataset '{dataset_name}' not found.")
        return

    ds_path = ds.path
    client = OllamaClient(base_url=base_url)

    # Resolve the model's caption target ONCE (constant per definition) so each
    # caption is refined with a style + token-budget the model actually
    # understands. If the definition can't be resolved, fall back to the legacy
    # preset prompt (system_prompt=None) rather than crashing the batch.
    try:
        cap_target = resolve_caption_target(definition_id)
    except Exception:
        logger.warning("refine_caption_target_unresolved", definition_id=definition_id)
        cap_target = None

    # Resolve the caption format ONCE for structured-output validation/normalisation.
    from app.core.captioning.formats import get_caption_format_for_definition

    fmt = get_caption_format_for_definition(definition_id)

    async def _run() -> tuple[int, int, str | None]:
        """Refine every image; return ``(ok, failed, last_error)``.

        The counters are RETURNED rather than kept private because the caller
        decides the task's terminal state from them — see
        ``TaskManager.finish_batch``. Before LANE-52 this coroutine returned
        None and the caller called ``complete()`` unconditionally, so a batch in
        which every single item raised still reported success."""
        ok = 0
        failed = 0
        last_error: str | None = None
        for i, rel in enumerate(image_rel_paths):
            if task_manager.is_cancelled(task_id):
                break
            stem = Path(rel).stem
            masked = target == "masked"
            source = caption_variants.resolve_caption(
                ds_path, stem, None, masked=masked
            )
            system_prompt = (
                caption_refine.build_refine_system_prompt(cap_target, preset, style)
                if cap_target is not None
                else None
            )
            try:
                refined = await caption_refine.refine_caption(
                    client, model, source, preset, system_prompt=system_prompt
                )
                if fmt.is_structured:
                    if not fmt.detect(refined):
                        logger.warning("refine_structured_parse_failed", rel=rel)
                        failed += 1
                        last_error = (
                            f"model output was not valid {fmt.id} for {rel}"
                        )
                        task_manager.update(
                            task_id, current=i + 1, item=rel, ok=ok, failed=failed
                        )
                        continue
                    refined = fmt.serialize(fmt.parse_and_normalize(refined))
                caption_suggestions.write_suggestion(
                    ds_path, definition_id, stem, refined, masked=masked
                )
                ok += 1
                if auto_accept:
                    # Promote straight to the live variant (snapshot + clear the
                    # suggestion) — no review step — and notify the grid/editor.
                    caption_suggestions.accept_suggestion(
                        ds_path, definition_id, stem, masked=masked
                    )
                    _emit_variant_written(
                        dataset_name=dataset_name,
                        stem=stem,
                        definition_id=definition_id,
                        target=target,
                    )
                else:
                    _emit_suggestion_written(
                        dataset_name=dataset_name,
                        stem=stem,
                        definition_id=definition_id,
                        target=target,
                        suggestion=refined,
                    )
            except Exception as exc:
                logger.exception("caption_refine_failed", rel=rel)
                failed += 1
                # An httpx.ReadTimeout stringifies to "" — a bare str(exc) would
                # hand the user a task that failed for no stated reason, which is
                # the same silence one layer down. Fall back to the class name.
                last_error = str(exc) or type(exc).__name__
            task_manager.update(task_id, current=i + 1, item=rel, ok=ok, failed=failed)
        return ok, failed, last_error

    try:
        ok, failed, last_error = asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        logger.exception("caption_refine_batch_failed")
        task_manager.fail(task_id, str(e) or type(e).__name__)
        return
    if task_manager.is_cancelled(task_id):
        task_manager.finish_cancelled(task_id)
        return
    task_manager.finish_batch(task_id, ok=ok, failed=failed, error=last_error)
