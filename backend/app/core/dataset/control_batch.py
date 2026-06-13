"""Degradation batch — produce control ("before") images for edit datasets.

The #1 community-wished feature for edit-model training is *producing* pairs:
take each target image and synthesize a degraded source so the LoRA learns the
inverse edit (grayscale→color, blurry→sharp, low-res→hi-res, noisy→clean…).

Everything here is PIL/numpy only — no GPU — so the worker runs on the
non-GPU ``background`` lane and never blocks a training/caption/mask task.

Layering (each independently testable):
  ``apply_degradations(img, ops)``  — pure: PIL image → degraded PIL image
  ``generate_controls(dataset_path, targets, ...)`` — pure-ish disk core:
        degrade each target's root image into a control slot; skip-existing
        unless ``overwrite``; returns a ``{ok, skipped, failed, written}`` summary
  ``run_control_batch(task_id, ...)`` — the Task-Center worker that resolves
        the dataset, drives progress, refreshes control metadata, emits the
        summary event and finalizes the task

Module-level seams (monkeypatchable in tests):
  ``_emit_control_summary(**kw)`` → broadcast ``control.generate_summary``
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from PIL import Image, ImageFilter

from app.core.dataset.control_helpers import (
    CONTROL_IMAGE_EXTS,
    control_slot_dir_name,
    detect_control_slots,
    prepare_control_slot_path,
)
from app.core.logger import get_logger
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Degradation ops (pure) ──────────────────────────────────────────────────


def _op_grayscale(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Drop colour — the canonical colourize-training source."""
    return img.convert("L").convert("RGB")


def _op_blur(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Gaussian blur (``radius`` px, default 2.0) — deblur/sharpen training."""
    radius = float(params.get("radius", 2.0))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _op_downscale(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Detail loss via downscale-then-upscale (``factor``, default 2.0).

    Keeps the output dimensions identical to the target so the pair buckets
    together — only high-frequency detail is destroyed (super-resolution
    training source).
    """
    factor = float(params.get("factor", 2.0))
    if factor <= 1.0:
        return img
    w, h = img.size
    small = img.resize(
        (max(1, int(round(w / factor))), max(1, int(round(h / factor)))),
        Image.BICUBIC,
    )
    return small.resize((w, h), Image.BICUBIC)


def _op_noise(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Additive Gaussian noise. ``sigma`` is a 0..1 fraction of full scale
    (default 0.05). ``seed`` makes a single op reproducible (used in tests)."""
    import numpy as np

    sigma = float(params.get("sigma", params.get("amount", 0.05))) * 255.0
    seed = params.get("seed")
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    arr = np.asarray(img, dtype=np.float32)
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))


_DEGRADATION_OPS: dict[str, Callable[[Image.Image, dict[str, Any]], Image.Image]] = {
    "grayscale": _op_grayscale,
    "blur": _op_blur,
    "downscale": _op_downscale,
    "noise": _op_noise,
}

# Op names accepted by the API (validated at the schema layer too).
DEGRADATION_OP_TYPES: tuple[str, ...] = tuple(_DEGRADATION_OPS)


def apply_degradations(img: Image.Image, ops: list[dict[str, Any]]) -> Image.Image:
    """Apply a sequence of degradation ops to a copy of ``img`` (→ RGB).

    Raises ``ValueError`` on an unknown op type so a bad request fails the
    whole batch loudly rather than silently writing identical controls.
    """
    out = img.convert("RGB")
    for op in ops:
        op_type = op.get("type")
        fn = _DEGRADATION_OPS.get(op_type)
        if fn is None:
            raise ValueError(f"unknown degradation op: {op_type!r}")
        out = fn(out, op.get("params") or {})
    return out


def _save_control(img: Image.Image, dest: str) -> None:
    ext = os.path.splitext(dest)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.save(dest, quality=95)
    else:
        img.save(dest)


# ── Disk core (pure-ish; no task_manager / dataset_manager) ─────────────────


def generate_controls(
    dataset_path: str,
    targets: list[tuple[str, str]],
    *,
    slot_index: int,
    ops: list[dict[str, Any]],
    overwrite: bool = False,
    progress: Callable[[int, str, int, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Degrade each target's root image into the given control slot.

    ``targets`` is a list of ``(root_rel_path, stem)``. Existing controls in
    the slot are skipped unless ``overwrite``. Returns a summary dict with
    ``ok`` / ``skipped`` / ``failed`` counts and the list of ``written``
    control rel-paths (used to refresh control metadata afterward).
    """
    slot_name = control_slot_dir_name(slot_index)
    ok = skipped = failed = 0
    written: list[str] = []

    for i, (root_rel, stem) in enumerate(targets):
        if is_cancelled and is_cancelled():
            break
        try:
            existing = detect_control_slots(dataset_path, stem).get(slot_name)
            if existing and not overwrite:
                skipped += 1
            else:
                ext = os.path.splitext(root_rel)[1].lower()
                if ext not in CONTROL_IMAGE_EXTS:
                    ext = ".jpg"
                dest = prepare_control_slot_path(dataset_path, slot_index, stem, ext)
                with Image.open(os.path.join(dataset_path, root_rel)) as im:
                    _save_control(apply_degradations(im, ops), dest)
                written.append(f"{slot_name}/{stem}{ext}")
                ok += 1
        except Exception as exc:  # noqa: BLE001 — isolate per-item failures
            failed += 1
            logger.warning(
                "control_generate_item_failed",
                dataset_path=dataset_path, stem=stem, error=str(exc),
            )
        if progress:
            progress(i + 1, root_rel, ok, skipped, failed)

    return {"ok": ok, "skipped": skipped, "failed": failed, "written": written}


# ── Summary event ───────────────────────────────────────────────────────────


def _emit_control_summary(
    *, dataset_name: str, slot: int, ok: int, skipped: int, failed: int
) -> None:
    """Broadcast a one-off ``control.generate_summary`` event (no-op pre-loop)."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "slot": slot,
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("control.generate_summary", payload), loop,
    )


# ── Worker ────────────────────────────────────────────────────────────────


def run_control_batch(
    task_id: str,
    *,
    dataset_name: str,
    slot: int,
    ops: list[dict[str, Any]],
    overwrite: bool = False,
    stems: list[str] | None = None,
) -> None:
    """Synchronous worker — runs on the ``background`` lane thread.

    Resolves the dataset, degrades each (optionally filtered) target into the
    control slot, then refreshes the affected pairs' control metadata so
    ``/pairs`` reflects the new controls. ``dataset.invalidated`` fires
    automatically on finish (any dataset-scoped task), refreshing the grid.
    """
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        task_manager.fail(task_id, f"Dataset '{dataset_name}' not found")
        return

    wanted = set(stems) if stems else None
    targets: list[tuple[str, str]] = []
    for key in dataset.media_metadata or {}:
        stem = os.path.splitext(os.path.basename(key))[0]
        if wanted is not None and stem not in wanted:
            continue
        targets.append((key, stem))

    def _progress(current: int, item: str, ok: int, skipped: int, failed: int) -> None:
        task_manager.update(
            task_id, current=current, item=item, ok=ok, failed=skipped + failed
        )

    try:
        summary = generate_controls(
            dataset.path, targets,
            slot_index=slot, ops=ops, overwrite=overwrite,
            progress=_progress,
            is_cancelled=lambda: task_manager.is_cancelled(task_id),
        )
    except Exception as exc:  # noqa: BLE001 — unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return

    for rel in summary["written"]:
        stem = os.path.splitext(os.path.basename(rel))[0]
        try:
            dm.refresh_control_metadata(dataset_name, stem)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "control_metadata_refresh_failed",
                dataset_name=dataset_name, stem=stem, error=str(exc),
            )

    _emit_control_summary(
        dataset_name=dataset_name, slot=slot,
        ok=summary["ok"], skipped=summary["skipped"], failed=summary["failed"],
    )

    if task_manager.is_cancelled(task_id):
        task_manager._finish(task_id, TaskStatus.CANCELLED)
    else:
        task_manager.complete(task_id)
