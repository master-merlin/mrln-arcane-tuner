"""Shared GPU-plugin unload helper.

``CaptionService``, ``MaskingService``, and ``ScoringService`` each manage a
dict of GPU-backed model plugins with the same unload shape: unload every
loaded plugin, clear the "active model" bookkeeping, then
``gc.collect()`` + ``torch.cuda.synchronize()`` + ``torch.cuda.empty_cache()``
so VRAM is actually released before the next load. This shape had been
triplicated across the three services and had drifted — masking's copy was
missing ``torch.cuda.synchronize()`` (now fixed by routing through here; see
P2c / B-CLEAN-9).
"""
from __future__ import annotations

import gc
from collections.abc import Iterable

import structlog
import torch

logger = structlog.get_logger(__name__)


def gpu_batch_active(task_types: Iterable[str]) -> bool:
    """True if any task of *task_types* is PENDING or RUNNING.

    The shared "is someone else using this GPU model right now?" predicate for
    the three GPU-plugin services' ``skip_if_batch_active`` mode. Each service
    owns its own tuple of task types (with the evidence for what is in and what
    is out on that constant); this only answers the question.

    Imported lazily inside the function: ``task_manager`` pulls in the whole
    task/event stack, and this module is imported by services that are
    themselves imported at startup (ARCHITECTURE D1).

    Callers MUST hold their service's ``_unload_lock`` across this check AND
    the unload — on its own this is a read, not a guard.
    """
    from app.core.tasks.task import TaskStatus
    from app.core.tasks.task_manager import task_manager

    wanted = set(task_types)
    return any(
        t.type in wanted and t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        for t in task_manager.list()
    )


def unload_gpu_plugins(
    owner: object,
    *,
    plugins: dict,
    active_attr: str,
    service_label: str,
) -> None:
    """Unload every plugin in *plugins*, reset ``owner.<active_attr>``, and
    release CUDA memory.

    ``owner`` is whatever carries the "active model" bookkeeping attribute —
    a classmethod-style caller passes ``cls``, an instance-method caller may
    pass ``self`` or ``self.__class__`` (both resolve the same class-level
    attribute for these singleton services).

    Logs ``unloading_{service_label}_models`` (only when a model was active)
    before unloading and ``all_{service_label}_models_unloaded`` after,
    matching each service's pre-existing structured-log keys.
    """
    active_key = getattr(owner, active_attr, None)
    if active_key:
        logger.info(f"unloading_{service_label}_models", active_model=active_key)

    for plugin in plugins.values():
        plugin.unload()

    setattr(owner, active_attr, None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logger.info(f"all_{service_label}_models_unloaded")
