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

import structlog
import torch

logger = structlog.get_logger(__name__)


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
