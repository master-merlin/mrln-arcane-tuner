"""Smoke tests for the HiDream-O1 model family.

Validates wiring (registration, definition loading, class shapes) without
touching the actual 8B checkpoint — safe to run on CI / CPU-only boxes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry


# ── Registration ─────────────────────────────────────────────────────────

def test_family_is_discovered():
    """``hidream_o1`` registers itself via the auto-discovery scan."""
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("hidream_o1")
    assert family_cls.family_name == "hidream_o1"


import torch


def test_loader_manifest_is_single_unified_model():
    """The unified architecture exposes one component, mapped to 'unet'.

    HiDream-O1 is a pixel-space Unified Transformer — no separate VAE,
    no separate text encoder. The loader's manifest reflects that with
    a single ``ComponentSpec`` keyed ``"unet"`` so ``GenericTrainingPipeline``
    plumbing recognizes the family has one primary model.
    """
    from app.engine.models.families.hidream_o1.loader import HiDreamO1Loader

    definition = ModelDefinition(id="x", family="hidream_o1", name="X")
    loader = HiDreamO1Loader(device=torch.device("cpu"))
    manifest = loader.get_component_manifest(definition)

    assert len(manifest) == 1
    spec = manifest[0]
    assert spec.key == "unet"
