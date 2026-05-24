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


def test_driver_reports_no_vae_no_text_encoder():
    """Pixel-space unified model — VAE and TE are intentionally None.

    HiDream-O1 has no separate VAE or text encoder. The base
    ``GenericTrainingPipeline`` is patched in Task 10 to tolerate this.
    """
    from app.engine.models.families.hidream_o1.driver import HiDreamO1Driver

    definition = ModelDefinition(id="x", family="hidream_o1", name="X")
    driver = HiDreamO1Driver(definition, device=torch.device("cpu"))

    # Before assign_components — all None / not yet wired
    assert driver.get_text_encoders() == {}
    assert driver.get_vae() is None

    # After assigning the unified model
    import torch.nn as nn
    fake_model = nn.Linear(4, 4)
    driver.assign_components({"unet": fake_model})

    assert driver.get_primary_model() is fake_model
    assert driver.get_text_encoders() == {}
    assert driver.get_vae() is None
