"""Tests for the longcat_image family (diffusers-0.39-native LongCat-Image).

TDD order (F2 plan):
  Task 1: family registration + definition
  Task 2: loader manifest (incl. the extra ``text_processor`` component)
  Task 3: driver — LoRA targets vs a tiny instantiated transformer,
          timestep ÷1000 exactly once, flow-match target, encode_text
          replicating ``LongCatImagePipeline._encode_prompt``
  Task 4: trainer — override trio + TE disk-cache layout
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.engine.core.definitions import ModelDefinition


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Snapshot + restore ``ModelRegistry`` class state around every test.

    Registration tests mutate the registry's class-level discovery caches
    inline (forcing a re-scan); left unrestored those mutations leak into
    later tests in the session (mirrors test_krea2_family.py's fixture).
    """
    from app.engine.models.registry import ModelRegistry

    saved = {
        "_families": dict(ModelRegistry._families),
        "_definitions": dict(ModelRegistry._definitions),
        "_paths": dict(ModelRegistry._paths),
        "_discovered": ModelRegistry._discovered,
        "_definitions_loaded": ModelRegistry._definitions_loaded,
    }
    try:
        yield
    finally:
        ModelRegistry._families = saved["_families"]
        ModelRegistry._definitions = saved["_definitions"]
        ModelRegistry._paths = saved["_paths"]
        ModelRegistry._discovered = saved["_discovered"]
        ModelRegistry._definitions_loaded = saved["_definitions_loaded"]


def _make_definition(**kwargs) -> MagicMock:
    """Build a mock longcat_image ModelDefinition."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "longcat_image"
    definition.id = kwargs.get("id", "longcat-image-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family registration + definition ────────────────────────────────


def test_family_registered():
    """longcat_image family must appear in ModelRegistry as latent_diffusion."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("longcat_image")
    assert fam is not None, "longcat_image family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """longcat-image-base definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    defn = ModelRegistry.get_definition("longcat-image-base")
    assert defn is not None, "longcat-image-base definition not loaded"
    assert defn.family == "longcat_image"

    # Canonical checkpoint repo
    repo = defn.components["repo"]
    repo_path = repo["path"] if isinstance(repo, dict) else repo.path
    assert repo_path == "huggingface:meituan-longcat/LongCat-Image", (
        f"wrong repo path: {repo_path}"
    )

    # Verified transformer config facts (diffusers 0.39 defaults)
    arch = defn.architecture_params
    assert arch.get("transformer.num_layers") == 19
    assert arch.get("transformer.num_single_layers") == 38
    assert arch.get("transformer.joint_attention_dim") == 3584
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("te.max_length") == 512
