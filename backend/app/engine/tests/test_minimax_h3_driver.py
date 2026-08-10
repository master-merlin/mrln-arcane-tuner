"""minimax_h3 family registration + capability flags + driver non-training surface."""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry


def _reload_registry() -> type[ModelRegistry]:
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()
    return ModelRegistry


def test_family_is_auto_discovered():
    assert "minimax_h3" in _reload_registry()._families


def test_capability_flags_declare_video_and_audio():
    family_cls = _reload_registry()._families["minimax_h3"]
    caps = family_cls.capability_overrides
    assert caps["is_video"] is True
    assert caps["has_audio"] is True
    assert caps["has_image_encoder"] is True
    # H3 is single-stream: there is no second expert to schedule.
    assert caps["dual_expert"] is False
    # supports_train_te is intentionally NOT asserted here: only sdxl may put
    # that key in capability_overrides (test_only_sdxl_overrides_train_te);
    # minimax_h3 relies on the latent_diffusion archetype's False default.
    # The 48 GB Qwen3-VL TE must be cacheable.
    assert caps["te_cache"] is True


# ---------------------------------------------------------------------------
# Task 6: driver non-training surface
# ---------------------------------------------------------------------------

def _driver(def_id: str = "minimax-h3-t2va"):
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return MiniMaxH3Driver(ModelRegistry._definitions[def_id], {})


def test_definition_ships_curated_target_list_matching_driver():
    """The nucleus_image contract: YAML and driver must agree EXACTLY.

    If they drift, the introspector's exhaustive catalog silently overwrites
    the curated list at first model load and the run adapts the wrong modules.
    """
    import pathlib
    import yaml

    defs_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "models" / "families" / "minimax_h3" / "definitions"
    )
    for path in defs_dir.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        driver = _driver(data["id"])
        assert sorted(driver.get_lora_targets()) == sorted(
            data["lora_targetable_modules"]
        ), f"{data['id']}: driver/YAML target-list drift"


def test_block_topology_is_read_from_the_definition():
    topo = _driver().get_block_topology()
    assert sum(entry["count"] for entry in topo) == 52


def test_init_scheduler_still_returns_none():
    """The deliberate exception to "raise, never a plausible default":
    ``None`` IS the real answer (H3 trains with flow matching, no external
    scheduler) — and the ONLY body ``hook_dispatch.TRIVIAL_BODIES`` treats as
    the ``init_scheduler`` no-op baseline. Any other body (including a raise)
    would enroll minimax_h3 in the auto-delegation allowlist and trip
    ``test_autodelegated_family_hook_set_is_exactly_expected``. See
    ``driver.py``'s module docstring."""
    assert _driver().init_scheduler() is None


def test_forward_pass_refuses_loudly_in_pr0():
    import pytest

    with pytest.raises(NotImplementedError, match="PR1"):
        _driver().forward_pass(None, None, None, {})


def test_get_saver_refuses_loudly_in_pr0():
    import pytest

    with pytest.raises(NotImplementedError, match="PR1"):
        _driver().get_saver()


def test_encode_text_refuses_loudly_in_pr0():
    import pytest
    import torch

    with pytest.raises(NotImplementedError, match="PR1"):
        _driver().encode_text([], torch.bfloat16)


def test_resolve_loading_dtype_is_bf16():
    import torch

    assert _driver().resolve_loading_dtype() is torch.bfloat16


def test_get_te_lora_targets_is_empty_te_never_trains():
    assert _driver().get_te_lora_targets() == []


def test_assign_components_wires_the_five_manifest_keys():
    driver = _driver()
    components = {
        "tokenizer": object(),
        "text_encoder": object(),
        "vae": object(),
        "audio_vae": object(),
        "transformer": object(),
    }
    driver.assign_components(components)
    assert driver.get_components() is components
    assert driver.get_primary_model() is components["transformer"]
    assert driver.get_text_encoders() == {"text_encoder": components["text_encoder"]}
