"""boogu_image definition-YAML pins (Task 2).

Guards against the dreamlite 2026-07-08 GPU-UAT precedent: a definition
whose ``lora_targetable_modules`` doesn't EXACTLY match what the real
vendored transformer offers silently either (a) starts empty and gets
overwritten by the introspector's exhaustive catalog at first model load
(``registry.enrich_definition``), or (b) omits real weight-bearing
Linears — here specifically the double-stream blocks' PROCESSOR-owned
img_to_q/img_to_k/... (the module-level ``attn.{to_q,to_k,to_v}`` are
``del``eted and re-homed on ``img_instruct_attn.processor``; see
``BooguImageDoubleStreamTransformerBlock`` in
``vendor/models/transformers/transformer_boogu.py``), leaving the
joint-attention path un-adapted at training time.

Method (dreamlite/ovis precedent, per task-2-brief.md): instantiate the
vendored transformer TINY on CPU, walk ``named_modules()`` to discover the
real per-block-type attention+feed-forward Linear suffix set, expand across
the REAL checkpoint's block counts (8 double-stream + 32 single-stream + 2
each of noise/ref_image/context refiner — the refiner depth is the vendored
class's ``num_refiner_layers`` DEFAULT; the real checkpoint's exact refiner
depth is not stated anywhere in the task-2 brief or the upstream code drop,
see task-2-report.md), and assert the shipped YAML matches EXACTLY.
"""

from __future__ import annotations

import pathlib

import torch.nn as nn

from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
)
from app.engine.models.registry import ModelRegistry

# Real checkpoint block counts (verified facts, task-2-brief.md).
NUM_DOUBLE_STREAM = 8
NUM_SINGLE_STREAM = 32
# ASSUMED: vendored BooguImageTransformer2DModel.__init__'s num_refiner_layers
# class DEFAULT (=2) — the brief states "3 refiner pairs noise/ref_image/
# context" but does not give a per-refiner depth, and the upstream code drop
# ships no checkpoint config.json to cross-check against. See task-2-report.md.
NUM_REFINER_LAYERS = 2

DEF_IDS = ("boogu-image-base", "boogu-image-turbo", "boogu-image-edit")

_DEFINITIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]  # .../app/engine
    / "models" / "families" / "boogu_image" / "definitions"
)


def _reload_registry() -> ModelRegistry:
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._definitions_loaded = False
    registry = ModelRegistry()
    registry.initialize()
    return registry


def _build_tiny_transformer() -> BooguImageTransformer2DModel:
    """Same tiny divisibility-respecting config as test_boogu_image_vendor.py."""
    model = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=None,
        hidden_size=16,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=8,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        axes_dim_rope=(2, 2, 4),
        axes_lens=(64, 64, 64),
        instruction_feature_configs=dict(
            instruction_feat_dim=8, reduce_type="mean", num_instruction_feat_layers=1,
        ),
        prompt_tuning_configs=dict(use_prompt_tuning=False),
        timestep_scale=1000.0,
    )
    model.eval()
    return model


def _is_modulation_linear(name: str) -> bool:
    """True for AdaLN modulation projections (e.g. ``norm1.linear``,
    ``img_norm2.linear``, ``instruct_norm1.linear``) — excluded from the
    curated LoRA surface, matching the house convention: krea2/ovis/
    dreamlite's curated lists are attention+feed-forward ONLY, never the
    modulation/embedder/output-norm Linears.

    Double-stream modulation norms are PREFIXED (``img_norm1``,
    ``instruct_norm2``, ...), not just ``norm*`` (single-stream/refiner use
    bare ``norm1``) — must check "norm" appears anywhere in the parent
    segment, not just as a prefix.
    """
    parts = name.split(".")
    return len(parts) >= 2 and parts[-1] == "linear" and "norm" in parts[-2]


def _attn_ff_suffixes(container: nn.Module) -> list[str]:
    """Real attention+feed-forward Linear suffixes for one block instance."""
    return sorted(
        name
        for name, mod in container.named_modules()
        if isinstance(mod, nn.Linear) and name and not _is_modulation_linear(name)
    )


def _expected_lora_targets() -> set[str]:
    """Derive the curated LoRA target set fresh from the vendored code."""
    model = _build_tiny_transformer()

    single_stream_suffixes = _attn_ff_suffixes(model.single_stream_layers[0])
    double_stream_suffixes = _attn_ff_suffixes(model.double_stream_layers[0])
    noise_refiner_suffixes = _attn_ff_suffixes(model.noise_refiner[0])
    ref_image_refiner_suffixes = _attn_ff_suffixes(model.ref_image_refiner[0])
    context_refiner_suffixes = _attn_ff_suffixes(model.context_refiner[0])

    # Refiners share BooguImageTransformerBlock with single_stream_layers —
    # noise_refiner/ref_image_refiner are modulated (norm1.linear present but
    # excluded above), context_refiner is not (modulation=False) — the
    # attn+FF suffix SET must be identical regardless.
    assert (
        single_stream_suffixes
        == noise_refiner_suffixes
        == ref_image_refiner_suffixes
        == context_refiner_suffixes
    )

    targets: set[str] = set()
    for i in range(NUM_DOUBLE_STREAM):
        for suf in double_stream_suffixes:
            targets.add(f"double_stream_layers.{i}.{suf}")
    for i in range(NUM_SINGLE_STREAM):
        for suf in single_stream_suffixes:
            targets.add(f"single_stream_layers.{i}.{suf}")
    for container_name in ("noise_refiner", "ref_image_refiner", "context_refiner"):
        for i in range(NUM_REFINER_LAYERS):
            for suf in single_stream_suffixes:
                targets.add(f"{container_name}.{i}.{suf}")

    return targets


class TestCuratedLoraTargetList:
    """Pins the curated ``lora_targetable_modules`` shipped in both definitions."""

    def test_expected_target_count_is_418(self):
        # 8 double-stream x 19 + 32 single-stream x 7 + 3 refiner types x 2 x 7
        # = 152 + 224 + 42 = 418 real weight-bearing Linears.
        expected = _expected_lora_targets()
        assert len(expected) == 418

    def test_definitions_are_registered(self):
        registry = _reload_registry()
        for def_id in DEF_IDS:
            assert def_id in registry._definitions, f"{def_id} not registered"
            assert registry._definitions[def_id].family == "boogu_image"

    def test_both_definitions_ship_curated_list_matching_tiny_model_expansion(self):
        registry = _reload_registry()
        expected = _expected_lora_targets()
        for def_id in DEF_IDS:
            defn = registry._definitions[def_id]
            shipped = set(defn.lora_targetable_modules or [])
            assert shipped, f"{def_id}: must ship a non-empty curated list"
            assert shipped == expected, (
                f"{def_id}: shipped list diverges from the curated/tested surface "
                f"(+{len(shipped - expected)} extra, -{len(expected - shipped)} missing). "
                f"Missing e.g. {sorted(expected - shipped)[:3]}; "
                f"extra e.g. {sorted(shipped - expected)[:3]}"
            )

    def test_both_definitions_ship_identical_curated_lists(self):
        registry = _reload_registry()
        base = set(registry._definitions["boogu-image-base"].lora_targetable_modules)
        turbo = set(registry._definitions["boogu-image-turbo"].lora_targetable_modules)
        assert base == turbo

    def test_edit_definition_ships_identical_curated_list_to_base(self):
        """Edit's checkpoint transformer/config.json is byte-identical to
        Base's (a4-report.md recon) — same 418-module curated surface."""
        registry = _reload_registry()
        base = set(registry._definitions["boogu-image-base"].lora_targetable_modules)
        edit = set(registry._definitions["boogu-image-edit"].lora_targetable_modules)
        assert base == edit

    def test_processor_owned_double_stream_names_present(self):
        """The load-bearing regression this task guards against: a curated
        list missing the processor-owned joint-attention projections
        silently leaves the double-stream cross-modal path un-adapted."""
        registry = _reload_registry()
        expected_processor_suffixes = [
            "img_instruct_attn.processor.img_to_q",
            "img_instruct_attn.processor.img_to_k",
            "img_instruct_attn.processor.img_to_v",
            "img_instruct_attn.processor.instruct_to_q",
            "img_instruct_attn.processor.instruct_to_k",
            "img_instruct_attn.processor.instruct_to_v",
            "img_instruct_attn.processor.img_out",
            "img_instruct_attn.processor.instruct_out",
        ]
        for def_id in DEF_IDS:
            shipped = set(registry._definitions[def_id].lora_targetable_modules)
            for suffix in expected_processor_suffixes:
                assert any(name.endswith(suffix) for name in shipped), (
                    f"{def_id}: missing processor-owned name ending {suffix!r}"
                )
            # Sanity-check a full block-0 path too.
            assert (
                "double_stream_layers.0.img_instruct_attn.processor.img_to_q"
                in shipped
            )
            # The retained stock joint-attention OUTPUT projection (fed by
            # processor.img_out/instruct_out — see attention_processor.py's
            # ``hidden_states = attn.to_out[0](hidden_states)``) must also be
            # present, or the joint-attention output path stays un-adapted.
            assert "double_stream_layers.0.img_instruct_attn.to_out.0" in shipped

    def test_gqa_widths_documented_in_yaml_comments(self):
        # num_kv_heads=7, attention_head_dim=120 -> to_k/to_v out width 840;
        # num_attention_heads=28 -> to_q/to_out width 3360. Both numbers must
        # appear in a comment (house convention per dreamlite base.yaml's MQA
        # width comment).
        for filename in ("base.yaml", "turbo.yaml", "edit.yaml"):
            text = (_DEFINITIONS_DIR / filename).read_text()
            assert "840" in text, f"{filename}: GQA to_k/to_v width (840) not documented"
            assert "3360" in text, f"{filename}: to_q/to_out width (3360) not documented"
            assert "gqa" in text.lower()

    def test_yaml_ships_nonempty_list(self):
        registry = _reload_registry()
        for def_id in DEF_IDS:
            shipped = registry._definitions[def_id].lora_targetable_modules
            assert shipped and len(shipped) > 0


class TestEditDefinition:
    """Task A4: ``boogu-image-edit`` — control_inputs + zero architecture
    drift from Base (verified by the byte-identical transformer/config.json
    recon in edit.yaml's header comment / a4-report.md)."""

    def test_control_inputs_is_one(self):
        registry = _reload_registry()
        edit = registry._definitions["boogu-image-edit"]
        assert edit.control_inputs == 1

    def test_base_and_turbo_are_pure_t2i(self):
        registry = _reload_registry()
        for def_id in ("boogu-image-base", "boogu-image-turbo"):
            assert registry._definitions[def_id].control_inputs == 0

    def test_repo_path_points_at_edit_checkpoint(self):
        registry = _reload_registry()
        edit = registry._definitions["boogu-image-edit"]
        path = edit.components["repo"].path
        assert path == "huggingface:Boogu/Boogu-Image-0.1-Edit"

    def test_architecture_params_identical_to_base(self):
        """Recon finding (a4-report.md): Edit's transformer/config.json is
        byte-identical to Base's — the edit checkpoint needs ZERO transformer
        geometry changes, only the reference-image forward wiring (driver.py
        ``_build_ref_image_hidden_states``)."""
        registry = _reload_registry()
        base = registry._definitions["boogu-image-base"].architecture_params
        edit = registry._definitions["boogu-image-edit"].architecture_params
        assert base == edit

    def test_capabilities_resolve_as_edit_model(self):
        """``resolve_capabilities`` derives ``is_edit`` / disables
        augmentation+masking purely from ``control_inputs`` — no boogu_image
        family-specific wiring needed (archetypes.py is fully generic)."""
        from app.engine.core.archetypes import resolve_capabilities

        registry = _reload_registry()
        edit = registry._definitions["boogu-image-edit"]
        resolved = resolve_capabilities(edit)
        caps = resolved["capabilities"]
        assert caps["control_inputs"] == 1
        assert caps["is_edit"] is True
        assert caps["supports_augmentation"] is False
        assert caps["supports_masking_variants"] is False
