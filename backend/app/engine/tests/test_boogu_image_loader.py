"""Tests for the boogu_image loader (Task 3) — manifest + manual vendored load.

Mirrors the krea2 loader test style (``app/engine/tests/test_krea2_family.py``
Step B — ``test_krea2_manifest_components``): a manifest-shape test plus a
full ``load()`` integration test.

The full ``load()`` test builds a REAL tiny on-disk checkpoint for the two
VENDORED components (transformer, scheduler) via ``save_pretrained`` /
``from_pretrained`` round trips (no network, CPU-only, matches the divisibility
-respecting tiny config already used by ``test_boogu_image_vendor.py`` /
``test_boogu_image_definitions.py``) — this proves real CLASS IDENTITY, not
just "didn't crash". The three STOCK components (text_encoder/mllm, processor,
vae) are heavy real HF classes we don't want to download or instantiate here,
so only ``GenericComponentLoader._load_component`` is mocked for those three
manifest entries; the class IMPORT (``_import_class``) still runs for real for
every component, proving the manifest's ``hf_class`` strings resolve to the
correct real classes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import GenericComponentLoader
from app.engine.models.families.boogu_image.loader import BooguImageLoader

# Same divisibility-respecting tiny config as test_boogu_image_vendor.py /
# test_boogu_image_definitions.py.
_TINY_TRANSFORMER_CFG = dict(
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


def _make_boogu_definition(root: str | None = None) -> MagicMock:
    """Build a mock Boogu-Image ModelDefinition for loader tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "boogu_image"
    definition.id = "boogu-image-test"
    definition.detected_precision = {}
    definition.architecture_params = {}
    definition.components = {}
    if root is not None:
        comp = MagicMock()
        comp.path = root
        definition.components["repo"] = comp
    return definition


class TestManifest:
    """Manifest-shape tests — no downloads, no instantiation."""

    def test_manifest_declares_stock_and_vendored_scheduler_components(self):
        loader = BooguImageLoader(torch.device("cpu"))
        definition = _make_boogu_definition()
        specs = loader.get_component_manifest(definition)

        keys = {s.key for s in specs}
        assert keys == {"text_encoder", "processor", "vae", "scheduler"}, (
            f"unexpected manifest keys: {keys}"
        )
        # The transformer is loaded by hand in load() — NOT via the manifest.
        assert "unet" not in keys
        assert "transformer" not in keys

    def test_text_encoder_is_stock_qwen3vl_no_rope_shim_needed(self):
        loader = BooguImageLoader(torch.device("cpu"))
        specs = {s.key: s for s in loader.get_component_manifest(_make_boogu_definition())}

        assert (
            specs["text_encoder"].hf_class
            == "transformers.Qwen3VLForConditionalGeneration"
        )
        assert specs["text_encoder"].subfolder == "mllm"

    def test_processor_is_qwen3vl_processor_not_a_torch_model(self):
        loader = BooguImageLoader(torch.device("cpu"))
        specs = {s.key: s for s in loader.get_component_manifest(_make_boogu_definition())}

        assert specs["processor"].hf_class == "transformers.Qwen3VLProcessor"
        assert specs["processor"].subfolder == "processor"
        assert specs["processor"].is_torch_model is False

    def test_vae_is_stock_autoencoder_kl(self):
        loader = BooguImageLoader(torch.device("cpu"))
        specs = {s.key: s for s in loader.get_component_manifest(_make_boogu_definition())}

        assert specs["vae"].hf_class == "diffusers.AutoencoderKL"
        assert specs["vae"].subfolder == "vae"

    def test_scheduler_is_the_vendored_class_not_stock_diffusers(self):
        loader = BooguImageLoader(torch.device("cpu"))
        specs = {s.key: s for s in loader.get_component_manifest(_make_boogu_definition())}

        sched_spec = specs["scheduler"]
        assert sched_spec.hf_class == (
            "app.engine.models.families.boogu_image.vendor.schedulers."
            "scheduling_flow_match_euler_discrete_time_shifting."
            "FlowMatchEulerDiscreteScheduler"
        )
        assert sched_spec.hf_class != "diffusers.FlowMatchEulerDiscreteScheduler"
        assert sched_spec.subfolder == "scheduler"
        # Not an nn.Module — no params/buffers, must not go through device
        # placement in the generic loader's load() step.
        assert sched_spec.is_torch_model is False

    def test_no_rope_translation_shim_copied_from_krea2(self):
        """Krea2Loader carries a ``_translate_qwen3vl_rope_config`` shim for its
        transformers-5.2-format checkpoint. Boogu's mllm config is already
        transformers-4.57-shaped (verified, task-3-brief.md) — prove the shim
        was NOT copied into BooguImageLoader."""
        assert not hasattr(BooguImageLoader, "_translate_qwen3vl_rope_config")


class TestFullLoad:
    """Full ``load()`` integration: real vendored classes, mocked stock ones."""

    def _build_tiny_scheduler(self, root: Path) -> None:
        from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler_dir = root / "scheduler"
        scheduler_dir.mkdir(parents=True)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            do_shift=True,
            dynamic_time_shift=False,
            time_shift_version="v1",
            seq_len=4096,
        )
        scheduler.save_pretrained(str(scheduler_dir))

    def _build_tiny_checkpoint(self, root: Path) -> None:
        from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )

        transformer_dir = root / "transformer"
        transformer_dir.mkdir(parents=True)
        model = BooguImageTransformer2DModel(**_TINY_TRANSFORMER_CFG)
        model.save_pretrained(str(transformer_dir))

        self._build_tiny_scheduler(root)

    def _fake_load_component(self, seen_classes: dict):
        """Side effect for ``_load_component``: real for the scheduler
        (vendored, cheap local config), mocked for the three stock/heavy
        components (text_encoder, processor, vae)."""

        def _side_effect(cls, path, dtype, spec, *, raw_safetensors=False):
            seen_classes[spec.key] = cls
            if spec.key == "scheduler":
                return GenericComponentLoader._load_component(
                    cls, path, dtype, spec, raw_safetensors=raw_safetensors,
                )
            mock = MagicMock(name=f"mock_{spec.key}")
            mock._boogu_test_key = spec.key
            return mock

        return _side_effect

    @pytest.mark.anyio
    async def test_load_wires_vendored_transformer_and_scheduler_by_class_identity(
        self, tmp_path,
    ):
        from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )
        from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )
        from diffusers import FlowMatchEulerDiscreteScheduler as StockFlowMatchScheduler

        self._build_tiny_checkpoint(tmp_path)
        definition = _make_boogu_definition(root=str(tmp_path))

        loader = BooguImageLoader(torch.device("cpu"))
        seen_classes: dict = {}
        with patch.object(
            loader, "_load_component", side_effect=self._fake_load_component(seen_classes),
        ):
            components = await loader.load(definition, torch_dtype=torch.bfloat16)

        # -- Vendored transformer: exact class identity, not just no-crash. --
        assert type(components["unet"]) is BooguImageTransformer2DModel

        # -- Vendored scheduler: exact class identity, and provably NOT the
        #    stock diffusers class of the same name (different module + it
        #    actually carries the Boogu-specific config keys). --
        assert type(components["scheduler"]) is FlowMatchEulerDiscreteScheduler
        assert type(components["scheduler"]) is not StockFlowMatchScheduler
        assert (
            type(components["scheduler"]).__module__
            .startswith("app.engine.models.families.boogu_image.vendor")
        )
        assert components["scheduler"].config.do_shift is True
        assert components["scheduler"].config.dynamic_time_shift is False
        assert components["scheduler"].config.time_shift_version == "v1"
        assert components["scheduler"].config.seq_len == 4096

        # -- Stock components: real class import proven via the classes
        #    _load_component actually received. --
        assert seen_classes["text_encoder"].__name__ == "Qwen3VLForConditionalGeneration"
        assert seen_classes["processor"].__name__ == "Qwen3VLProcessor"
        assert seen_classes["vae"].__name__ == "AutoencoderKL"
        assert components["text_encoder"]._boogu_test_key == "text_encoder"
        assert components["processor"]._boogu_test_key == "processor"
        assert components["vae"]._boogu_test_key == "vae"

    @pytest.mark.anyio
    async def test_load_places_transformer_in_requested_dtype(self, tmp_path):
        self._build_tiny_checkpoint(tmp_path)
        definition = _make_boogu_definition(root=str(tmp_path))

        loader = BooguImageLoader(torch.device("cpu"))
        with patch.object(
            loader, "_load_component", side_effect=self._fake_load_component({}),
        ):
            components = await loader.load(definition, torch_dtype=torch.bfloat16)

        model = components["unet"]
        assert next(model.parameters()).dtype == torch.bfloat16

    @pytest.mark.anyio
    async def test_load_tolerates_missing_prompt_embedding_weights(self, tmp_path):
        """``use_prompt_tuning: false`` (both shipped definitions) -- the
        checkpoint ships no PromptEmbedding weights, and the loader must not
        require them. A model built with ``use_prompt_tuning=False`` never
        instantiates that submodule, so from_pretrained neither expects nor
        needs those weights."""
        self._build_tiny_checkpoint(tmp_path)
        definition = _make_boogu_definition(root=str(tmp_path))

        loader = BooguImageLoader(torch.device("cpu"))
        with patch.object(
            loader, "_load_component", side_effect=self._fake_load_component({}),
        ):
            components = await loader.load(definition, torch_dtype=torch.bfloat16)

        model = components["unet"]
        assert model.config.prompt_tuning_configs["use_prompt_tuning"] is False
        assert not hasattr(model, "prompt_embedding")

    @pytest.mark.anyio
    async def test_load_falls_back_to_root_when_transformer_subfolder_missing(
        self, tmp_path,
    ):
        """No 'transformer/' subfolder and no root config.json -> FileNotFoundError
        with a clear message (mirrors krea2's guard). The manifest components
        (scheduler real, text_encoder/processor/vae mocked) load fine first —
        only the hand-loaded transformer step is missing its directory."""
        self._build_tiny_scheduler(tmp_path)
        definition = _make_boogu_definition(root=str(tmp_path))
        loader = BooguImageLoader(torch.device("cpu"))

        with patch.object(
            loader, "_load_component", side_effect=self._fake_load_component({}),
        ):
            with pytest.raises(FileNotFoundError, match="transformer"):
                await loader.load(definition, torch_dtype=torch.bfloat16)
