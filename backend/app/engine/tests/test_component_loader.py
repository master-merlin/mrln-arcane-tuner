"""Tests for GenericComponentLoader — manifest loading, hooks, meta-device."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubLoader(GenericComponentLoader):
    """Minimal loader subclass returning a fixed manifest."""

    def __init__(self, manifest: list[ComponentSpec]):
        super().__init__(torch.device("cpu"))
        self._manifest = manifest

    def get_component_manifest(self, definition):
        return self._manifest


def _make_definition(**component_paths) -> MagicMock:
    """Build a mock ModelDefinition with component paths."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "test"
    definition.id = "test-id"
    definition.detected_precision = {}
    definition.components = {}
    for key, path in component_paths.items():
        comp = MagicMock()
        comp.path = path
        definition.components[key] = comp
    return definition


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComponentSpec:
    """Validate ComponentSpec defaults and new fields."""

    def test_defaults(self):
        spec = ComponentSpec(key="unet", hf_class="diffusers.UNet2DConditionModel")
        assert spec.is_torch_model is True
        assert spec.dtype_override is None
        assert spec.root_key is None
        assert spec.separate_repo is False
        assert spec.fallback_to_root is False

    def test_root_key_field(self):
        spec = ComponentSpec(
            key="tok", hf_class="transformers.CLIPTokenizer",
            root_key="unet",
        )
        assert spec.root_key == "unet"

    def test_separate_repo_field(self):
        spec = ComponentSpec(
            key="vae", hf_class="diffusers.AutoencoderKL",
            separate_repo=True, definition_key="vae",
        )
        assert spec.separate_repo is True


class TestGenericComponentLoader:
    """Test manifest-driven loading mechanics."""

    @pytest.mark.anyio
    async def test_manifest_keys(self):
        """Loaded components dict has keys matching manifest specs."""
        manifest = [
            ComponentSpec(
                key="unet",
                hf_class="torch.nn.Linear",
                is_torch_model=False,
            ),
        ]
        loader = _StubLoader(manifest)
        definition = _make_definition(repo="/fake/repo")

        # Patch _import_class and _load_component
        with patch.object(loader, "_import_class") as mock_import, \
             patch.object(loader, "_load_component") as mock_load:
            mock_import.return_value = nn.Linear
            mock_load.return_value = "fake_model"

            components = await loader.load(definition)

        assert "unet" in components
        assert components["unet"] == "fake_model"

    @pytest.mark.anyio
    async def test_dtype_override(self):
        """ComponentSpec.dtype_override overrides global dtype."""
        manifest = [
            ComponentSpec(
                key="vae",
                hf_class="torch.nn.Linear",
                is_torch_model=False,
                dtype_override=torch.float32,
            ),
        ]
        loader = _StubLoader(manifest)
        definition = _make_definition(repo="/fake/repo")

        with patch.object(loader, "_import_class") as mock_import, \
             patch.object(loader, "_load_component") as mock_load:
            mock_import.return_value = nn.Linear
            mock_load.return_value = "fake_vae"

            await loader.load(definition, torch_dtype=torch.bfloat16)

        # _load_component should receive float32 (from spec override)
        call_args = mock_load.call_args
        assert call_args[0][2] == torch.float32  # dtype arg

    @pytest.mark.anyio
    async def test_post_load_hook_called(self):
        """Post-load hook method is invoked after loading."""
        manifest = [
            ComponentSpec(
                key="unet",
                hf_class="torch.nn.Linear",
                is_torch_model=False,
                post_load_hook="_my_hook",
            ),
        ]
        loader = _StubLoader(manifest)
        loader._my_hook = MagicMock(return_value="hooked_model")
        definition = _make_definition(repo="/fake/repo")

        with patch.object(loader, "_import_class") as mock_import, \
             patch.object(loader, "_load_component") as mock_load:
            mock_import.return_value = nn.Linear
            mock_load.return_value = "raw_model"

            components = await loader.load(definition)

        loader._my_hook.assert_called_once()
        assert components["unet"] == "hooked_model"

    def test_resolve_root_fallback_chain(self):
        """Root resolution: repo → path → unet fallback."""
        manifest = [ComponentSpec(key="x", hf_class="x")]
        loader = _StubLoader(manifest)

        # No "repo" key, only "unet"
        definition = _make_definition(unet="/models/sdxl-base")
        root = loader._resolve_root(definition)
        assert root == "/models/sdxl-base"

    def test_resolve_root_raises_on_missing(self):
        """No root key found → ValueError."""
        manifest = [ComponentSpec(key="x", hf_class="x")]
        loader = _StubLoader(manifest)
        definition = _make_definition()

        with pytest.raises(ValueError, match="must specify"):
            loader._resolve_root(definition)


class TestSeparateRepoSubfolder:
    """Separate-repo components that keep a Diffusers subfolder layout.

    Covers the ostris Z-Image-De-Turbo case: the transformer lives in its
    own repo under a ``transformer/`` subfolder, while VAE/TE/tokenizer
    come from the base repo root.
    """

    def _loader(self):
        loader = _StubLoader([ComponentSpec(key="x", hf_class="x")])
        return loader

    def test_descends_into_subfolder_when_present(self):
        """Separate repo + subfolder dir exists → returns the subfolder."""
        loader = self._loader()
        spec = ComponentSpec(
            key="unet",
            hf_class="diffusers.models.ZImageTransformer2DModel",
            subfolder="transformer",
            definition_key="transformer",
            separate_repo=True,
        )
        definition = _make_definition(transformer="huggingface:ostris/Z-Image-De-Turbo")

        with patch(
            "app.engine.core.pipeline.loader_base.ModelPathResolver.resolve",
            return_value="/cache/deturbo",
        ), patch(
            "app.engine.core.pipeline.loader_base.os.path.isdir",
            return_value=True,
        ):
            path = loader._resolve_component_path(spec, definition, "/cache/base")

        import os
        assert path == os.path.join("/cache/deturbo", "transformer")

    def test_falls_back_to_root_when_no_subfolder(self):
        """Separate repo but subfolder dir absent → returns the repo root."""
        loader = self._loader()
        spec = ComponentSpec(
            key="unet",
            hf_class="x",
            subfolder="transformer",
            definition_key="transformer",
            separate_repo=True,
        )
        definition = _make_definition(transformer="huggingface:some/flat-repo")

        with patch(
            "app.engine.core.pipeline.loader_base.ModelPathResolver.resolve",
            return_value="/cache/flat",
        ), patch(
            "app.engine.core.pipeline.loader_base.os.path.isdir",
            return_value=False,
        ):
            path = loader._resolve_component_path(spec, definition, "/cache/base")

        assert path == "/cache/flat"

    def test_subfolder_kwarg_specs_do_not_descend(self):
        """use_subfolder_kwarg specs (e.g. SDXL VAE) keep prior behaviour:
        return the separate repo root, never the joined subfolder."""
        loader = self._loader()
        spec = ComponentSpec(
            key="vae",
            hf_class="diffusers.AutoencoderKL",
            subfolder="vae",
            definition_key="vae",
            separate_repo=True,
            use_subfolder_kwarg=True,
        )
        definition = _make_definition(vae="huggingface:madebyollin/sdxl-vae")

        with patch(
            "app.engine.core.pipeline.loader_base.ModelPathResolver.resolve",
            return_value="/cache/vae",
        ), patch(
            "app.engine.core.pipeline.loader_base.os.path.isdir",
            return_value=True,
        ):
            path = loader._resolve_component_path(spec, definition, "/cache/base")

        assert path == "/cache/vae"


class TestImportClass:
    """Test dynamic class import utility."""

    def test_valid_import(self):
        cls = GenericComponentLoader._import_class("torch.nn.Linear")
        assert cls is nn.Linear

    def test_invalid_path(self):
        with pytest.raises(ImportError, match="Invalid class path"):
            GenericComponentLoader._import_class("nodots")
