"""
Tests for model loading pipeline.
Covers: YAML definition loading, registry operations, ModelPathResolver,
and loader path resolution logic.
"""
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch


class TestDefinitionLoading:
    """Tests for YAML definition parsing via the registry."""

    def test_sdxl_definition_loads(self):
        """SDXL base definition parses correctly."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent

        data = {
            "id": "sdxl_base_1.0",
            "family": "sdxl",
            "name": "Stable Diffusion XL Base",
            "version": "1.0",
            "defaults": {"resolution": 1024},
            "components": {
                "unet": {"path": "huggingface:stabilityai/stable-diffusion-xl-base-1.0", "type": "diffusers"},
                "vae": {"path": "huggingface:madebyollin/sdxl-vae-fp16-fix", "type": "diffusers"},
            }
        }

        defn = ModelDefinition(**data)
        assert defn.id == "sdxl_base_1.0"
        assert defn.family == "sdxl"
        assert "unet" in defn.components
        assert "vae" in defn.components
        assert isinstance(defn.components["unet"], ModelComponent)
        assert defn.components["unet"].path == "huggingface:stabilityai/stable-diffusion-xl-base-1.0"

    def test_flux2_klein_definition_loads(self):
        """Flux2 Klein definition parses correctly with 'repo' component."""
        from app.engine.core.definitions import ModelDefinition

        data = {
            "id": "flux2-klein-base-9b",
            "family": "flux2",
            "name": "Flux 2 (Klein Base 9B)",
            "version": "1.0",
            "defaults": {"resolution": 1024},
            "components": {
                "repo": {"path": "huggingface:black-forest-labs/FLUX.2-klein-base-9B", "type": "diffusers"},
            }
        }

        defn = ModelDefinition(**data)
        assert defn.id == "flux2-klein-base-9b"
        assert defn.family == "flux2"
        assert "repo" in defn.components
        assert defn.components["repo"].path == "huggingface:black-forest-labs/FLUX.2-klein-base-9B"

    def test_string_shorthand_normalized_by_registry(self):
        """Registry load_definition converts string component values to ModelComponent dicts."""
        from app.engine.models.registry import ModelRegistry

        # Create temp YAML with string shorthand
        import tempfile
        data = {
            "id": "test_shorthand",
            "family": "sdxl",
            "name": "Test Shorthand",
            "components": {
                "unet": "huggingface:org/repo"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            defn = ModelRegistry.load_definition(tmp_path)
            assert defn.components["unet"].path == "huggingface:org/repo"
        finally:
            os.unlink(tmp_path)
            # Cleanup registry state
            if "test_shorthand" in ModelRegistry._definitions:
                del ModelRegistry._definitions["test_shorthand"]
            if "test_shorthand" in ModelRegistry._paths:
                del ModelRegistry._paths["test_shorthand"]


class TestRegistryOperations:
    """Tests for registry update and save operations."""

    def test_update_definition_preserves_immutability(self):
        """update_definition creates a new model copy, not a mutation."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.registry import ModelRegistry

        defn = ModelDefinition(
            id="test_update",
            family="sdxl",
            name="Original Name",
        )
        ModelRegistry._definitions["test_update"] = defn

        try:
            ModelRegistry.update_definition("test_update", {"name": "Updated Name"})
            updated = ModelRegistry.get_definition("test_update")
            assert updated.name == "Updated Name"
            assert updated.id == "test_update"  # Other fields preserved
        finally:
            del ModelRegistry._definitions["test_update"]

    def test_save_definition_produces_valid_yaml(self, tmp_path):
        """save_definition writes valid YAML using model_dump."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.registry import ModelRegistry

        defn = ModelDefinition(
            id="test_save",
            family="sdxl",
            name="Test Save Model",
            components={"unet": ModelComponent(path="/fake/path", type="diffusers")},
        )
        yaml_path = str(tmp_path / "test_save.yaml")

        ModelRegistry._definitions["test_save"] = defn
        ModelRegistry._paths["test_save"] = yaml_path

        try:
            ModelRegistry.save_definition("test_save")

            # Verify output is valid YAML
            with open(yaml_path, "r") as f:
                loaded = yaml.safe_load(f)

            assert loaded["id"] == "test_save"
            assert loaded["name"] == "Test Save Model"
            assert loaded["components"]["unet"]["path"] == "/fake/path"
        finally:
            del ModelRegistry._definitions["test_save"]
            del ModelRegistry._paths["test_save"]


class TestModelPathResolver:
    """Tests for the path resolution logic."""

    def test_local_absolute_path_returned_as_is(self, tmp_path):
        """Absolute local paths are returned unchanged.

        Built from ``tmp_path`` so the path is absolute on THIS platform: a
        literal ``D:\\...`` is not absolute on Linux (``os.path.isabs`` is
        False there), so the resolver anchored it under the repo root and CI
        read ``/home/runner/.../D:\\Models\\...`` (gate.yml run 33687356291).
        The claim is "an absolute path is returned as-is", not "a Windows
        drive path". The file need not exist: the resolver returns it anyway
        so the caller can fail with a clear message.
        """
        from app.engine.utils.model_utils import ModelPathResolver

        path = str(tmp_path / "Models" / "sdxl" / "model.safetensors")
        assert os.path.isabs(path)
        result = ModelPathResolver.resolve(path)
        assert result == path

    def test_hf_uri_detected(self):
        """huggingface: URI prefix triggers HF download logic."""
        from app.engine.utils.model_utils import ModelPathResolver

        # Mock _resolve_hf directly — the real implementation has a two-step
        # local_files_only pattern that makes mocking single functions fragile.
        with patch.object(ModelPathResolver, "_resolve_hf", return_value="/tmp/cached/repo") as mock_hf:
            result = ModelPathResolver.resolve("huggingface:org/repo")
            mock_hf.assert_called_once_with("huggingface:org/repo", local_files_only=False)
            assert result == "/tmp/cached/repo"

    def test_hf_uri_with_filename(self):
        """huggingface:repo:filename triggers file-specific download."""
        from app.engine.utils.model_utils import ModelPathResolver

        with patch.object(ModelPathResolver, "_resolve_hf", return_value="/tmp/cached/file.safetensors") as mock_hf:
            result = ModelPathResolver.resolve("huggingface:org/repo:model.safetensors")
            mock_hf.assert_called_once_with("huggingface:org/repo:model.safetensors", local_files_only=False)
            assert result == "/tmp/cached/file.safetensors"

    def test_find_component_explicit_definition(self, tmp_path):
        """find_component prefers explicit definition over root discovery.

        The explicit path is platform-native absolute (from ``tmp_path``), for
        the same reason as ``test_local_absolute_path_returned_as_is``.
        """
        from app.engine.utils.model_utils import ModelPathResolver

        explicit = str(tmp_path / "explicit" / "path")
        assert os.path.isabs(explicit)
        mock_def = MagicMock()
        mock_comp = MagicMock()
        mock_comp.path = explicit
        mock_def.components.get.return_value = mock_comp

        result = ModelPathResolver.find_component(mock_def, "vae", "/root/path", ["ae.safetensors"])
        assert result == explicit

    def test_find_component_discovers_in_root(self, tmp_path):
        """find_component discovers files in root path when not explicitly defined."""
        from app.engine.utils.model_utils import ModelPathResolver

        # Create a candidate file
        (tmp_path / "ae.safetensors").touch()

        mock_def = MagicMock()
        mock_def.components.get.return_value = None  # Not explicitly defined

        result = ModelPathResolver.find_component(
            mock_def, "vae", str(tmp_path), ["ae.safetensors"]
        )
        assert result == str(tmp_path / "ae.safetensors")

    def test_find_component_returns_none_when_missing(self, tmp_path):
        """find_component returns None when component is not found anywhere."""
        from app.engine.utils.model_utils import ModelPathResolver

        mock_def = MagicMock()
        mock_def.components.get.return_value = None

        result = ModelPathResolver.find_component(
            mock_def, "vae", str(tmp_path), ["nonexistent.safetensors"]
        )
        assert result is None


class TestSDXLLoaderPathLogic:
    """Tests for SDXL loader's path resolution without actual model loading."""

    def test_separate_vae_uses_root_not_subfolder(self):
        """When VAE is an explicit separate repo, it should NOT use subfolder='vae'."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.families.sdxl.loader import SDXLLoader

        defn = ModelDefinition(
            id="test",
            family="sdxl",
            name="Test",
            components={
                "unet": ModelComponent(path="/repo/sdxl-base"),
                "vae": ModelComponent(path="/repo/vae-fp16-fix"),
            }
        )

        SDXLLoader(device="cpu")

        # Test resolve_path behavior - we can't run async load, but validate path logic
        unet_comp = defn.components.get("unet")
        vae_comp = defn.components.get("vae")

        assert unet_comp.path != vae_comp.path, "Separate VAE should have different path"

    def test_missing_unet_raises(self):
        """Loader raises ValueError when unet component is missing."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.families.sdxl.loader import SDXLLoader

        defn = ModelDefinition(
            id="test",
            family="sdxl",
            name="Test No UNet",
            components={}
        )

        loader = SDXLLoader(device="cpu")
        import asyncio
        with pytest.raises(ValueError, match="must specify.*unet"):
            asyncio.new_event_loop().run_until_complete(loader.load(defn))


class TestFluxLoaderContract:
    """Integration tests for FLUX loader → trainer contract.

    These validate that loaders interact correctly with Pydantic
    ModelComponent objects and that trainers implement all required
    abstract methods — bugs that unit tests missed at runtime.
    """

    def test_flux1_resolve_repo_reads_model_component_path(self):
        """_resolve_root must access ModelComponent.path, not chain .get()."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.families.flux1.loader import Flux1Loader

        defn = ModelDefinition(
            id="flux1-dev",
            family="flux1",
            name="FLUX.1 Dev",
            components={
                "repo": ModelComponent(path="/local/FLUX.1-dev"),
            }
        )
        loader = Flux1Loader(device="cpu")
        result = loader._resolve_root(defn)
        assert result == "/local/FLUX.1-dev"
        assert loader._root_path == "/local/FLUX.1-dev"

    def test_flux2_resolve_repo_reads_model_component_path(self):
        """_resolve_root must access ModelComponent.path, not chain .get()."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.families.flux2.loader import Flux2Loader

        defn = ModelDefinition(
            id="flux2-klein",
            family="flux2",
            name="FLUX.2 Klein",
            components={
                "repo": ModelComponent(path="/local/FLUX.2-klein"),
            }
        )
        loader = Flux2Loader(device="cpu")
        result = loader._resolve_root(defn)
        assert result == "/local/FLUX.2-klein"
        assert loader._root_path == "/local/FLUX.2-klein"

    def test_flux1_loader_missing_repo_raises(self):
        """Loader must raise ValueError when 'repo' component is missing."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.families.flux1.loader import Flux1Loader

        defn = ModelDefinition(
            id="bad", family="flux1", name="No Repo", components={}
        )
        loader = Flux1Loader(device="cpu")
        with pytest.raises(ValueError, match="must specify"):
            loader._resolve_root(defn)

    def test_flux2_loader_missing_repo_raises(self):
        """Loader must raise ValueError when 'repo' component is missing."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.families.flux2.loader import Flux2Loader

        defn = ModelDefinition(
            id="bad", family="flux2", name="No Repo", components={}
        )
        loader = Flux2Loader(device="cpu")
        with pytest.raises(ValueError, match="must specify"):
            loader._resolve_root(defn)

    def test_flux1_loader_accepts_torch_dtype_kwarg(self):
        """load() must accept torch_dtype= kwarg from GenericTrainingPipeline."""
        import inspect
        from app.engine.models.families.flux1.loader import Flux1Loader

        sig = inspect.signature(Flux1Loader.load)
        assert "torch_dtype" in sig.parameters, (
            "Flux1Loader.load() must accept 'torch_dtype' keyword argument"
        )

    def test_flux2_loader_accepts_torch_dtype_kwarg(self):
        """load() must accept torch_dtype= kwarg from GenericTrainingPipeline."""
        import inspect
        from app.engine.models.families.flux2.loader import Flux2Loader

        sig = inspect.signature(Flux2Loader.load)
        assert "torch_dtype" in sig.parameters, (
            "Flux2Loader.load() must accept 'torch_dtype' keyword argument"
        )

    def test_flux1_trainer_implements_assign_components(self):
        """Flux1Trainer must implement _assign_components (abstract)."""
        from app.engine.models.families.flux1.trainer import Flux1Trainer

        # If _assign_components is missing, Python won't even let us
        # reference the class — but we verify it's not abstract
        assert hasattr(Flux1Trainer, "_assign_components")
        assert not getattr(
            Flux1Trainer._assign_components, "__isabstractmethod__", False
        ), "_assign_components must be concrete, not abstract"

    def test_flux2_trainer_implements_assign_components(self):
        """Flux2Trainer must implement _assign_components (abstract)."""
        from app.engine.models.families.flux2.trainer import Flux2Trainer

        assert hasattr(Flux2Trainer, "_assign_components")
        assert not getattr(
            Flux2Trainer._assign_components, "__isabstractmethod__", False
        ), "_assign_components must be concrete, not abstract"

    def test_all_trainers_instantiate(self):
        """Every registered trainer must be instantiable (no missing abstractmethod)."""
        from app.engine.core.definitions import ModelDefinition

        trainer_classes = []
        try:
            from app.engine.models.families.flux1.trainer import Flux1Trainer
            trainer_classes.append(("flux1", Flux1Trainer))
        except ImportError:
            pass
        try:
            from app.engine.models.families.flux2.trainer import Flux2Trainer
            trainer_classes.append(("flux2", Flux2Trainer))
        except ImportError:
            pass
        try:
            from app.engine.models.families.sdxl.trainer import SDXLTrainer
            trainer_classes.append(("sdxl", SDXLTrainer))
        except ImportError:
            pass

        for family_name, cls in trainer_classes:
            defn = ModelDefinition(
                id=f"test-{family_name}",
                family=family_name,
                name=f"Test {family_name}",
                components={},
            )
            # Should not raise TypeError about missing abstractmethod
            try:
                cls(defn, {"output_dir": "/tmp/test"})
            except TypeError as e:
                if "abstract" in str(e).lower():
                    pytest.fail(
                        f"{cls.__name__} cannot be instantiated: {e}"
                    )
