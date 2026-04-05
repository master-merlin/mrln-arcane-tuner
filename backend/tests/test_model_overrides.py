"""
Tests for Model Source Override system — schemas, manager, and API routes.
"""

import json
import os

import pytest
from pydantic import ValidationError
from unittest.mock import patch

from app.core.schemas.model_overrides import (
    ModelOverride,
    ModelSettings,
    ModelSourceType,
)
from app.core.settings_manager import SettingsManager
from app.engine.utils.model_override_manager import ModelOverrideManager


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Guarantee a fresh SettingsManager singleton for every test."""
    SettingsManager._instance = None
    yield
    SettingsManager._instance = None


@pytest.fixture()
def settings_file(tmp_path):
    """Return a temporary settings.json path inside *tmp_path*."""
    return str(tmp_path / "settings.json")


def _make_manager(settings_file: str) -> SettingsManager:
    """Create a SettingsManager pointing at the given file."""
    mgr = SettingsManager.__new__(SettingsManager)
    mgr.root_dir = os.path.dirname(settings_file)
    mgr.storage_file = settings_file
    mgr.settings = {}
    mgr.load()
    defaults = {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"}
    if "application" not in mgr.settings:
        mgr.settings["application"] = defaults
        mgr.save()
    else:
        changed = False
        for k, v in defaults.items():
            if k not in mgr.settings["application"]:
                mgr.settings["application"][k] = v
                changed = True
        if changed:
            mgr.save()
    return mgr


# ── Schema Validation ────────────────────────────────────────────────────


class TestModelOverrideSchema:
    """Tests for ModelOverride Pydantic V2 schema."""

    def test_default_values(self):
        override = ModelOverride()
        assert override.source_type == ModelSourceType.HF_HUB
        assert override.local_path is None
        assert override.skip_update is False

    def test_hf_hub_no_path_ok(self):
        override = ModelOverride(source_type="hf_hub")
        assert override.source_type == ModelSourceType.HF_HUB
        assert override.local_path is None

    def test_hf_hub_with_skip_update(self):
        override = ModelOverride(source_type="hf_hub", skip_update=True)
        assert override.skip_update is True

    def test_local_diffusers_with_path(self):
        override = ModelOverride(
            source_type="local_diffusers",
            local_path="D:\\Models\\sdxl-base",
        )
        assert override.source_type == ModelSourceType.LOCAL_DIFFUSERS
        assert override.local_path == "D:\\Models\\sdxl-base"

    def test_local_diffusers_without_path_raises(self):
        with pytest.raises(ValidationError, match="local_path is required"):
            ModelOverride(source_type="local_diffusers")

    def test_local_safetensors_without_path_raises(self):
        with pytest.raises(ValidationError, match="local_path is required"):
            ModelOverride(source_type="local_safetensors")

    def test_local_safetensors_with_path(self):
        override = ModelOverride(
            source_type="local_safetensors",
            local_path="/models/flux/safetensors",
        )
        assert override.source_type == ModelSourceType.LOCAL_SAFETENSORS

    def test_model_dump_roundtrip(self):
        original = ModelOverride(
            source_type="local_diffusers",
            local_path="D:\\Models\\flux2",
            skip_update=False,
        )
        dumped = original.model_dump()
        restored = ModelOverride.model_validate(dumped)
        assert original == restored

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValidationError):
            ModelOverride(source_type="s3_bucket")


class TestModelSettingsSchema:
    """Tests for top-level ModelSettings schema."""

    def test_defaults(self):
        settings = ModelSettings()
        assert settings.global_offline_mode is False
        assert settings.overrides == {}

    def test_with_overrides(self):
        settings = ModelSettings(
            global_offline_mode=True,
            overrides={
                "flux2-dev": ModelOverride(source_type="hf_hub", skip_update=True),
                "sdxl-base": ModelOverride(
                    source_type="local_diffusers",
                    local_path="/sd/sdxl",
                ),
            },
        )
        assert len(settings.overrides) == 2
        assert settings.global_offline_mode is True

    def test_model_dump_json_serializable(self):
        settings = ModelSettings(
            overrides={
                "test": ModelOverride(
                    source_type="local_diffusers", local_path="/tmp/model"
                )
            }
        )
        dumped = settings.model_dump()
        # Should be JSON-serializable
        json_str = json.dumps(dumped)
        assert "local_diffusers" in json_str


# ── ModelOverrideManager ────────────────────────────────────────────────


class TestModelOverrideManager:
    """Tests for CRUD and query helpers."""

    def test_get_override_empty(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            result = ModelOverrideManager.get_override("nonexistent")
            assert result is None

    def test_set_and_get_override(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            override = ModelOverride(
                source_type="local_diffusers",
                local_path="/models/sdxl",
            )
            ModelOverrideManager.set_override("sdxl-base", override)
            result = ModelOverrideManager.get_override("sdxl-base")
            assert result is not None
            assert result.source_type == ModelSourceType.LOCAL_DIFFUSERS
            assert result.local_path == "/models/sdxl"

    def test_delete_override(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            override = ModelOverride(
                source_type="local_diffusers", local_path="/models/flux"
            )
            ModelOverrideManager.set_override("flux2-dev", override)
            assert ModelOverrideManager.get_override("flux2-dev") is not None

            ModelOverrideManager.delete_override("flux2-dev")
            assert ModelOverrideManager.get_override("flux2-dev") is None

    def test_delete_nonexistent_is_noop(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            # Should not raise
            ModelOverrideManager.delete_override("does_not_exist")

    def test_get_all_returns_full_settings(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "a",
                ModelOverride(source_type="local_diffusers", local_path="/a"),
            )
            ModelOverrideManager.set_override(
                "b",
                ModelOverride(source_type="hf_hub", skip_update=True),
            )
            all_settings = ModelOverrideManager.get_all()
            assert isinstance(all_settings, ModelSettings)
            assert len(all_settings.overrides) == 2

    def test_set_global_offline(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_global_offline(True)
            settings = ModelOverrideManager.get_all()
            assert settings.global_offline_mode is True

            ModelOverrideManager.set_global_offline(False)
            settings = ModelOverrideManager.get_all()
            assert settings.global_offline_mode is False

    def test_is_offline_global(self, settings_file):
        """Global offline mode makes any definition 'offline'."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_global_offline(True)
            assert ModelOverrideManager.is_offline("any_model") is True

    def test_is_offline_per_model(self, settings_file):
        """Per-model skip_update makes that specific definition 'offline'."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "flux2-dev",
                ModelOverride(source_type="hf_hub", skip_update=True),
            )
            assert ModelOverrideManager.is_offline("flux2-dev") is True
            assert ModelOverrideManager.is_offline("other_model") is False

    def test_is_offline_global_overrides_per_model(self, settings_file):
        """Global offline should override even models without skip_update."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_global_offline(True)
            # Model has no override but global offline is on
            assert ModelOverrideManager.is_offline("no_override_model") is True

    def test_resolve_effective_source_hf_default(self, settings_file):
        """No override → HF_HUB, no local path, not offline."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            src_type, path, offline = ModelOverrideManager.resolve_effective_source(
                "unknown"
            )
            assert src_type == ModelSourceType.HF_HUB
            assert path is None
            assert offline is False

    def test_resolve_effective_source_hf_skip_update(self, settings_file):
        """HF_HUB with skip_update → local_files_only=True."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "flux2-dev",
                ModelOverride(source_type="hf_hub", skip_update=True),
            )
            src_type, path, offline = ModelOverrideManager.resolve_effective_source(
                "flux2-dev"
            )
            assert src_type == ModelSourceType.HF_HUB
            assert path is None
            assert offline is True

    def test_resolve_effective_source_local_diffusers(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "sdxl-base",
                ModelOverride(
                    source_type="local_diffusers", local_path="/models/sdxl"
                ),
            )
            src_type, path, offline = ModelOverrideManager.resolve_effective_source(
                "sdxl-base"
            )
            assert src_type == ModelSourceType.LOCAL_DIFFUSERS
            assert path == "/models/sdxl"
            assert offline is False

    def test_resolve_effective_source_local_safetensors(self, settings_file):
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "flux2-dev",
                ModelOverride(
                    source_type="local_safetensors",
                    local_path="/models/flux/raw",
                ),
            )
            src_type, path, offline = ModelOverrideManager.resolve_effective_source(
                "flux2-dev"
            )
            assert src_type == ModelSourceType.LOCAL_SAFETENSORS
            assert path == "/models/flux/raw"
            assert offline is False

    def test_resolve_effective_source_global_offline_affects_hf(self, settings_file):
        """Global offline mode should set local_files_only for HF models."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_global_offline(True)
            src_type, path, offline = ModelOverrideManager.resolve_effective_source(
                "any_model"
            )
            assert src_type == ModelSourceType.HF_HUB
            assert offline is True

    def test_settings_persistence_to_disk(self, settings_file):
        """Overrides should survive a settings roundtrip to disk."""
        mgr = _make_manager(settings_file)
        with patch(
            "app.engine.utils.model_override_manager.get_settings_manager",
            return_value=mgr,
        ):
            ModelOverrideManager.set_override(
                "flux2-dev",
                ModelOverride(
                    source_type="local_diffusers", local_path="/models/flux"
                ),
            )

        # Re-read from disk
        with open(settings_file) as f:
            on_disk = json.load(f)

        assert "models" in on_disk
        assert "flux2-dev" in on_disk["models"]["overrides"]
        assert (
            on_disk["models"]["overrides"]["flux2-dev"]["source_type"]
            == "local_diffusers"
        )


# ── Path Validation Logic ───────────────────────────────────────────────


class TestPathValidation:
    """Unit tests for the path validation probe logic (simulate directory structures)."""

    def test_validate_nonexistent_path(self, tmp_path):
        """Non-existent path returns valid=False."""
        result = self._probe(tmp_path / "does_not_exist")
        assert result["valid"] is False
        assert result["type"] == "unknown"

    def test_validate_diffusers_directory(self, tmp_path):
        """Directory with model_index.json should be identified as diffusers."""
        model_dir = tmp_path / "sdxl-base"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text("{}")
        (model_dir / "unet").mkdir()
        (model_dir / "vae").mkdir()
        (model_dir / "text_encoder").mkdir()

        result = self._probe(model_dir)
        assert result["valid"] is True
        assert result["type"] == "diffusers"
        assert "unet" in result["components_found"]
        assert "vae" in result["components_found"]

    def test_validate_diffusers_no_model_index_but_subdirs(self, tmp_path):
        """2+ known subdirs without model_index.json → still diffusers."""
        model_dir = tmp_path / "sdxl-copy"
        model_dir.mkdir()
        (model_dir / "transformer").mkdir()
        (model_dir / "vae").mkdir()
        (model_dir / "text_encoder").mkdir()

        result = self._probe(model_dir)
        assert result["valid"] is True
        assert result["type"] == "diffusers"

    def test_validate_safetensors_directory(self, tmp_path):
        """Directory with .safetensors files → safetensors type."""
        model_dir = tmp_path / "flux-raw"
        model_dir.mkdir()
        (model_dir / "transformer.safetensors").write_bytes(b"\x00" * 16)
        (model_dir / "vae.safetensors").write_bytes(b"\x00" * 16)

        result = self._probe(model_dir)
        assert result["valid"] is True
        assert result["type"] == "safetensors"
        assert "transformer" in result["components_found"]
        assert "vae" in result["components_found"]
        assert len(result["warnings"]) == 1
        assert "Raw safetensors" in result["warnings"][0]

    def test_validate_single_safetensors_file(self, tmp_path):
        """A single .safetensors file → safetensors type."""
        st_file = tmp_path / "model.safetensors"
        st_file.write_bytes(b"\x00" * 16)

        result = self._probe(st_file)
        assert result["valid"] is True
        assert result["type"] == "safetensors"
        assert result["components_found"] == ["model"]

    def test_validate_empty_directory(self, tmp_path):
        """Empty directory → valid but with warning."""
        model_dir = tmp_path / "empty"
        model_dir.mkdir()

        result = self._probe(model_dir)
        assert result["valid"] is True
        assert result["type"] == "unknown"
        assert len(result["warnings"]) == 1
        assert "no model files detected" in result["warnings"][0]

    def test_validate_single_subdir_not_diffusers(self, tmp_path):
        """Only 1 known subdir should not classify as diffusers."""
        model_dir = tmp_path / "partial"
        model_dir.mkdir()
        (model_dir / "vae").mkdir()

        result = self._probe(model_dir)
        assert result["valid"] is True
        # Less than 2 known subdirs and no safetensors → no classification
        assert result["type"] == "unknown"

    @staticmethod
    def _probe(path) -> dict:
        """Reproduce the path validation logic from the API route."""
        from pathlib import Path

        p = Path(path)
        result = {
            "valid": False,
            "type": "unknown",
            "components_found": [],
            "warnings": [],
        }

        if not p.exists():
            return result

        result["valid"] = True

        if p.is_file() and p.suffix == ".safetensors":
            result["type"] = "safetensors"
            result["components_found"] = [p.stem]
            return result

        if p.is_dir():
            has_model_index = (p / "model_index.json").is_file()
            known_subdirs = [
                "transformer", "unet", "vae", "text_encoder",
                "text_encoder_2", "tokenizer", "tokenizer_2",
                "scheduler", "ae",
            ]
            found = [d for d in known_subdirs if (p / d).is_dir()]
            safetensors_files = list(p.glob("*.safetensors"))

            if has_model_index or len(found) >= 2:
                result["type"] = "diffusers"
                result["components_found"] = found
            elif safetensors_files:
                result["type"] = "safetensors"
                result["components_found"] = [f.stem for f in safetensors_files]
                result["warnings"].append(
                    "Raw safetensors detected. Ensure all required "
                    "components are present and the model definition "
                    "has been enriched with architecture_params."
                )
            else:
                result["warnings"].append(
                    "Directory exists but no model files detected."
                )

        return result


# ── ModelPathResolver ────────────────────────────────────────────────────


class TestModelPathResolverLocalFilesOnly:
    """Tests for local_files_only flag in ModelPathResolver."""

    def test_resolve_with_local_files_only_missing_raises(self):
        """When local_files_only=True and model not cached, should raise."""
        from app.engine.utils.model_utils import ModelPathResolver

        with pytest.raises(FileNotFoundError, match="not found in local HF cache"):
            ModelPathResolver._resolve_hf(
                "huggingface:nonexistent/repo-that-doesnt-exist-12345",
                local_files_only=True,
            )

    def test_resolve_passes_local_files_only_to_hf(self):
        """resolve() should propagate local_files_only to _resolve_hf."""
        from app.engine.utils.model_utils import ModelPathResolver

        with patch.object(
            ModelPathResolver,
            "_resolve_hf",
            side_effect=FileNotFoundError("test"),
        ) as mock_hf:
            with pytest.raises(FileNotFoundError):
                ModelPathResolver.resolve(
                    "huggingface:test/model",
                    local_files_only=True,
                )
            mock_hf.assert_called_once_with(
                "huggingface:test/model",
                local_files_only=True,
            )
