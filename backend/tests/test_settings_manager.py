"""
Tests for SettingsManager singleton — covers persistence, defaults migration,
module get/update, merge strategy, and defensive save behaviour.
"""

import json
import os

import pytest

from app.core.settings_manager import SettingsManager


# ── Helpers ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Guarantee a fresh singleton for every test."""
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
    # Replicate the default-application-settings init logic
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


# ── Singleton ────────────────────────────────────────────────────────────


class TestSingleton:
    """Tests for the singleton lifecycle."""

    def test_get_instance_returns_same_object(self):
        """Two consecutive calls must return the same instance."""
        a = SettingsManager.get_instance()
        b = SettingsManager.get_instance()
        assert a is b

    def test_reset_clears_instance(self):
        """After reset, get_instance creates a *new* object."""
        first = SettingsManager.get_instance()
        SettingsManager._instance = None
        second = SettingsManager.get_instance()
        assert first is not second


# ── Load / Save ──────────────────────────────────────────────────────────


class TestLoadSave:
    """Tests for load() and save() round-trips."""

    def test_load_from_existing_file(self, settings_file):
        """Pre-populated JSON should be loaded correctly."""
        data = {"application": {"backend_port": 9999, "frontend_port": 4200, "log_level": "DEBUG"}, "ui": {"theme": "dark"}}
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        assert mgr.settings["application"]["backend_port"] == 9999
        assert mgr.settings["ui"]["theme"] == "dark"

    def test_load_from_missing_file(self, settings_file):
        """Missing file should produce empty settings then defaults written."""
        mgr = _make_manager(settings_file)
        assert "application" in mgr.settings
        assert os.path.exists(settings_file)

    def test_load_from_corrupt_file(self, settings_file):
        """Corrupt JSON should fall back to empty settings."""
        with open(settings_file, "w") as f:
            f.write("{{{invalid json")

        mgr = _make_manager(settings_file)
        # Should have recovered with defaults
        assert "application" in mgr.settings

    def test_save_writes_json(self, settings_file):
        """save() should persist current settings to disk."""
        mgr = _make_manager(settings_file)
        mgr.settings["custom"] = {"key": "value"}
        mgr.save()

        with open(settings_file) as f:
            on_disk = json.load(f)
        assert on_disk["custom"]["key"] == "value"

    def test_save_application_key_first(self, settings_file):
        """The 'application' key must appear first in the JSON for readability."""
        mgr = _make_manager(settings_file)
        mgr.settings["zzz"] = {"late": True}
        mgr.settings["aaa"] = {"early": True}
        mgr.save()

        with open(settings_file) as f:
            raw = f.read()
        # "application" should appear before "zzz" and "aaa"
        assert raw.index('"application"') < raw.index('"zzz"')
        assert raw.index('"application"') < raw.index('"aaa"')


# ── Default Migration ────────────────────────────────────────────────────


class TestDefaultMigration:
    """Tests for default-value backfilling in the application block."""

    def test_missing_application_block_created(self, settings_file):
        """If settings exist but have no 'application', it should be added."""
        with open(settings_file, "w") as f:
            json.dump({"ui": {"theme": "light"}}, f)

        mgr = _make_manager(settings_file)
        assert "application" in mgr.settings
        assert mgr.settings["application"]["backend_port"] == 8000

    def test_partial_application_block_backfilled(self, settings_file):
        """Existing keys survive; missing keys are inserted."""
        with open(settings_file, "w") as f:
            json.dump({"application": {"backend_port": 5555}}, f)

        mgr = _make_manager(settings_file)
        assert mgr.settings["application"]["backend_port"] == 5555  # untouched
        assert mgr.settings["application"]["frontend_port"] == 4200  # backfilled
        assert mgr.settings["application"]["log_level"] == "INFO"  # backfilled

    def test_complete_application_block_untouched(self, settings_file):
        """If all defaults already exist, no re-save should happen."""
        data = {"application": {"backend_port": 1, "frontend_port": 2, "log_level": "WARN"}}
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        # Values should be preserved
        assert mgr.settings["application"]["backend_port"] == 1
        assert mgr.settings["application"]["frontend_port"] == 2


# ── Module Settings ──────────────────────────────────────────────────────


class TestModuleSettings:
    """Tests for get_module_settings and update_module_settings."""

    def test_get_unknown_module_returns_empty(self, settings_file):
        """Querying a module that doesn't exist should return {}."""
        mgr = _make_manager(settings_file)
        assert mgr.get_module_settings("nonexistent") == {}

    def test_get_returns_correct_module(self, settings_file):
        """Stored module settings should be returned as-is."""
        with open(settings_file, "w") as f:
            json.dump({"application": {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"}, "ui": {"theme": "dark"}}, f)

        mgr = _make_manager(settings_file)
        assert mgr.get_module_settings("ui") == {"theme": "dark"}

    def test_update_new_module_creates_it(self, settings_file):
        """update_module_settings for a new module should create the key."""
        mgr = _make_manager(settings_file)
        mgr.update_module_settings("training", {"lr": 0.001})
        assert mgr.settings["training"]["lr"] == 0.001

    def test_update_merges_dict_keys(self, settings_file):
        """Updating an existing dict module should merge, not replace."""
        mgr = _make_manager(settings_file)
        mgr.update_module_settings("training", {"lr": 0.001, "epochs": 10})
        mgr.update_module_settings("training", {"lr": 0.01, "batch_size": 4})
        assert mgr.settings["training"]["lr"] == 0.01  # updated
        assert mgr.settings["training"]["epochs"] == 10  # preserved
        assert mgr.settings["training"]["batch_size"] == 4  # new

    def test_update_persists_to_disk(self, settings_file):
        """After update, the settings.json file should reflect the change."""
        mgr = _make_manager(settings_file)
        mgr.update_module_settings("alpha", {"enabled": True})

        with open(settings_file) as f:
            on_disk = json.load(f)
        assert on_disk["alpha"]["enabled"] is True

    def test_save_restores_application_if_missing(self, settings_file):
        """If application block is somehow deleted before save, it should be restored."""
        mgr = _make_manager(settings_file)
        del mgr.settings["application"]
        mgr.save()

        with open(settings_file) as f:
            on_disk = json.load(f)
        assert "application" in on_disk
        assert on_disk["application"]["backend_port"] == 8000


# ── Captioning Settings (Typed Accessors) ────────────────────────────────


class TestCaptioningSettings:
    """Tests for typed captioning settings accessor methods."""

    def test_get_captioning_settings_empty(self, settings_file):
        """When no captioning module exists, return defaults."""
        mgr = _make_manager(settings_file)
        caps = mgr.get_captioning_settings()
        assert caps.selected_model == "florence-2"
        assert caps.qwen3_variant == "4B-Instruct"
        assert caps.models == {}

    def test_get_captioning_settings_valid(self, settings_file):
        """Existing captioning data should be parsed into typed models."""
        data = {
            "application": {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"},
            "captioning": {
                "models": {
                    "florence-2": {
                        "active_template_id": "default",
                        "templates": [
                            {
                                "id": "default",
                                "name": "Default",
                                "is_default": True,
                                "readonly": True,
                                "system_prompt": "Describe this image.",
                                "params": {"task_type": "Detailed Caption", "max_tokens": 512}
                            }
                        ]
                    }
                },
                "selected_model": "florence-2",
                "qwen3_variant": "8B-Instruct"
            }
        }
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        caps = mgr.get_captioning_settings()

        assert caps.selected_model == "florence-2"
        assert caps.qwen3_variant == "8B-Instruct"
        assert "florence-2" in caps.models
        model = caps.models["florence-2"]
        assert model.active_template_id == "default"
        assert len(model.templates) == 1
        tpl = model.templates[0]
        assert tpl.id == "default"
        assert tpl.name == "Default"
        assert tpl.is_default is True
        assert tpl.readonly is True
        assert tpl.params["task_type"] == "Detailed Caption"

    def test_update_captioning_settings_roundtrip(self, settings_file):
        """Write captioning settings then read them back."""
        from app.core.schemas.captioning_settings import (
            CaptioningSettings, CaptionModelSettings, CaptionTemplate,
        )
        mgr = _make_manager(settings_file)
        settings = CaptioningSettings(
            selected_model="qwen3-vl",
            qwen3_variant="32B-Thinking",
            models={
                "qwen3-vl": CaptionModelSettings(
                    active_template_id="tpl_123",
                    templates=[
                        CaptionTemplate(
                            id="default", name="Default",
                            is_default=True, readonly=True,
                            params={"temperature": 0.7}
                        ),
                        CaptionTemplate(
                            id="tpl_123", name="Custom",
                            system_prompt="Be concise.",
                            params={"temperature": 0.3, "max_tokens": 256}
                        ),
                    ]
                )
            }
        )
        mgr.update_captioning_settings(settings)

        # Re-read from disk
        mgr2 = _make_manager(settings_file)
        caps = mgr2.get_captioning_settings()
        assert caps.selected_model == "qwen3-vl"
        assert caps.qwen3_variant == "32B-Thinking"
        assert len(caps.models["qwen3-vl"].templates) == 2
        custom = caps.models["qwen3-vl"].templates[1]
        assert custom.system_prompt == "Be concise."
        assert custom.params["max_tokens"] == 256

    def test_captioning_settings_rejects_invalid_template(self, settings_file):
        """Templates with missing required 'id' should raise ValidationError."""
        from pydantic import ValidationError
        from app.core.schemas.captioning_settings import CaptionTemplate

        with pytest.raises(ValidationError):
            CaptionTemplate(name="No ID")  # type: ignore[call-arg]

    def test_captioning_partial_data_uses_defaults(self, settings_file):
        """Partial captioning data should fill in defaults for missing fields."""
        data = {
            "application": {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"},
            "captioning": {
                "selected_model": "youtu-vl"
                # no 'models', no 'qwen3_variant'
            }
        }
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        caps = mgr.get_captioning_settings()
        assert caps.selected_model == "youtu-vl"
        assert caps.qwen3_variant == "4B-Instruct"  # default
        assert caps.models == {}  # default


# ── Per-Model Param Validation ───────────────────────────────────────────


class TestCaptionParamModels:
    """Tests for typed per-model captioning parameter schemas."""

    def test_florence2_params_valid(self):
        from app.core.schemas.captioning_settings import Florence2Params
        p = Florence2Params(task_type="Detailed Caption", max_tokens=512, num_beams=5)
        assert p.task_type == "Detailed Caption"
        assert p.max_tokens == 512

    def test_youtu_vl_params_valid(self):
        from app.core.schemas.captioning_settings import YoutuVLParams
        p = YoutuVLParams(temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=512)
        assert p.temperature == 0.1

    def test_qwen3_vl_params_valid(self):
        from app.core.schemas.captioning_settings import Qwen3VLParams
        p = Qwen3VLParams(temperature=0.7, top_p=0.8, num_beams=1, repetition_penalty=1.2, max_tokens=512, frames=16)
        assert p.frames == 16

    def test_joycaption_params_valid(self):
        from app.core.schemas.captioning_settings import JoyCaptionParams
        p = JoyCaptionParams(caption_type="Descriptive", caption_length="long", temperature=0.6, top_p=0.9, max_tokens=512)
        assert p.caption_type == "Descriptive"

    def test_param_constraint_violation(self):
        from pydantic import ValidationError
        from app.core.schemas.captioning_settings import Florence2Params
        with pytest.raises(ValidationError):
            Florence2Params(task_type="Detailed Caption", max_tokens=2, num_beams=5)  # max_tokens < 64

    def test_validate_all_params_detects_issues(self):
        from app.core.schemas.captioning_settings import (
            CaptioningSettings, CaptionModelSettings, CaptionTemplate,
        )
        settings = CaptioningSettings(
            models={
                "florence-2": CaptionModelSettings(
                    templates=[
                        CaptionTemplate(
                            id="bad", name="Bad",
                            params={"task_type": "INVALID_TYPE", "max_tokens": 512, "num_beams": 5}
                        )
                    ]
                ),
                "unknown-model": CaptionModelSettings(
                    templates=[
                        CaptionTemplate(id="x", name="X", params={"anything": True})
                    ]
                )
            }
        )
        warnings = settings.validate_all_params()
        # florence-2 template has invalid task_type → 1 warning
        assert len(warnings) == 1
        assert "[florence-2]" in warnings[0]
        # unknown-model is skipped silently


# ── Masking Settings (Typed Accessors) ───────────────────────────────────


class TestMaskingSettings:
    """Tests for typed masking settings accessor methods."""

    def test_get_masking_settings_empty(self, settings_file):
        """When no masking module exists, return defaults."""
        mgr = _make_manager(settings_file)
        ms = mgr.get_masking_settings()
        assert ms.selected_model == "sam3"
        assert ms.models == {}
        assert ms.saved_concepts == []

    def test_get_masking_settings_valid(self, settings_file):
        """Existing masking data should be parsed into typed models."""
        data = {
            "application": {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"},
            "masking": {
                "models": {
                    "sam3": {
                        "active_template_id": "default",
                        "templates": [
                            {
                                "id": "default",
                                "name": "Default",
                                "is_default": True,
                                "readonly": True,
                                "params": {"text_prompt": "subject", "multimask_output": True, "max_hole_area": 100}
                            }
                        ]
                    }
                },
                "selected_model": "sam3",
                "saved_concepts": ["hat", "shoes"]
            }
        }
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        ms = mgr.get_masking_settings()
        assert ms.selected_model == "sam3"
        assert ms.saved_concepts == ["hat", "shoes"]
        assert "sam3" in ms.models
        assert len(ms.models["sam3"].templates) == 1
        assert ms.models["sam3"].templates[0].params["text_prompt"] == "subject"

    def test_update_masking_settings_roundtrip(self, settings_file):
        """Write masking settings then read them back."""
        from app.core.schemas.masking_settings import (
            MaskingSettings, MaskingModelSettings, MaskingTemplate,
        )
        mgr = _make_manager(settings_file)
        settings = MaskingSettings(
            selected_model="rembg",
            saved_concepts=["helmet"],
            models={
                "rembg": MaskingModelSettings(
                    active_template_id="tpl_42",
                    templates=[
                        MaskingTemplate(
                            id="default", name="Default",
                            is_default=True, readonly=True,
                            params={"model_name": "u2net", "alpha_matting": False}
                        ),
                        MaskingTemplate(
                            id="tpl_42", name="Alpha",
                            params={"model_name": "u2net", "alpha_matting": True,
                                    "alpha_matting_foreground_threshold": 200}
                        ),
                    ]
                )
            }
        )
        mgr.update_masking_settings(settings)

        mgr2 = _make_manager(settings_file)
        ms = mgr2.get_masking_settings()
        assert ms.selected_model == "rembg"
        assert ms.saved_concepts == ["helmet"]
        assert len(ms.models["rembg"].templates) == 2
        alpha_tpl = ms.models["rembg"].templates[1]
        assert alpha_tpl.params["alpha_matting"] is True


# ── Per-Model Masking Param Validation ───────────────────────────────────


class TestMaskingParamModels:
    """Tests for typed per-model masking parameter schemas."""

    def test_sam3_params_valid(self):
        from app.core.schemas.masking_settings import Sam3Params
        p = Sam3Params(text_prompt="person", max_hole_area=50, multimask_output=False)
        assert p.text_prompt == "person"
        assert p.max_hole_area == 50

    def test_rembg_params_valid(self):
        from app.core.schemas.masking_settings import RembgParams
        p = RembgParams(model_name="isnet-anime", alpha_matting=True,
                        alpha_matting_foreground_threshold=200)
        assert p.model_name == "isnet-anime"

    def test_rembg_constraint_violation(self):
        from pydantic import ValidationError
        from app.core.schemas.masking_settings import RembgParams
        with pytest.raises(ValidationError):
            RembgParams(model_name="invalid_model")  # not in Literal

    def test_validate_all_masking_params_detects_issues(self):
        from app.core.schemas.masking_settings import (
            MaskingSettings, MaskingModelSettings, MaskingTemplate,
        )
        settings = MaskingSettings(
            models={
                "rembg": MaskingModelSettings(
                    templates=[
                        MaskingTemplate(
                            id="bad", name="Bad",
                            params={"model_name": "nonexistent_model", "alpha_matting": False}
                        )
                    ]
                ),
            }
        )
        warnings = settings.validate_all_params()
        assert len(warnings) == 1
        assert "[rembg]" in warnings[0]


# ── Training Settings (Typed Accessors) ──────────────────────────────────


class TestTrainingSettings:
    """Tests for typed training settings accessor methods."""

    def test_get_training_settings_empty(self, settings_file):
        """When no training module exists, return defaults."""
        mgr = _make_manager(settings_file)
        ts = mgr.get_training_settings()
        assert ts.templates == []

    def test_get_training_settings_valid(self, settings_file):
        """Existing training data should be parsed into typed models."""
        data = {
            "application": {"backend_port": 8000, "frontend_port": 4200, "log_level": "INFO"},
            "training": {
                "templates": [
                    {
                        "id": "tpl_1",
                        "name": "My Training",
                        "definition_id": "flux2-dev",
                        "is_default": False,
                        "config": {
                            "lora_name": "test_lora",
                            "global_triggerword": "TestTrigger",
                            "mixed_precision": "bf16",
                            "save_precision": "bf16",
                            "quantization": "none",
                            "te_quantization": "none",
                            "output_dir": "./outputs",
                            "datasets": [{"dataset_name": "TestDS", "caption_prefix": "", "caption_dropout_rate": 0.1, "num_repeats": 1}],
                            "cache_latents": True,
                            "max_train_steps": 1000,
                            "train_batch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "gradient_checkpointing": True,
                            "save_every_n_steps": 250,
                            "optimizer_type": "AdamW8bit",
                            "learning_rate": 0.0001,
                            "weight_decay": 0.01,
                            "network_rank": 16,
                            "network_alpha": 8.0,
                        }
                    }
                ]
            }
        }
        with open(settings_file, "w") as f:
            json.dump(data, f)

        mgr = _make_manager(settings_file)
        ts = mgr.get_training_settings()
        assert len(ts.templates) == 1
        tpl = ts.templates[0]
        assert tpl.name == "My Training"
        assert tpl.definition_id == "flux2-dev"
        assert tpl.config["optimizer_type"] == "AdamW8bit"

        # Deep validation should pass
        warnings = tpl.validate_config()
        assert len(warnings) == 0

    def test_update_training_settings_roundtrip(self, settings_file):
        """Write training settings then read them back."""
        from app.core.schemas.training_settings import TrainingSettings, TrainingTemplate
        mgr = _make_manager(settings_file)
        settings = TrainingSettings(
            templates=[
                TrainingTemplate(
                    id="tpl_a", name="LoRA A", definition_id="sdxl_base_1.0",
                    config={"lora_name": "lora_a", "optimizer_type": "Prodigy",
                            "learning_rate": 1.0, "d_coef": 0.8}
                )
            ]
        )
        mgr.update_training_settings(settings)

        mgr2 = _make_manager(settings_file)
        ts = mgr2.get_training_settings()
        assert len(ts.templates) == 1
        assert ts.templates[0].config["optimizer_type"] == "Prodigy"
        assert ts.templates[0].config["d_coef"] == 0.8

    def test_validate_all_configs_detects_issues(self):
        """Invalid config fields should produce warnings."""
        from app.core.schemas.training_settings import TrainingSettings, TrainingTemplate
        settings = TrainingSettings(
            templates=[
                TrainingTemplate(
                    id="bad", name="Bad Config", definition_id="flux2-dev",
                    config={
                        "lora_name": "test",
                        "mixed_precision": "invalid_precision",
                        "datasets": [{"dataset_name": "DS"}],
                    }
                )
            ]
        )
        warnings = settings.validate_all_configs()
        assert len(warnings) == 1
        assert "[flux2-dev]" in warnings[0]

    def test_training_string_numbers_coerced(self, settings_file):
        """Numeric values stored as strings (from frontend sliders) should coerce correctly."""
        from app.engine.models.base import BaseTrainingConfig
        config = {
            "lora_name": "test",
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "quantization": "none",
            "te_quantization": "none",
            "output_dir": "./outputs",
            "datasets": [{"dataset_name": "DS", "caption_prefix": "", "caption_dropout_rate": "0.1", "num_repeats": 1}],
            "max_train_steps": "6000",
            "network_rank": "32",
            "network_alpha": "32",
            "save_every_n_steps": "250",
            "noise_offset": "0",
            "optimizer_type": "AdamW8bit",
            "learning_rate": 0.0001,
        }
        validated = BaseTrainingConfig.model_validate(config)
        assert validated.max_train_steps == 6000
        assert validated.network_rank == 32
        assert isinstance(validated.max_train_steps, int)




