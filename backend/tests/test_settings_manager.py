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


# ── W4.T6: atomic writes + write lock + honest cache ─────────────────────


class TestAtomicWriteAndLock:
    """save() must never leave settings.json truncated/corrupt (it holds the
    HF token and the jobs.auto_queue/auto_resume prefs the queue reads), and
    concurrent update_module_settings callers must not lose each other's
    writes."""

    def test_save_survives_crash_mid_write(self, settings_file, monkeypatch):
        """A crash partway through json.dump must leave the ORIGINAL file
        content intact (atomic tmp-file + os.replace), not a truncated one."""
        mgr = _make_manager(settings_file)
        mgr.settings["custom"] = {"key": "before-crash"}
        mgr.save()
        with open(settings_file) as f:
            original = f.read()

        import json as json_module

        def _boom(obj, fp, **kwargs):
            fp.write('{"partial": true')  # simulate partial output
            raise RuntimeError("boom mid-write")

        monkeypatch.setattr(json_module, "dump", _boom)

        mgr.settings["custom"] = {"key": "after-crash-attempt"}
        mgr.save()  # must not raise, must not corrupt the real file

        with open(settings_file) as f:
            after = f.read()
        assert after == original
        assert not os.path.exists(settings_file + ".tmp")

    def test_concurrent_update_module_settings_no_lost_update(self, settings_file):
        """Two threads hammering update_module_settings on DIFFERENT modules
        must both end up fully persisted — no lost update from an
        unsynchronized load-merge-save race."""
        import threading

        mgr = _make_manager(settings_file)

        def worker(module_name: str):
            for i in range(100):
                mgr.update_module_settings(module_name, {f"k{i}": i})

        t1 = threading.Thread(target=worker, args=("mod_a",))
        t2 = threading.Thread(target=worker, args=("mod_b",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        with open(settings_file) as f:
            on_disk = json.load(f)
        assert len(on_disk["mod_a"]) == 100
        assert len(on_disk["mod_b"]) == 100
        assert on_disk["mod_a"]["k99"] == 99
        assert on_disk["mod_b"]["k99"] == 99

    def test_get_module_settings_reads_disk_once_when_unchanged(
        self, settings_file, monkeypatch,
    ):
        """get_module_settings must not re-read the file from disk on every
        call — only when the on-disk mtime actually changed (external edit)."""
        mgr = _make_manager(settings_file)
        mgr.update_module_settings("ui", {"theme": "dark"})

        open_calls = {"count": 0}
        real_open = open

        def _counting_open(*args, **kwargs):
            open_calls["count"] += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", _counting_open)

        first = mgr.get_module_settings("ui")
        second = mgr.get_module_settings("ui")

        assert first == second == {"theme": "dark"}
        assert open_calls["count"] <= 1

    def test_get_module_settings_reloads_after_external_edit(self, settings_file):
        """An external writer touching settings.json (different mtime) must
        be picked up — the cache must not go stale forever."""
        import time as time_module

        mgr = _make_manager(settings_file)
        assert mgr.get_module_settings("ui") == {}

        # Simulate an external process editing the file directly.
        time_module.sleep(0.01)
        with open(settings_file) as f:
            data = json.load(f)
        data["ui"] = {"theme": "light"}
        with open(settings_file, "w") as f:
            json.dump(data, f)
        # Force a distinct mtime on filesystems with coarse resolution.
        stat = os.stat(settings_file)
        os.utime(settings_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        assert mgr.get_module_settings("ui") == {"theme": "light"}


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

    def test_validate_params_detects_issues(self):
        """CaptionModelSettings.validate_params flags templates with bad params."""
        from app.core.schemas.captioning_settings import CaptionModelSettings, CaptionTemplate

        model_settings = CaptionModelSettings(
            templates=[
                CaptionTemplate(
                    id="bad", name="Bad",
                    params={"task_type": "INVALID_TYPE", "max_tokens": 512, "num_beams": 5}
                )
            ]
        )
        warnings = model_settings.validate_params("florence-2")
        assert len(warnings) == 1
        assert "Bad" in warnings[0]

    def test_validate_params_skips_unknown_model(self):
        """Unknown model ids are skipped silently (no schema to validate against)."""
        from app.core.schemas.captioning_settings import CaptionModelSettings, CaptionTemplate

        model_settings = CaptionModelSettings(
            templates=[CaptionTemplate(id="x", name="X", params={"anything": True})]
        )
        warnings = model_settings.validate_params("unknown-model")
        assert warnings == []

    def test_caption_template_rejects_missing_id(self):
        """Templates with missing required 'id' should raise ValidationError."""
        from pydantic import ValidationError
        from app.core.schemas.captioning_settings import CaptionTemplate

        with pytest.raises(ValidationError):
            CaptionTemplate(name="No ID")  # type: ignore[call-arg]


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

    def test_validate_params_detects_issues(self):
        """MaskingModelSettings.validate_params flags templates with bad params."""
        from app.core.schemas.masking_settings import MaskingModelSettings, MaskingTemplate

        model_settings = MaskingModelSettings(
            templates=[
                MaskingTemplate(
                    id="bad", name="Bad",
                    params={"model_name": "nonexistent_model", "alpha_matting": False}
                )
            ]
        )
        warnings = model_settings.validate_params("rembg")
        assert len(warnings) == 1
        assert "Bad" in warnings[0]


# ── Training Config Coercion (live functionality) ─────────────────────────


class TestTrainingConfigCoercion:
    """Tests for BaseTrainingConfig's tolerant coercion of frontend-supplied values.

    The legacy training-settings template schema (TrainingSettings/TrainingTemplate)
    was removed as dead code; this test targets the still-live BaseTrainingConfig.
    """

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




