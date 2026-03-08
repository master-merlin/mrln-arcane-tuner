"""
Tests for checkpoint_inspector — covers metadata extraction, component detection,
adapter detection, and error handling.
"""

import json


from app.engine.utils.checkpoint_inspector import inspect_checkpoint


class TestInspectCheckpoint:
    def test_valid_checkpoint(self, tmp_path):
        """A directory with training_state.json should return valid=True."""
        state = {"global_step": 500, "timestamp": 1234567890.0, "config": {"lr": 0.001}}
        (tmp_path / "training_state.json").write_text(json.dumps(state))

        result = inspect_checkpoint(str(tmp_path))
        assert result["valid"] is True
        assert result["global_step"] == 500
        assert result["config"]["lr"] == 0.001

    def test_missing_path(self):
        """Non-existent path should return valid=False."""
        result = inspect_checkpoint("/nonexistent/checkpoint")
        assert result["valid"] is False
        assert "not exist" in result["error"]

    def test_file_not_directory(self, tmp_path):
        """A file (not directory) should return valid=False."""
        fpath = tmp_path / "not_a_dir.txt"
        fpath.write_text("hello")
        result = inspect_checkpoint(str(fpath))
        assert result["valid"] is False
        assert "not a directory" in result["error"]

    def test_no_training_state(self, tmp_path):
        """Missing training_state.json should return valid=False."""
        result = inspect_checkpoint(str(tmp_path))
        assert result["valid"] is False
        assert "training_state.json" in result["error"]

    def test_corrupt_training_state(self, tmp_path):
        """Malformed JSON should return valid=False."""
        (tmp_path / "training_state.json").write_text("{{{bad json")
        result = inspect_checkpoint(str(tmp_path))
        assert result["valid"] is False
        assert "Failed to parse" in result["error"]


class TestComponentDetection:
    def test_detects_pt_files(self, tmp_path):
        """*.pt files (excluding infra) should appear in components."""
        (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 1}))
        (tmp_path / "unet.pt").write_bytes(b"\x00" * 100)
        (tmp_path / "text_encoder.pt").write_bytes(b"\x00" * 50)

        result = inspect_checkpoint(str(tmp_path))
        assert "unet" in result["components"]
        assert "text_encoder" in result["components"]

    def test_excludes_infrastructure_pt(self, tmp_path):
        """optimizer.pt, scheduler.pt, etc. should not be in components."""
        (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 1}))
        (tmp_path / "optimizer.pt").write_bytes(b"\x00" * 100)
        (tmp_path / "scheduler.pt").write_bytes(b"\x00" * 50)
        (tmp_path / "scaler.pt").write_bytes(b"\x00" * 20)

        result = inspect_checkpoint(str(tmp_path))
        assert result["components"] == []

    def test_capability_booleans(self, tmp_path):
        """has_optimizer, has_ema, etc. should reflect file presence."""
        (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 1}))
        (tmp_path / "optimizer.pt").write_bytes(b"\x00")
        (tmp_path / "ema_shadow.pt").write_bytes(b"\x00")

        result = inspect_checkpoint(str(tmp_path))
        assert result["has_optimizer"] is True
        assert result["has_ema"] is True
        assert result["has_scheduler"] is False


class TestAdapterDetection:
    def test_detects_peft_adapter(self, tmp_path):
        """Subdirectory with adapter_config.json should appear in adapters."""
        (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 1}))
        adapter_dir = tmp_path / "lora_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 200)

        result = inspect_checkpoint(str(tmp_path))
        assert "lora_adapter" in result["adapters"]
        # Adapter files should appear in the files manifest
        assert "lora_adapter/adapter_config.json" in result["files"]


class TestFilesManifest:
    def test_total_size_calculated(self, tmp_path):
        """total_size_bytes should be the sum of all file sizes."""
        (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 1}))
        (tmp_path / "model.pt").write_bytes(b"\x00" * 1024)

        result = inspect_checkpoint(str(tmp_path))
        assert result["total_size_bytes"] > 0
        assert result["files"]["model.pt"] == 1024
