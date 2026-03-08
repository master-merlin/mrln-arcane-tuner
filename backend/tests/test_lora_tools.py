"""
Tests for Phase 9: LoRA Tooling (inspect + resize).

Covers:
- inspect_lora: metadata reading, format detection, rank/alpha extraction,
  component breakdown, module counting, weight statistics, training params,
  tag frequency, dataset info, layer details, norm summary
- resize_lora: SVD-based rank change (down and up), alpha scaling,
  weight shape validation, metadata preservation
- Error handling: missing files, invalid rank, no pairs
"""

import json
import os
import pytest
import torch
from safetensors.torch import save_file, load_file

from app.engine.utils.lora_tools import (
    inspect_lora,
    resize_lora,
    _detect_format,
    _extract_rank,
    _extract_alpha,
    _find_lora_pairs,
    _compute_weight_stats,
    _parse_training_params,
    _parse_tag_frequency,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def kohya_lora(tmp_path):
    """Create a minimal Kohya-format LoRA safetensors file with rich metadata."""
    rank = 16
    alpha = 8.0

    state_dict = {
        # UNet module 1
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": torch.randn(rank, 320),
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight": torch.randn(320, rank),
        "lora_unet_down_blocks_0_attentions_0_to_q.alpha": torch.tensor(alpha),
        # UNet module 2
        "lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight": torch.randn(rank, 320),
        "lora_unet_down_blocks_0_attentions_0_to_k.lora_up.weight": torch.randn(320, rank),
        "lora_unet_down_blocks_0_attentions_0_to_k.alpha": torch.tensor(alpha),
        # TE1 module
        "lora_te1_text_model_encoder_layers_0_q_proj.lora_down.weight": torch.randn(rank, 768),
        "lora_te1_text_model_encoder_layers_0_q_proj.lora_up.weight": torch.randn(768, rank),
        "lora_te1_text_model_encoder_layers_0_q_proj.alpha": torch.tensor(alpha),
    }

    metadata = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "networks.lora",
        "ss_learning_rate": "0.0001",
        "ss_unet_lr": "0.0001",
        "ss_text_encoder_lr": "5e-05",
        "ss_optimizer": "AdamW(weight_decay=0.01)",
        "ss_lr_scheduler": "cosine",
        "ss_epoch": "10",
        "ss_num_train_images": "57",
        "ss_num_reg_images": "0",
        "ss_num_batches_per_epoch": "57",
        "ss_batch_size_per_device": "1",
        "ss_gradient_accumulation_steps": "4",
        "ss_noise_offset": "0.05",
        "ss_min_snr_gamma": "5.0",
        "ss_resolution": "512,512",
        "ss_base_model_version": "sdxl_1.0",
        "ss_sd_model_name": "sdxl_base",
        "ss_tag_frequency": json.dumps({
            "50_photos": {
                "1girl": 45,
                "solo": 40,
                "looking_at_viewer": 30,
                "smile": 20,
            }
        }),
        "ss_dataset_config": json.dumps({
            "datasets": [{
                "subsets": [{
                    "image_dir": "/data/training/photos",
                    "num_repeats": 50,
                }]
            }]
        }),
        "ss_network_args": json.dumps({
            "block_dims": "4,4,4,8,8,8,16,16,16,8,8,8",
            "block_alphas": "4,4,4,8,8,8,16,16,16,8,8,8",
        }),
        "software": "test",
    }

    path = str(tmp_path / "kohya_lora.safetensors")
    save_file(state_dict, path, metadata=metadata)
    return path, rank, alpha


@pytest.fixture
def aitoolkit_lora(tmp_path):
    """Create a minimal ai-toolkit format LoRA safetensors file."""
    rank = 8

    state_dict = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.randn(rank, 3072),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.randn(3072, rank),
        "diffusion_model.double_blocks.0.img_mlp.0.lora_A.weight": torch.randn(rank, 3072),
        "diffusion_model.double_blocks.0.img_mlp.0.lora_B.weight": torch.randn(3072, rank),
    }

    path = str(tmp_path / "flux_lora.safetensors")
    save_file(state_dict, path)
    return path, rank


# ── Inspect Tests ────────────────────────────────────────────────────────


class TestInspectLora:
    """Tests for inspect_lora()."""

    def test_inspect_kohya(self, kohya_lora):
        """Should correctly inspect a Kohya-format LoRA."""
        path, rank, alpha = kohya_lora
        result = inspect_lora(path)

        assert result["path"] == path
        assert result["format"] == "kohya"
        assert result["rank"] == rank
        assert result["alpha"] == alpha
        assert result["total_keys"] == 9  # 6 weights + 3 alpha
        assert result["lora_modules"] == 3
        assert result["file_size_mb"] > 0

    def test_inspect_aitoolkit(self, aitoolkit_lora):
        """Should correctly inspect an ai-toolkit format LoRA."""
        path, rank = aitoolkit_lora
        result = inspect_lora(path)

        assert result["format"] == "ai-toolkit"
        assert result["rank"] == rank
        assert result["lora_modules"] == 2
        assert result["total_keys"] == 4

    def test_inspect_components_breakdown(self, kohya_lora):
        """Should break down key counts by component."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        comps = result["components"]

        assert "unet" in comps
        assert "text_encoder_1" in comps
        assert comps["unet"] == 6  # 2 modules × (down + up + alpha)
        assert comps["text_encoder_1"] == 3  # 1 module × (down + up + alpha)

    def test_inspect_dtype_detection(self, kohya_lora):
        """Should detect the dtype of stored weights."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)

        assert result["dtype"] == "torch.float32"

    def test_inspect_metadata(self, kohya_lora):
        """Should include safetensors metadata."""
        path, rank, alpha = kohya_lora
        result = inspect_lora(path)
        meta = result["metadata"]

        assert meta["ss_network_dim"] == str(rank)
        assert meta["ss_network_alpha"] == str(alpha)

    def test_inspect_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            inspect_lora("/nonexistent/path.safetensors")

    def test_inspect_module_list(self, kohya_lora):
        """Should return sorted list of module names."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)

        assert isinstance(result["module_list"], list)
        assert len(result["module_list"]) == 3
        assert result["module_list"] == sorted(result["module_list"])


# ── Weight Stats Tests ───────────────────────────────────────────────────


class TestWeightStats:
    """Tests for weight statistics (magnitude, strength)."""

    def test_weight_stats_present(self, kohya_lora):
        """inspect_lora should include weight_stats."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        assert "weight_stats" in result
        ws = result["weight_stats"]
        assert "unet" in ws
        assert "text_encoder_1" in ws

    def test_weight_stats_fields(self, kohya_lora):
        """Each component should have magnitude, strength, num_tensors, total_params."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        unet = result["weight_stats"]["unet"]

        assert "avg_magnitude" in unet
        assert "avg_strength" in unet
        assert "num_tensors" in unet
        assert "total_params" in unet
        assert unet["avg_magnitude"] > 0
        assert unet["avg_strength"] > 0
        assert unet["num_tensors"] == 4  # 2 modules × (down + up)
        assert unet["total_params"] > 0

    def test_weight_stats_helper(self):
        """_compute_weight_stats should work on raw state dicts."""
        sd = {
            "lora_unet_foo.lora_down.weight": torch.ones(4, 16),
            "lora_unet_foo.lora_up.weight": torch.ones(16, 4),
        }
        stats = _compute_weight_stats(sd)
        assert "unet" in stats
        assert stats["unet"]["avg_strength"] == 1.0  # all ones
        assert stats["unet"]["num_tensors"] == 2


# ── Layer Details Tests ──────────────────────────────────────────────────


class TestLayerDetails:
    """Tests for per-layer breakdown."""

    def test_layer_details_present(self, kohya_lora):
        """inspect_lora should include layer_details."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        assert "layer_details" in result
        details = result["layer_details"]
        assert len(details) == 3  # 2 unet + 1 te1 modules

    def test_layer_detail_fields(self, kohya_lora):
        """Each layer should have comprehensive metrics."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        layer = result["layer_details"][0]

        assert "module" in layer
        assert "component" in layer
        assert "rank" in layer
        assert "in_features" in layer
        assert "out_features" in layer
        assert "params" in layer
        assert "norm_a" in layer
        assert "norm_b" in layer
        assert "norm_delta" in layer
        assert "delta_mean" in layer
        assert "delta_std" in layer
        assert "magnitude" in layer
        assert "strength" in layer

    def test_layer_detail_values(self, kohya_lora):
        """Layer details should have correct dimensions."""
        path, rank, _ = kohya_lora
        result = inspect_lora(path)

        for layer in result["layer_details"]:
            assert layer["rank"] == rank
            assert layer["norm_a"] > 0
            assert layer["norm_b"] > 0
            assert layer["norm_delta"] > 0


# ── Training Params Tests ────────────────────────────────────────────────


class TestTrainingParams:
    """Tests for ss_ metadata parsing."""

    def test_training_params_present(self, kohya_lora):
        """inspect_lora should include training_params."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        tp = result["training_params"]

        assert tp["learning_rate"] == 0.0001
        assert tp["unet_lr"] == 0.0001
        assert tp["text_encoder_lr"] == 5e-05
        assert tp["optimizer"] == "AdamW(weight_decay=0.01)"
        assert tp["scheduler"] == "cosine"
        assert tp["epochs"] == 10
        assert tp["train_images"] == 57
        assert tp["noise_offset"] == 0.05
        assert tp["min_snr_gamma"] == 5.0

    def test_training_params_empty(self, aitoolkit_lora):
        """ai-toolkit LoRA without ss_ keys → empty training_params."""
        path, _ = aitoolkit_lora
        result = inspect_lora(path)
        assert result["training_params"] == {}

    def test_parse_training_params_helper(self):
        """_parse_training_params should parse the mapping correctly."""
        meta = {
            "ss_learning_rate": "0.001",
            "ss_epoch": "5",
            "ss_optimizer": "Prodigy",
        }
        params = _parse_training_params(meta)
        assert params["learning_rate"] == 0.001
        assert params["epochs"] == 5
        assert params["optimizer"] == "Prodigy"


# ── Tag Frequency Tests ──────────────────────────────────────────────────


class TestTagFrequency:
    """Tests for tag frequency parsing."""

    def test_tag_frequency_present(self, kohya_lora):
        """Should parse tag frequency from metadata."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        tf = result["tag_frequency"]

        assert "50_photos" in tf
        tags = tf["50_photos"]
        assert tags[0]["tag"] == "1girl"
        assert tags[0]["count"] == 45
        assert len(tags) == 4

    def test_tag_frequency_sorted(self, kohya_lora):
        """Tags should be sorted descending by count."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        tags = result["tag_frequency"]["50_photos"]
        counts = [t["count"] for t in tags]
        assert counts == sorted(counts, reverse=True)

    def test_tag_frequency_empty(self, aitoolkit_lora):
        """Empty when no ss_tag_frequency metadata."""
        path, _ = aitoolkit_lora
        result = inspect_lora(path)
        assert result["tag_frequency"] == {}

    def test_parse_tag_frequency_invalid_json(self):
        """Invalid JSON should return empty."""
        assert _parse_tag_frequency({"ss_tag_frequency": "not_json"}) == {}


# ── Dataset Info Tests ───────────────────────────────────────────────────


class TestDatasetInfo:
    """Tests for dataset config parsing."""

    def test_dataset_info_present(self, kohya_lora):
        """Should parse dataset config from metadata."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        ds = result["dataset_info"]

        assert ds["train_images"] == 57
        assert ds["regularization_images"] == 0
        assert len(ds["directories"]) == 1
        assert ds["directories"][0]["directory"] == "/data/training/photos"
        assert ds["directories"][0]["num_repeats"] == 50

    def test_dataset_info_empty(self, aitoolkit_lora):
        """Empty when no dataset metadata."""
        path, _ = aitoolkit_lora
        result = inspect_lora(path)
        assert result["dataset_info"] == {}


# ── Block Config Tests ───────────────────────────────────────────────────


class TestBlockConfig:
    """Tests for variable block dims/alphas."""

    def test_block_config_present(self, kohya_lora):
        """Should parse block dims/alphas from network args."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        bc = result["block_config"]

        assert "block_dims" in bc
        assert len(bc["block_dims"]) == 12
        assert bc["block_dims"][0] == 4
        assert bc["block_dims"][6] == 16

    def test_block_config_empty(self, aitoolkit_lora):
        """Empty when no block config."""
        path, _ = aitoolkit_lora
        result = inspect_lora(path)
        assert result["block_config"] == {}


# ── Norm Summary Tests ───────────────────────────────────────────────────


class TestNormSummary:
    """Tests for norm distribution summary."""

    def test_norm_summary_present(self, kohya_lora):
        """Should compute norm summary across layers."""
        path, _, _ = kohya_lora
        result = inspect_lora(path)
        ns = result["norm_summary"]

        assert "mean_norm" in ns
        assert "std_norm" in ns
        assert "max_norm" in ns
        assert "min_norm" in ns
        assert "max_norm_layer" in ns
        assert "min_norm_layer" in ns
        assert "total_layers" in ns
        assert ns["total_layers"] == 3
        assert ns["max_norm"] >= ns["min_norm"]


# ── Resize Tests ─────────────────────────────────────────────────────────


class TestResizeLora:
    """Tests for resize_lora()."""

    def test_resize_rank_down(self, kohya_lora, tmp_path):
        """Should reduce rank from 16 to 8 and update alpha proportionally."""
        input_path, old_rank, old_alpha = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")

        result = resize_lora(input_path, output_path, new_rank=8)

        assert result["old_rank"] == 16
        assert result["new_rank"] == 8
        assert result["old_alpha"] == old_alpha
        assert result["new_alpha"] == old_alpha * (8 / 16)  # 4.0
        assert result["modules_resized"] == 3  # 2 unet + 1 te1
        assert os.path.exists(output_path)

    def test_resize_rank_up(self, kohya_lora, tmp_path):
        """Should increase rank from 16 to 32 with zero-padding."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized_up.safetensors")

        result = resize_lora(input_path, output_path, new_rank=32)

        assert result["new_rank"] == 32
        assert result["modules_resized"] == 3

        # Verify weight shapes
        sd = load_file(output_path)
        for key, value in sd.items():
            if "lora_down" in key:
                assert value.shape[0] == 32  # [new_rank, in_features]
            elif "lora_up" in key:
                assert value.shape[1] == 32  # [out_features, new_rank]

    def test_resize_weight_shapes(self, kohya_lora, tmp_path):
        """Resized weights should have correct dimensions."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")
        new_rank = 4

        resize_lora(input_path, output_path, new_rank=new_rank)
        sd = load_file(output_path)

        for key, value in sd.items():
            if "lora_down" in key:
                assert value.shape[0] == new_rank
            elif "lora_up" in key:
                assert value.shape[1] == new_rank

    def test_resize_explicit_alpha(self, kohya_lora, tmp_path):
        """Should use explicitly provided alpha instead of proportional."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")

        result = resize_lora(input_path, output_path, new_rank=8, new_alpha=1.0)

        assert result["new_alpha"] == 1.0

    def test_resize_updates_alpha_tensors(self, kohya_lora, tmp_path):
        """Alpha tensors in the output should match new_alpha."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")

        resize_lora(input_path, output_path, new_rank=8, new_alpha=4.0)
        sd = load_file(output_path)

        for key, value in sd.items():
            if ".alpha" in key:
                assert value.item() == 4.0

    def test_resize_preserves_metadata(self, kohya_lora, tmp_path):
        """Output file metadata should include rank/alpha and resize history."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")

        resize_lora(input_path, output_path, new_rank=8)

        result = inspect_lora(output_path)
        meta = result["metadata"]

        assert meta["ss_network_dim"] == "8"
        assert meta["arcane_resized_from_rank"] == "16"

    def test_resize_aitoolkit_format(self, aitoolkit_lora, tmp_path):
        """Should resize ai-toolkit format LoRAs correctly."""
        input_path, old_rank = aitoolkit_lora
        output_path = str(tmp_path / "resized_flux.safetensors")

        result = resize_lora(input_path, output_path, new_rank=4)

        assert result["old_rank"] == 8
        assert result["new_rank"] == 4
        assert result["modules_resized"] == 2

    def test_resize_save_dtype(self, kohya_lora, tmp_path):
        """Should apply specified save dtype to output weights."""
        input_path, _, _ = kohya_lora
        output_path = str(tmp_path / "resized.safetensors")

        resize_lora(input_path, output_path, new_rank=8, save_dtype=torch.float16)
        sd = load_file(output_path)

        for key, value in sd.items():
            if "lora_down" in key or "lora_up" in key:
                assert value.dtype == torch.float16

    def test_resize_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            resize_lora("/nonexistent.safetensors", str(tmp_path / "out.safetensors"), 8)

    def test_resize_invalid_rank(self, kohya_lora, tmp_path):
        """Should raise ValueError for rank < 1."""
        input_path, _, _ = kohya_lora
        with pytest.raises(ValueError, match="new_rank must be"):
            resize_lora(input_path, str(tmp_path / "out.safetensors"), new_rank=0)


# ── Internal Helper Tests ────────────────────────────────────────────────


class TestFormatDetection:
    """Tests for _detect_format()."""

    def test_detects_kohya(self):
        keys = {"lora_unet_foo.lora_down.weight": torch.zeros(1)}
        assert _detect_format(keys, {}) == "kohya"

    def test_detects_aitoolkit(self):
        keys = {"diffusion_model.foo.lora_A.weight": torch.zeros(1)}
        assert _detect_format(keys, {}) == "ai-toolkit"

    def test_detects_peft(self):
        keys = {"base_model.model.foo.lora_A.weight": torch.zeros(1)}
        assert _detect_format(keys, {}) == "peft"

    def test_unknown_format(self):
        keys = {"random_key": torch.zeros(1)}
        assert _detect_format(keys, {}) == "unknown"


class TestRankExtraction:
    """Tests for _extract_rank()."""

    def test_rank_from_metadata(self):
        assert _extract_rank({}, {"ss_network_dim": "32"}) == 32

    def test_rank_from_weights(self):
        sd = {"foo.lora_A.weight": torch.randn(16, 128)}
        assert _extract_rank(sd, {}) == 16

    def test_rank_from_kohya_weights(self):
        sd = {"foo.lora_down.weight": torch.randn(8, 64)}
        assert _extract_rank(sd, {}) == 8

    def test_rank_none_when_empty(self):
        assert _extract_rank({}, {}) is None


class TestAlphaExtraction:
    """Tests for _extract_alpha()."""

    def test_alpha_from_metadata(self):
        assert _extract_alpha({}, {"ss_network_alpha": "8.0"}) == 8.0

    def test_alpha_from_tensor(self):
        sd = {"foo.alpha": torch.tensor(4.0)}
        assert _extract_alpha(sd, {}) == 4.0

    def test_alpha_none_when_empty(self):
        assert _extract_alpha({}, {}) is None


class TestPairFinding:
    """Tests for _find_lora_pairs()."""

    def test_finds_peft_pairs(self):
        sd = {
            "foo.lora_A.weight": torch.zeros(1),
            "foo.lora_B.weight": torch.zeros(1),
            "bar.lora_A.weight": torch.zeros(1),
            # bar.lora_B missing → incomplete
        }
        pairs = _find_lora_pairs(sd, "peft")
        assert len(pairs) == 1
        assert "foo" in pairs

    def test_finds_kohya_pairs(self):
        sd = {
            "mod.lora_down.weight": torch.zeros(1),
            "mod.lora_up.weight": torch.zeros(1),
        }
        pairs = _find_lora_pairs(sd, "kohya")
        assert len(pairs) == 1
        assert "mod" in pairs
