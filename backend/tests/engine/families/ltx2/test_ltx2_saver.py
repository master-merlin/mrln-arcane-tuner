"""LTX 2.3 saver tests — ComfyUI ``diffusion_model.*`` keys, audio gating.

Uses a fake PEFT module (``peft_config`` + monkeypatched
``get_peft_model_state_dict``) so no real weights / model are needed.
"""

from unittest.mock import MagicMock

import pytest
import torch
from safetensors import safe_open

from app.engine.models.families.ltx2.saver import Ltx2Saver


def _fake_unet(rank: int = 16):
    unet = MagicMock()
    unet.peft_config = {"default": MagicMock(r=rank, lora_alpha=rank)}
    return unet


def _video_sd():
    return {
        "base_model.model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(16, 64),
        "base_model.model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(64, 16),
        "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.randn(16, 64),
        "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.randn(64, 16),
    }


def _audio_sd():
    sd = _video_sd()
    sd.update({
        "base_model.model.transformer_blocks.0.audio_attn1.to_q.lora_A.weight": torch.randn(16, 64),
        "base_model.model.transformer_blocks.0.audio_attn1.to_q.lora_B.weight": torch.randn(64, 16),
        "base_model.model.transformer_blocks.0.video_to_audio_attn.to_q.lora_A.weight": torch.randn(16, 64),
        "base_model.model.transformer_blocks.0.video_to_audio_attn.to_q.lora_B.weight": torch.randn(64, 16),
    })
    return sd


def _save(tmp_path, mock_sd, config=None):
    saver = Ltx2Saver()
    unet = _fake_unet()
    path = tmp_path / "ltx2.safetensors"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.engine.models.families.ltx2.saver.get_peft_model_state_dict",
            lambda _: mock_sd,
        )
        saver.save(
            components={"unet": unet, "config": config or {}},
            path=path,
        )
    return path


def test_save_uses_diffusion_model_prefix(tmp_path):
    path = _save(tmp_path, _video_sd())
    assert path.exists()
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            assert key.startswith("diffusion_model."), key


def test_save_uses_peft_lora_keys(tmp_path):
    path = _save(tmp_path, _video_sd())
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        assert any("lora_A" in k for k in keys)
        assert any("lora_B" in k for k in keys)
        # No Kohya-style down/up renaming.
        assert not any("lora_down" in k for k in keys)


def test_modelspec_architecture_is_ltx_2_3(tmp_path):
    path = _save(tmp_path, _video_sd())
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        assert meta["modelspec.architecture"] == "ltx-2.3"


def test_audio_keys_absent_when_video_only(tmp_path):
    """A video-only PEFT state dict yields NO audio-stream keys."""
    path = _save(tmp_path, _video_sd())
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
    assert not any("audio_attn" in k or "video_to_audio_attn" in k for k in keys)


def test_audio_keys_present_when_audio_trained(tmp_path):
    """When the PEFT state dict carries audio modules, they are written."""
    path = _save(tmp_path, _audio_sd())
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
    assert any("audio_attn1.to_q" in k for k in keys)
    assert any("video_to_audio_attn.to_q" in k for k in keys)
    # Video keys still present alongside the audio ones.
    assert any("transformer_blocks.0.attn1.to_q" in k for k in keys)


def test_save_default_bf16(tmp_path):
    path = _save(tmp_path, _video_sd(), config={})
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            assert f.get_tensor(key).dtype == torch.bfloat16
