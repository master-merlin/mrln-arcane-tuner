"""LTX 2.3 LoRA-target gating tests — audio modules toggle with train_audio."""

import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver

_AUDIO_MARKERS = (
    "audio_attn1", "audio_attn2", "audio_ff",
    "audio_to_video_attn", "video_to_audio_attn",
)


def _driver(train_audio: bool) -> Ltx2Driver:
    d = Ltx2Driver(definition=None, device=torch.device("cpu"))
    d.train_audio = train_audio
    return d


def _has_audio_target(targets) -> bool:
    return any(any(m in t for m in _AUDIO_MARKERS) for t in targets)


def test_video_only_excludes_audio_modules():
    targets = _driver(train_audio=False).get_lora_targets()
    # Video stream always present.
    assert "attn1.to_q" in targets
    assert "attn2.to_out.0" in targets
    assert "ff.net.0.proj" in targets
    assert "ff.net.2" in targets
    # No audio / cross-modal modules when audio off.
    assert not _has_audio_target(targets)


def test_train_audio_includes_audio_modules():
    targets = _driver(train_audio=True).get_lora_targets()
    # Video stream still present.
    assert "attn1.to_q" in targets
    # Audio sub-stream + cross-modal bridges present.
    assert "audio_attn1.to_q" in targets
    assert "audio_ff.net.0.proj" in targets
    assert "audio_to_video_attn.to_q" in targets
    assert "video_to_audio_attn.to_q" in targets


def test_audio_and_video_targets_are_disjoint_suffixes():
    """The audio modules use ``audio_`` prefixes, so PEFT suffix matching of a
    video target (e.g. ``.attn1.to_q``) never accidentally grabs the audio one
    (``audio_attn1.to_q`` ends with ``_attn1.to_q``, not ``.attn1.to_q``)."""
    targets = _driver(train_audio=True).get_lora_targets()
    video_key = "transformer_blocks.0.attn1.to_q"
    audio_key = "transformer_blocks.0.audio_attn1.to_q"
    # The video target matches the video key but NOT the audio key.
    assert video_key.endswith(".attn1.to_q")
    assert not audio_key.endswith(".attn1.to_q")
    # The audio target matches the audio key.
    assert audio_key.endswith(".audio_attn1.to_q")
    assert "audio_attn1.to_q" in targets
