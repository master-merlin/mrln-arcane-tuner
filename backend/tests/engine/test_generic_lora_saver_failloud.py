"""GenericLoRASaver: silent-failure policy (A0 saver-ledger follow-up).

A LoRA write failure (unwritable target / safetensors error) must ESCAPE
``GenericLoRASaver.save()`` so the CheckpointManager can decide fatality
(final → fail the job; periodic → log-and-continue). A swallowed write let a
training job report "success" while writing no LoRA file — the exact
silent-success bug the A0 SDXL saver fix closed; every ai-toolkit-format
family (hv15, qwen_image, flux1, zimage, …) inherits this base.
"""
from unittest.mock import patch

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class _Saver(GenericLoRASaver):
    architecture_name = "test_arch"


def _tiny_peft_unet():
    model = nn.Sequential(nn.Linear(8, 8))
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["0"])
    return get_peft_model(model, cfg)


def test_save_writes_lora_file(tmp_path):
    out = tmp_path / "lora.safetensors"
    _Saver().save({"unet": _tiny_peft_unet(), "config": {}}, out)
    assert out.exists()


def test_save_failure_propagates(tmp_path):
    """A safetensors write failure must raise out of save(), not be swallowed."""
    out = tmp_path / "lora.safetensors"
    with patch(
        "app.engine.core.pipeline.saver_base.safe_save_file",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            _Saver().save({"unet": _tiny_peft_unet(), "config": {}}, out)
    assert not out.exists()
