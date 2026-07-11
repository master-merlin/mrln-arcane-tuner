"""SDXL saver: regression test for silent-failure bug class.

A LoRA save failure (e.g. unwritable target / safetensors write error)
must escape ``SDXLSaver.save()`` — never be logged-and-swallowed into a
silent "job succeeded, no file written" outcome.
"""
from unittest.mock import patch

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from app.engine.models.families.sdxl.saver import SDXLSaver


def _tiny_peft_unet():
    model = nn.Sequential(nn.Linear(8, 8))
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["0"])
    return get_peft_model(model, cfg)


def test_save_writes_lora_file(tmp_path):
    unet = _tiny_peft_unet()
    out = tmp_path / "sdxl_lora.safetensors"
    SDXLSaver().save(
        {"unet": unet, "config": {"network_rank": 4, "network_alpha": 4, "save_precision": "fp16"}},
        out,
    )
    assert out.exists()


def test_save_failure_propagates(tmp_path):
    """A safetensors write failure must raise out of save(), not be swallowed."""
    unet = _tiny_peft_unet()
    out = tmp_path / "sdxl_lora.safetensors"

    with patch(
        "app.engine.models.families.sdxl.saver.safe_save_file",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            SDXLSaver().save(
                {"unet": unet, "config": {"network_rank": 4, "network_alpha": 4, "save_precision": "fp16"}},
                out,
            )
    assert not out.exists()
