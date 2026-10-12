"""LANE-150: resume refuses an ideogram4 adapter written on the retired fused layout.

Before LANE-150 the ideogram4 transformer exposed ``attention.qkv`` / ``attention.o``;
the release layout is ``to_q/to_k/to_v/to_out.0``. PEFT's ``load_adapter`` would load
such a directory onto the new model with unmatched keys (fresh adapters) and the
partial-restore warning compares component names only, so the checkpoint layout is
checked by key name before the load.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from app.engine.components.checkpoints import (
    CheckpointManager,
    CheckpointUnreadable,
    legacy_adapter_layout_problem,
)


class _Attn(nn.Module):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        for n in names:
            if n == "to_out":
                setattr(self, n, nn.ModuleList([nn.Linear(8, 8)]))
            else:
                setattr(self, n, nn.Linear(8, 8))


class _Layer(nn.Module):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.attention = _Attn(names)


class _Tiny(nn.Module):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(names)])


def _peft(names: list[str], targets: list[str]):
    return get_peft_model(_Tiny(names), LoraConfig(r=4, lora_alpha=4, target_modules=targets))


def _save(tmp_path, model) -> str:
    return CheckpointManager(str(tmp_path)).save_checkpoint(
        step=5, components={"unet": model}, config={"lora_name": "t"}
    )


def test_legacy_fused_layout_is_refused_by_name(tmp_path):
    path = _save(tmp_path, _peft(["qkv", "o"], ["qkv", "o"]))
    mgr = CheckpointManager(str(tmp_path))
    try:
        mgr.load_checkpoint(path, peft_components={"unet": _peft(["qkv", "o"], ["qkv", "o"])})
    except CheckpointUnreadable as e:
        msg = str(e)
        assert "LANE-150" in msg
        assert path in msg
        assert "Start a new job" in msg
        return
    pytest.fail("LANE-150: resume did not refuse the legacy ideogram4 adapter layout")


def test_release_layout_resumes(tmp_path):
    names = ["to_q", "to_k", "to_v", "to_out"]
    targets = ["to_q", "to_k", "to_v", "to_out.0"]
    path = _save(tmp_path, _peft(names, targets))
    state = CheckpointManager(str(tmp_path)).load_checkpoint(
        path, peft_components={"unet": _peft(names, targets)}
    )
    assert state.adapters_loaded == ["unet"]


def test_helper_missing_file_is_none(tmp_path):
    assert legacy_adapter_layout_problem(str(tmp_path)) is None
