"""Tests for the trainer-subprocess environment injection in
``StandardPlugin.start_training``.

The trainer subprocess must launch with an anti-fragmentation CUDA allocator
config (``PYTORCH_CUDA_ALLOC_CONF``) so the in-training sampler's transient
VRAM spikes don't fragment the caching-allocator pool into a Windows/WDDM
shared-memory spill (the "freeze" failure mode). A user-provided value must be
respected, not overwritten.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app.engine.models.training_plugin as tp
from app.engine.models.training_plugin import StandardPlugin


def _run_start_training(monkeypatch, tmp_path, parent_env):
    """Call start_training with Popen monkeypatched; return the captured env."""
    captured = {}

    def _fake_popen(cmd, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(tp.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(tp.os, "environ", parent_env)

    plugin = StandardPlugin()
    plugin.start_training(
        {
            "definition_id": "sdxl_base_1.0",
            "output_dir": str(tmp_path),
            "job_id": "test-job",
        }
    )
    return captured


def test_injects_alloc_conf_when_absent(monkeypatch, tmp_path):
    env = _run_start_training(monkeypatch, tmp_path, {"PATH": "/x"})
    child_env = env["env"]
    assert (
        child_env["PYTORCH_CUDA_ALLOC_CONF"]
        == "garbage_collection_threshold:0.8"
    )


def test_respects_user_provided_alloc_conf(monkeypatch, tmp_path):
    parent = {"PATH": "/x", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    env = _run_start_training(monkeypatch, tmp_path, parent)
    child_env = env["env"]
    # User override must be preserved verbatim.
    assert child_env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_child_env_is_a_copy_not_parent(monkeypatch, tmp_path):
    parent = {"PATH": "/x"}
    env = _run_start_training(monkeypatch, tmp_path, parent)
    # Injection must not mutate the parent process environment.
    assert "PYTORCH_CUDA_ALLOC_CONF" not in parent
    assert env["env"] is not parent
