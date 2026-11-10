"""Tests for the trainer-subprocess environment injection in
``StandardPlugin.start_training``.

The trainer subprocess gets its CUDA allocator config (``PYTORCH_CUDA_ALLOC_CONF``)
from run_trainer.py alone: expandable segments where the platform's torch can map
them (Linux), none on Windows. A user-provided value must be respected, not
overwritten.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import app.engine.models.training_plugin as tp
from app.engine.models.training_plugin import StandardPlugin

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_RUN_TRAINER = _BACKEND_ROOT / "run_trainer.py"


def _exec_run_trainer_prefix(preset_value: str | None, os_name: str | None = None) -> str:
    """Exec the leading env-setup of run_trainer.py (everything before
    ``import sys``) in a clean subprocess and return the resulting
    PYTORCH_CUDA_ALLOC_CONF. Isolated so it can't mutate the test process.
    ``os_name`` stands in for ``os.name``, so each platform's choice is tested on any.
    """
    code = (
        "import os\n"
        "os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)\n"
        + (f"os.environ['PYTORCH_CUDA_ALLOC_CONF'] = {preset_value!r}\n" if preset_value else "")
        + (f"os.name = {os_name!r}\n" if os_name else "")
        + "src = open(r'" + str(_RUN_TRAINER) + "', encoding='utf-8').read().split('import sys', 1)[0]\n"
        "exec(compile(src, 'run_trainer_prefix', 'exec'))\n"
        "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_BACKEND_ROOT),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_run_trainer_sets_expandable_segments_off_windows():
    assert _exec_run_trainer_prefix(preset_value=None, os_name="posix") == "expandable_segments:True"


def test_run_trainer_sets_no_expandable_segments_on_windows():
    """Windows torch up to 2.12 IGNORED the option (no segment was ever expandable), and 2.15
    honours it with a virtual-memory mapping WDDM refuses: the first pool growth fails with
    ``CUDA driver error: unknown error`` (cuMemSetAccess), so no training starts (LANE-138)."""
    assert _exec_run_trainer_prefix(preset_value=None, os_name="nt") == ""


def test_run_trainer_respects_preset_alloc_conf():
    for os_name in ("nt", "posix"):
        assert _exec_run_trainer_prefix(preset_value="max_split_size_mb:64", os_name=os_name) == "max_split_size_mb:64"


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


def test_the_launcher_leaves_the_allocator_choice_to_run_trainer(monkeypatch, tmp_path):
    """One place decides: run_trainer.py, which every training runs through, before torch."""
    env = _run_start_training(monkeypatch, tmp_path, {"PATH": "/x"})
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env["env"]


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
