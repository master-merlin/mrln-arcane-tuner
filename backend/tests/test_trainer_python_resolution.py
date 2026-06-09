"""Tests for trainer-subprocess Python interpreter resolution.

In the container there is no project venv (deps are installed system-wide),
so the trainer should use an explicit interpreter via MRLN_TRAINER_PYTHON
rather than warning about a missing Windows venv and guessing.
"""
import os
import sys

from app.engine.models.training_plugin import _resolve_trainer_python


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MRLN_TRAINER_PYTHON", "/usr/bin/python")
    assert _resolve_trainer_python(str(tmp_path)) == "/usr/bin/python"


def test_falls_back_to_current_interpreter_when_no_venv(monkeypatch, tmp_path):
    monkeypatch.delenv("MRLN_TRAINER_PYTHON", raising=False)
    # tmp_path has no venv/ — should use the running interpreter.
    assert _resolve_trainer_python(str(tmp_path)) == sys.executable


def test_detects_posix_venv(monkeypatch, tmp_path):
    monkeypatch.delenv("MRLN_TRAINER_PYTHON", raising=False)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")  # presence is what matters
    assert _resolve_trainer_python(str(tmp_path)) == os.path.abspath(str(venv_python))


def test_detects_windows_venv(monkeypatch, tmp_path):
    monkeypatch.delenv("MRLN_TRAINER_PYTHON", raising=False)
    venv_python = tmp_path / "venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert _resolve_trainer_python(str(tmp_path)) == os.path.abspath(str(venv_python))
