"""A checkpoint saves the random streams and a resume continues them.

Without this a seeded job's resume replays the stream from its seed: the noise
and timesteps of steps 0, 1, 2 come again. Every test goes through the real
``CheckpointManager``, so the file is read back by ``weights_only`` loading.

``torch.cuda``'s state calls are stubbed where a test needs devices, so that
logic runs on any machine; one test skips without CUDA and uses the real thing.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from structlog.testing import capture_logs

from app.engine.components.checkpoints import (
    CheckpointManager,
    CheckpointUnreadable,
    checkpoint_resumable,
)


def _snapshot() -> dict:
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().tolist(),
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
    }


@pytest.fixture(autouse=True)
def _restore_streams():
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _seed_streams(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _draws() -> dict:
    # The normal draws cover the half-used pair Python and NumPy each keep.
    return {
        "torch": torch.rand(4).tolist(),
        "python": [random.random(), random.gauss(0.0, 1.0)],
        "numpy": [*np.random.random(4).tolist(), float(np.random.standard_normal())],
    }


def _save(run: Path, *, seed: object = 42, is_final: bool = False) -> Path:
    model = nn.Linear(2, 2)
    config: dict = {"lora_name": "t"}
    if seed is not None:
        config["seed"] = seed
    return Path(
        CheckpointManager(str(run)).save_checkpoint(10, {"model": model}, config=config, is_final=is_final)
    )


def _load(folder: Path, **kwargs) -> None:
    CheckpointManager(str(folder.parent)).load_checkpoint(str(folder), **kwargs)


def _stub_cuda(monkeypatch, *, count: int, saved: list | None = None, initialized: bool = True) -> list:
    """Stub the CUDA state calls; returns the ``(device, state)`` of each restore."""
    calls: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: list(saved or []))
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device=-1: calls.append((device, state.clone()))
    )
    return calls


def _cuda_states(n: int) -> list[torch.Tensor]:
    return [torch.full((16,), i + 1, dtype=torch.uint8) for i in range(n)]


def _events(logs: list[dict], name: str) -> list[dict]:
    return [e for e in logs if e["event"] == name]


# ── The streams continue ─────────────────────────────────────────────────


def test_next_draws_equal_what_the_saving_process_would_have_drawn(tmp_path):
    _seed_streams(5)
    _draws()
    # An odd number of normals, so each of the two keeps the pair's other half.
    random.gauss(0.0, 1.0)
    np.random.standard_normal()
    folder = _save(tmp_path)
    expected = _draws()

    _seed_streams(999)
    _load(folder)

    assert _draws() == expected


def test_the_file_holds_plain_types_and_tensors(tmp_path):
    folder = _save(tmp_path)

    saved = torch.load(folder / "rng_state.pt", map_location="cpu", weights_only=True)

    assert saved["seed"] == 42
    assert isinstance(saved["torch"], torch.Tensor)


def test_load_restores_after_everything_else_it_loads(tmp_path):
    """Loading an adapter may draw; that must not shift the restored streams."""

    class DrawingAdapter:
        def save_pretrained(self, path: str) -> None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump({"peft_type": "LORA"}, f)

        def load_adapter(self, path: str, adapter_name: str, is_trainable: bool) -> None:
            torch.rand(8)
            random.random()
            np.random.random(8)

        def parameters(self):
            return iter([])

    adapter = DrawingAdapter()
    _seed_streams(5)
    folder = Path(
        CheckpointManager(str(tmp_path)).save_checkpoint(10, {"unet": adapter}, config={"lora_name": "t"})
    )
    expected = _draws()

    _seed_streams(999)
    _load(folder, peft_components={"unet": adapter})

    assert _draws() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_cuda_draws_continue(tmp_path):
    torch.cuda.manual_seed_all(5)
    torch.rand(4, device="cuda")
    folder = _save(tmp_path)
    expected = torch.rand(4, device="cuda").tolist()

    torch.cuda.manual_seed_all(999)
    _load(folder)

    assert torch.rand(4, device="cuda").tolist() == expected


# ── CUDA, stubbed ────────────────────────────────────────────────────────


def test_all_saved_cuda_streams_are_passed_on_restore(tmp_path, monkeypatch):
    saved = _cuda_states(2)
    calls = _stub_cuda(monkeypatch, count=2, saved=saved)
    folder = _save(tmp_path)

    _load(folder)

    assert [device for device, _ in calls] == [0, 1]
    assert all(torch.equal(state, saved[device]) for device, state in calls)


def test_more_saved_devices_than_present_restores_those_present(tmp_path, monkeypatch):
    saved = _cuda_states(3)
    calls = _stub_cuda(monkeypatch, count=3, saved=saved)
    folder = _save(tmp_path)
    calls = _stub_cuda(monkeypatch, count=1, saved=[])

    with capture_logs() as logs:
        _load(folder)

    assert [device for device, _ in calls] == [0]
    assert torch.equal(calls[0][1], saved[0])
    assert len(_events(logs, "rng_state_fewer_cuda_devices_on_resume")) == 1


def test_a_cpu_only_checkpoint_on_a_cuda_machine_leaves_cuda_alone(tmp_path, monkeypatch):
    _stub_cuda(monkeypatch, count=0)
    folder = _save(tmp_path)
    calls = _stub_cuda(monkeypatch, count=2, saved=_cuda_states(2))

    with capture_logs() as logs:
        _load(folder)

    assert calls == []
    assert _events(logs, "rng_state_fewer_cuda_devices_on_resume") == []


def test_a_cuda_stream_the_device_refuses_warns_and_restores_nothing(tmp_path, monkeypatch):
    _stub_cuda(monkeypatch, count=1, saved=_cuda_states(1))
    folder = _save(tmp_path)

    def refuse(state, device=-1):
        raise RuntimeError("wrong state size")

    monkeypatch.setattr(torch.cuda, "set_rng_state", refuse)
    _seed_streams(999)
    before = _snapshot()

    with capture_logs() as logs:
        _load(folder)

    assert _snapshot() == before
    assert len(_events(logs, "rng_state_unreadable_on_resume")) == 1


def test_saving_does_not_start_cuda_in_a_process_that_never_used_it(tmp_path, monkeypatch):
    _stub_cuda(monkeypatch, count=2, saved=_cuda_states(2), initialized=False)

    folder = _save(tmp_path)

    saved = torch.load(folder / "rng_state.pt", map_location="cpu", weights_only=True)
    assert saved["cuda"] == []


# ── The folder ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("is_final", [False, True])
def test_the_file_is_in_the_manifest_and_the_folder_is_resumable(tmp_path, is_final):
    folder = _save(tmp_path, is_final=is_final)

    manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))

    assert manifest["files"]["rng_state.pt"] == (folder / "rng_state.pt").stat().st_size
    assert checkpoint_resumable(folder) == (True, "")


def test_a_truncated_file_is_refused_by_name(tmp_path):
    folder = _save(tmp_path)
    path = folder / "rng_state.pt"
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.seek(size - 1024)
        f.write(b"\x00" * 1024)

    with pytest.raises(CheckpointUnreadable, match="rng_state.pt"):
        _load(folder)


@pytest.mark.parametrize("broken", ["numpy", "python", "torch", "cuda"])
def test_a_malformed_file_warns_and_restores_nothing(tmp_path, broken):
    folder = _save(tmp_path)
    path = folder / "rng_state.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved[broken] = {"cuda": [torch.zeros(3)]}.get(broken, "junk")
    torch.save(saved, path)
    manifest_path = folder / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["rng_state.pt"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _seed_streams(999)
    before = _snapshot()

    with capture_logs() as logs:
        _load(folder)

    assert _snapshot() == before
    assert len(_events(logs, "rng_state_unreadable_on_resume")) == 1


def test_a_failed_capture_fails_nothing_else_and_is_logged(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("no state")

    monkeypatch.setattr(torch, "get_rng_state", boom)
    with capture_logs() as logs:
        folder = _save(tmp_path)
    monkeypatch.undo()

    assert not (folder / "rng_state.pt").exists()
    assert checkpoint_resumable(folder) == (True, "")
    assert len(_events(logs, "failed_to_save_rng_state")) == 1


# ── Saving moves nothing ─────────────────────────────────────────────────


@pytest.mark.parametrize("is_final", [False, True])
def test_saving_moves_no_stream(tmp_path, is_final):
    model = nn.Linear(2, 2)
    _seed_streams(5)
    before = _snapshot()

    CheckpointManager(str(tmp_path)).save_checkpoint(
        10, {"model": model}, config={"lora_name": "t", "seed": 42}, is_final=is_final
    )

    assert _snapshot() == before


# ── A changed seed wins ──────────────────────────────────────────────────


@pytest.mark.parametrize("current", [7, "7", "", None])
def test_a_changed_seed_leaves_the_streams_alone_and_warns(tmp_path, current):
    folder = _save(tmp_path, seed=42)
    _seed_streams(999)
    before = _snapshot()

    with capture_logs() as logs:
        _load(folder, current_config={"lora_name": "t", "seed": current})

    assert _snapshot() == before
    warned = _events(logs, "rng_state_seed_changed_on_resume")
    assert len(warned) == 1
    assert warned[0]["saved_seed"] == 42


def test_a_seed_set_on_an_unseeded_job_counts_as_changed(tmp_path):
    folder = _save(tmp_path, seed=None)
    _seed_streams(999)
    before = _snapshot()

    with capture_logs() as logs:
        _load(folder, current_config={"lora_name": "t", "seed": 7})

    assert _snapshot() == before
    assert len(_events(logs, "rng_state_seed_changed_on_resume")) == 1


@pytest.mark.parametrize(
    ("saved_seed", "current_config"),
    [
        (42, {"lora_name": "t", "seed": 42}),
        ("42", {"lora_name": "t", "seed": 42}),
        (42, {"lora_name": "t", "seed": "42"}),
        (42, {"lora_name": "t"}),  # no seed sent: the checkpoint's stays, as in apply_overrides
        (None, {"lora_name": "t"}),
        (None, {"lora_name": "t", "seed": ""}),
        ("not a number", {"lora_name": "t", "seed": None}),  # both unseeded to _apply_run_seed
    ],
)
def test_an_unchanged_seed_restores(tmp_path, saved_seed, current_config):
    _seed_streams(5)
    folder = _save(tmp_path, seed=saved_seed)
    expected = _draws()

    _seed_streams(999)
    with capture_logs() as logs:
        _load(folder, current_config=current_config)

    assert _draws() == expected
    assert _events(logs, "rng_state_seed_changed_on_resume") == []


# ── Checkpoints written before this ──────────────────────────────────────


def _todays_layout(folder: Path, *, with_manifest: bool) -> Path:
    """A folder as ``main`` writes it, built by hand: no new file can sneak in."""
    folder.mkdir(parents=True)
    torch.save(nn.Linear(2, 2).state_dict(), folder / "model.pt")
    (folder / "training_state.json").write_text(
        json.dumps({"global_step": 10, "elapsed_time": 1.0, "config": {"lora_name": "t", "seed": 42}}),
        encoding="utf-8",
    )
    if with_manifest:
        files = {name: (folder / name).stat().st_size for name in ("model.pt", "training_state.json")}
        (folder / "checkpoint_manifest.json").write_text(
            json.dumps({"format": 2, "files": files}), encoding="utf-8"
        )
    return folder


@pytest.mark.parametrize("with_manifest", [True, False])
def test_a_checkpoint_from_before_resumes(tmp_path, with_manifest):
    folder = _todays_layout(tmp_path / "checkpoint-000010", with_manifest=with_manifest)
    assert checkpoint_resumable(folder) == (True, "")
    model = nn.Linear(2, 2)

    state = CheckpointManager(str(tmp_path)).load_checkpoint(
        str(folder), components={"model": model}, current_config={"lora_name": "t", "seed": 42}
    )

    assert state.global_step == 10
    assert state.components_loaded == ["model"]


def test_a_checkpoint_without_the_file_warns_once_and_leaves_the_streams_alone(tmp_path):
    folder = _todays_layout(tmp_path / "checkpoint-000010", with_manifest=True)
    _seed_streams(999)
    before = _snapshot()

    with capture_logs() as logs:
        _load(folder, current_config={"lora_name": "t", "seed": 42})

    assert _snapshot() == before
    assert len(_events(logs, "rng_state_missing_on_resume")) == 1
