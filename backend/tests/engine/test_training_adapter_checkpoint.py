"""The training adapter in a resumable checkpoint (``CheckpointManager``).

The frozen training adapter (``core/training_adapter.py``) is rebuilt on every start from its
pinned, verified file, so a resumable checkpoint holds only the user's adapter:

* the real ``save_checkpoint`` writes the user's adapter alone (no ``training_adapter``
  subfolder, no key of it), and the manifest lists every adapter file the folder holds;
* it never reads the training adapter's tensors: block swap releases a frozen weight's storage
  between runs (``untyped_storage().resize_(0)``), and the save still succeeds;
* a resume (the pipeline's own attach from the pinned file, then the real ``load_checkpoint``)
  restores the user's LoRA, with the training adapter attached once, frozen, at the pinned
  file's values, and only the user's parameters trainable;
* a checkpoint that already holds a ``training_adapter`` subfolder resumes the same way: that
  copy is never loaded over the verified one;
* a model with the user's adapter alone saves exactly the files ``save_pretrained`` writes.

On CPU, with a tiny two-Linear model; nothing is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import structlog
import torch
import torch.nn as nn
from safetensors import safe_open

from app.engine.components.checkpoints import MANIFEST_NAME, CheckpointManager, checkpoint_resumable
from app.engine.core import training_adapter as ta
from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin

R = 2
CONFIG = {"lora_name": "t"}


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 6)
        self.b = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(x)))


def _user_lora(seed: int = 1) -> nn.Module:
    """The user's LoRA on ``a`` only, with non-zero weights (PEFT zero-inits B)."""
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_Tiny(), LoraConfig(r=R, lora_alpha=2 * R, target_modules=["a"]))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if ".lora_" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return model


def _pinned_file(tmp_path: Path) -> Path:
    """The adapter file, on ``a`` (beside the user's LoRA) and ``b`` (where the user has none)."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(2)
    tensors = {
        "a.A": torch.randn(R, 4, generator=g),
        "a.B": torch.randn(6, R, generator=g),
        "b.A": torch.randn(R, 6, generator=g),
        "b.B": torch.randn(3, R, generator=g),
    }
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path))
    return path


def _key_map(tensors):
    return {m: (tensors[f"{m}.A"], tensors[f"{m}.B"]) for m in ("a", "b")}


def _attach_like_the_pipeline(model: nn.Module, pinned: Path) -> None:
    """The pipeline's own attach step, from the file ``setup()`` verified."""
    owner = SimpleNamespace(
        _training_adapter_path=str(pinned),
        _training_adapter_key_map=_key_map,
        logger=structlog.get_logger("test"),
    )
    PipelineOptimizationMixin._attach_training_adapter(owner, model)


def _with_adapter(tmp_path: Path, seed: int = 1) -> nn.Module:
    model = _user_lora(seed)
    _attach_like_the_pipeline(model, _pinned_file(tmp_path))
    return model


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # As _configure_optimization collects them: the primary model's trainable params.
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)


def _save(run: Path, model: nn.Module, step: int = 5, *, before_save=None) -> Path:
    opt = _optimizer(model)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    if before_save is not None:
        before_save(model)
    CheckpointManager(str(run)).save_checkpoint(step, {"unet": model}, optimizer=opt, config=CONFIG)
    return run / f"checkpoint-{step:06d}"


def _keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt") as sf:
        return set(sf.keys())


def _lora(model: nn.Module, adapter: str) -> dict[str, torch.Tensor]:
    marker = f".{adapter}."
    return {n: p.detach().clone() for n, p in model.named_parameters() if marker in n}


def _files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def _release_like_block_swap(model: nn.Module) -> None:
    """Free the training adapter's storage in place, as BlockSwappingManager._free does."""
    for _, p in ta.adapter_parameters(model):
        p.data.untyped_storage().resize_(0)


# ── The save ─────────────────────────────────────────────────────────────────


def test_a_resumable_checkpoint_holds_only_the_users_adapter(tmp_path):
    model = _with_adapter(tmp_path)
    folder = _save(tmp_path / "run", model)

    unet = folder / "unet"
    assert not (unet / ta.TRAINING_ADAPTER).exists(), (
        f"the checkpoint holds the training adapter: {sorted(_files(unet))}"
    )
    assert not any(ta.TRAINING_ADAPTER in f for f in _files(folder)), sorted(_files(folder))

    twin = tmp_path / "twin"
    _user_lora().save_pretrained(str(twin))
    keys = _keys(unet / "adapter_model.safetensors")
    assert keys == _keys(twin / "adapter_model.safetensors")
    assert not any(ta.TRAINING_ADAPTER in k for k in keys)
    config = json.loads((unet / "adapter_config.json").read_text())
    assert config["target_modules"] == ["a"]

    ok, reason = checkpoint_resumable(folder)
    assert ok, reason
    listed = json.loads((folder / MANIFEST_NAME).read_text())["files"]
    assert {k for k in listed if k.startswith("unet/")} == {f"unet/{f}" for f in _files(unet)}


def test_the_save_never_reads_the_training_adapters_released_storage(tmp_path):
    # A save that reads the released tensors reads freed memory: on CPU that is an access
    # violation, which ends the test process (the worker's crash is the failure).
    from safetensors.torch import load_file

    model = _with_adapter(tmp_path)

    # Released after the step (block swap frees a block once its backward is done).
    folder = _save(tmp_path / "run", model, before_save=_release_like_block_swap)

    ok, reason = checkpoint_resumable(folder)
    assert ok, reason
    user = _lora(model, ta.USER_ADAPTER)
    saved = load_file(str(folder / "unet" / "adapter_model.safetensors"))
    assert len(saved) == len(user)
    for name, value in user.items():
        assert torch.equal(saved[name.replace(f".{ta.USER_ADAPTER}.", ".")], value), name
    assert all(p.untyped_storage().nbytes() == 0 for _, p in ta.adapter_parameters(model)), (
        "the save refilled the released storage"
    )


# ── The resume ───────────────────────────────────────────────────────────────


def _resume(tmp_path: Path, folder: Path) -> tuple[nn.Module, torch.optim.Optimizer]:
    """A fresh run: the user's LoRA, the training adapter from its pinned file, the optimizer,
    then the real load (the order of prepare_for_training)."""
    fresh = _user_lora(seed=99)
    _attach_like_the_pipeline(fresh, _pinned_file(tmp_path))
    opt = _optimizer(fresh)
    CheckpointManager(str(folder.parent)).load_checkpoint(
        str(folder), peft_components={"unet": fresh}, optimizer=opt, current_config=CONFIG
    )
    return fresh, opt


def _assert_resumed(fresh: nn.Module, opt: torch.optim.Optimizer, saved_user: dict, pinned: dict) -> None:
    assert set(fresh.peft_config) == {ta.USER_ADAPTER, ta.TRAINING_ADAPTER}
    for layer in ta._tuner_layers(fresh):
        assert set(layer.lora_A) <= {ta.USER_ADAPTER, ta.TRAINING_ADAPTER}
        assert set(layer.active_adapters) == {ta.USER_ADAPTER, ta.TRAINING_ADAPTER}
    resumed_user = _lora(fresh, ta.USER_ADAPTER)
    assert resumed_user.keys() == saved_user.keys()
    for name, value in saved_user.items():
        assert torch.equal(resumed_user[name], value), name
    resumed_adapter = _lora(fresh, ta.TRAINING_ADAPTER)
    assert resumed_adapter.keys() == pinned.keys()
    for name, value in pinned.items():
        assert torch.equal(resumed_adapter[name], value), f"{name} is not the pinned file's value"
    trainable = {n for n, p in fresh.named_parameters() if p.requires_grad}
    assert trainable == set(saved_user), sorted(trainable)
    in_optimizer = {id(p) for g in opt.param_groups for p in g["params"]}
    assert in_optimizer == {id(p) for n, p in fresh.named_parameters() if n in saved_user}


def test_a_resume_rebuilds_the_training_adapter_from_its_pinned_file(tmp_path):
    model = _with_adapter(tmp_path)
    pinned = _lora(model, ta.TRAINING_ADAPTER)
    folder = _save(tmp_path / "run", model)
    saved_user = _lora(model, ta.USER_ADAPTER)  # after the step the save follows

    fresh, opt = _resume(tmp_path, folder)

    _assert_resumed(fresh, opt, saved_user, pinned)


def test_an_older_checkpoint_with_the_adapter_inside_never_loads_it(tmp_path):
    model = _with_adapter(tmp_path)
    pinned = _lora(model, ta.TRAINING_ADAPTER)
    folder = _save(tmp_path / "run", model)
    saved_user = _lora(model, ta.USER_ADAPTER)

    # The layout an unselected save_pretrained writes: the user's adapter at the top, the
    # training adapter in its own subfolder, here with values that are not the pinned file's.
    old = tmp_path / "old"
    with torch.no_grad():
        for _, p in ta.adapter_parameters(model):
            p.fill_(7.0)
    model.save_pretrained(str(old))
    assert (old / ta.TRAINING_ADAPTER / "adapter_model.safetensors").is_file()
    (folder / "unet" / ta.TRAINING_ADAPTER).mkdir(exist_ok=True)
    for f in (old / ta.TRAINING_ADAPTER).iterdir():
        (folder / "unet" / ta.TRAINING_ADAPTER / f.name).write_bytes(f.read_bytes())
    ok, reason = checkpoint_resumable(folder)
    assert ok, reason

    fresh, opt = _resume(tmp_path, folder)

    _assert_resumed(fresh, opt, saved_user, pinned)


# ── A family without the training adapter ────────────────────────────────────


def test_a_model_with_the_users_adapter_alone_saves_as_save_pretrained_does(tmp_path):
    model = _user_lora()
    folder = _save(tmp_path / "run", model)
    reference = tmp_path / "reference"
    model.save_pretrained(str(reference))

    unet = folder / "unet"
    assert _files(unet) == _files(reference)
    for f in _files(reference):
        assert (unet / f).read_bytes() == (reference / f).read_bytes(), f
