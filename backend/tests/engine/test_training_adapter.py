"""The training adapter, the shared mechanism (``core/training_adapter.py``).

A frozen LoRA a distilled definition names (``distillation.training_adapter``): pinned by
revision and sha256, attached beside the user's LoRA as a second PEFT adapter, live in every
training forward pass, and off for every preview and validation sample. On CPU, with a tiny
two-Linear model, nothing mocked but the download:

* the pinned file is resolved through the shared HF path and its sha256 checked;
* it attaches with 0 unmatched and 0 unused, at scale 1 (the file's scale is in ``lora_B``),
  frozen, and both adapters active;
* the one context manager turns it off and back on, also when the block raises, leaving
  every ``requires_grad`` as it was;
* a preview through the real ``GenericSamplingPipeline.generate_samples`` sees the model
  without it (the output equals the model's with no adapter attached), and it is on again
  afterwards;
* the generic saver writes only the user's adapter: the same key set as without it.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.engine.core import training_adapter as ta

R = 2


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 6)
        self.b = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(x)))


def _user_lora() -> nn.Module:
    """The user's LoRA on ``a`` only, with non-zero weights (PEFT zero-inits B)."""
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_Tiny(), LoraConfig(r=R, lora_alpha=2 * R, target_modules=["a"]))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if ".lora_" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return model


def _pairs(seed: int = 2) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """An adapter on ``a`` (beside the user's LoRA) and ``b`` (where the user has none)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "a": (torch.randn(R, 4, generator=g), torch.randn(6, R, generator=g)),
        "b": (torch.randn(R, 6, generator=g), torch.randn(3, R, generator=g)),
    }


def _x() -> torch.Tensor:
    return torch.randn(5, 4, generator=torch.Generator().manual_seed(3))


def _with_adapter() -> tuple[nn.Module, nn.Module]:
    """``(model with the user's LoRA and the adapter, its twin with the user's LoRA only)``."""
    model = _user_lora()
    plain = copy.deepcopy(model)
    ta.attach(model, _pairs())
    return model, plain


def _active(model: nn.Module) -> set[tuple[str, ...]]:
    return {tuple(layer.active_adapters) for layer in ta._tuner_layers(model)}


# ── The pinned file ──────────────────────────────────────────────────────────


@pytest.fixture
def adapter_file(tmp_path: Path) -> tuple[Path, str]:
    from safetensors.torch import save_file

    path = tmp_path / "adapter.safetensors"
    save_file({"x.lora_A.weight": torch.ones(2, 2)}, str(path))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(sha256: str) -> SimpleNamespace:
    return SimpleNamespace(repo="someone/adapter", revision="e4e119ca" * 5, file="adapter.safetensors", sha256=sha256)


def _resolve_to(monkeypatch, path: Path | None, seen: list[str]) -> None:
    from app.engine.utils.model_utils import ModelPathResolver

    def resolve(uri, base_dir=None, *, local_files_only=False):
        seen.append(uri)
        return str(path) if path is not None else None

    monkeypatch.setattr(ModelPathResolver, "resolve", staticmethod(resolve))


def test_the_pinned_file_is_resolved_through_the_shared_path_and_checked(monkeypatch, adapter_file):
    path, digest = adapter_file
    seen: list[str] = []
    _resolve_to(monkeypatch, path, seen)
    assert ta.fetch_verified(_spec(digest)) == str(path)
    assert seen == [f"huggingface:someone/adapter@{'e4e119ca' * 5}:adapter.safetensors"]


def test_a_wrong_sha256_is_refused(monkeypatch, adapter_file):
    path, _ = adapter_file
    _resolve_to(monkeypatch, path, [])
    with pytest.raises(ValueError, match="does not match the pinned"):
        ta.fetch_verified(_spec("0" * 64))


def test_a_missing_file_is_refused(monkeypatch, tmp_path):
    _resolve_to(monkeypatch, tmp_path / "gone.safetensors", [])
    with pytest.raises(FileNotFoundError, match="training adapter"):
        ta.fetch_verified(_spec("0" * 64))


def test_a_family_without_a_key_map_is_refused():
    with pytest.raises(ValueError, match="training_adapter_from_native"):
        ta.family_key_map(SimpleNamespace(family="sdxl"))


# ── Attaching ────────────────────────────────────────────────────────────────


def test_the_adapter_is_active_at_scale_one_beside_the_users_lora():
    model, plain = _with_adapter()
    x = _x()
    with torch.no_grad():
        assert not torch.allclose(model(x), plain(x)), "the adapter changed nothing"
        # By hand: W_a + s*B_u@A_u + B_t@A_t, W_b + B_t@A_t (scale 1 for the adapter).
        reference = _Tiny()
        user = plain.base_model.model.a
        pairs = _pairs()
        reference.a.weight += float(user.scaling["default"]) * (
            user.lora_B["default"].weight @ user.lora_A["default"].weight
        )
        reference.a.weight += pairs["a"][1] @ pairs["a"][0]
        reference.b.weight += pairs["b"][1] @ pairs["b"][0]
        torch.testing.assert_close(model(x), reference(x), atol=1e-6, rtol=0)
    assert _active(model) == {("default", "training_adapter")}


def test_the_adapter_is_frozen_and_the_users_lora_is_not():
    model, _ = _with_adapter()
    adapter = ta.adapter_parameters(model)
    assert len(adapter) == 4  # A and B on two modules
    assert not any(p.requires_grad for _, p in adapter)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".default." in n for n in trainable)


def test_a_backward_pass_reaches_only_the_users_lora():
    model, _ = _with_adapter()
    model(_x()).pow(2).mean().backward()
    for name, p in model.named_parameters():
        if ".default." in name:
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
        else:
            assert p.grad is None, f"{name} got a gradient"


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ({"c": (torch.zeros(R, 4), torch.zeros(6, R))}, "not in the model"),
        ({"a": (torch.zeros(R, 5), torch.zeros(6, R))}, "the model's layer"),
        ({"a": (torch.zeros(R, 4), torch.zeros(6, R)), "b": (torch.zeros(3, 6), torch.zeros(3, 3))}, "one rank"),
    ],
)
def test_a_pair_that_does_not_fit_the_model_is_refused(pairs, message):
    with pytest.raises(ValueError, match=message):
        ta.attach(_user_lora(), pairs)


def test_the_adapter_needs_the_users_peft_model_and_attaches_once():
    with pytest.raises(ValueError, match="not a PEFT model"):
        ta.attach(_Tiny(), _pairs())
    model, _ = _with_adapter()
    with pytest.raises(ValueError, match="already attached"):
        ta.attach(model, _pairs())


# ── Off for samples ──────────────────────────────────────────────────────────


def test_off_means_the_model_without_the_adapter_and_on_again_after():
    model, plain = _with_adapter()
    x = _x()
    with torch.no_grad():
        on = model(x)
        with ta.adapter_off(model):
            assert torch.equal(model(x), plain(x))
            assert _active(model) == {("default",)}
        assert torch.equal(model(x), on)
    assert _active(model) == {("default", "training_adapter")}


def test_off_comes_back_on_when_the_block_raises():
    model, _ = _with_adapter()
    with pytest.raises(RuntimeError, match="sampling failed"):
        with ta.adapter_off(model):
            raise RuntimeError("sampling failed")
    assert _active(model) == {("default", "training_adapter")}


def test_off_and_on_leave_every_requires_grad_as_it_was():
    model, _ = _with_adapter()
    # Targeted training froze the user's LoRA on ``a``'s A matrix.
    frozen = model.base_model.model.a.lora_A["default"].weight
    frozen.requires_grad_(False)
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    with ta.adapter_off(model):
        assert {n: p.requires_grad for n, p in model.named_parameters()} == before
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before


def test_a_model_without_the_adapter_is_untouched():
    model = _user_lora()
    x = _x()
    with torch.no_grad():
        before = model(x)
        with ta.adapter_off(model):
            assert torch.equal(model(x), before)
    with ta.adapter_off(None):
        pass


class _Sampler:
    """The real ``generate_samples`` over a stub family: records the model's output per prompt."""

    @staticmethod
    def make(model: nn.Module, tmp_path: Path, fail: bool = False):
        from app.engine.core.sampling import GenericSamplingPipeline

        class Sampler(GenericSamplingPipeline):
            def encode_prompt(self, prompt):  # pragma: no cover - _sample_single is stubbed
                return None

            def denoise(self, *args, **kwargs):  # pragma: no cover
                return None

            def decode_latents(self, *args, **kwargs):  # pragma: no cover
                return None

            def _create_initial_noise(self, width, height, generator):  # pragma: no cover
                return torch.zeros(1)

            def _vram_headroom_ok(self):
                return True

            def _resolve_output_dir(self):
                return tmp_path

            def _sample_single(self, cfg, step):
                if fail:
                    raise RuntimeError("sampling failed")
                with torch.no_grad():
                    self.seen.append(model(_x()))
                return object()

            def _persist_artifact(self, artifact, output_dir, i, displayed_step, final):
                return tmp_path / f"{i}.png"

            def _broadcast_sample(self, *args, **kwargs):
                return None

        pipeline = SimpleNamespace(
            config={"sample_prompts": [{"prompt": "a"}, {"prompt": "b"}]},
            device=torch.device("cpu"),
            _get_primary_model=lambda: model,
            optimizer=None,
            ema_handler=None,
        )
        sampler = Sampler(pipeline)
        sampler.seen: list[Any] = []
        return sampler


def test_a_preview_runs_without_the_adapter(tmp_path):
    model, plain = _with_adapter()
    sampler = _Sampler.make(model, tmp_path)
    assert len(sampler.generate_samples(step=9)) == 2
    with torch.no_grad():
        expected = plain(_x())
    assert len(sampler.seen) == 2
    for output in sampler.seen:
        assert torch.equal(output, expected), "a preview ran with the training adapter on"
    assert _active(model) == {("default", "training_adapter")}


def test_the_adapter_is_on_again_after_a_preview_raises(tmp_path):
    model, _ = _with_adapter()
    sampler = _Sampler.make(model, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="sampling failed"):
        sampler.generate_samples(step=9)
    assert _active(model) == {("default", "training_adapter")}


# ── Never saved ──────────────────────────────────────────────────────────────


def test_the_generic_saver_writes_only_the_users_adapter(tmp_path):
    from safetensors import safe_open

    from app.engine.core.pipeline.saver_base import GenericLoRASaver

    class Saver(GenericLoRASaver):
        architecture_name = "tiny"

    def saved(model: nn.Module, name: str) -> tuple[set[str], dict[str, str]]:
        path = tmp_path / name
        Saver().save({"unet": model, "config": {"save_precision": "fp32"}}, path)
        with safe_open(str(path), framework="pt") as fh:
            return set(fh.keys()), dict(fh.metadata() or {})

    model, plain = _with_adapter()
    with_keys, with_meta = saved(model, "with.safetensors")
    plain_keys, plain_meta = saved(plain, "plain.safetensors")
    assert with_keys == plain_keys
    assert not any("training_adapter" in k for k in with_keys)
    assert with_meta["ss_network_dim"] == plain_meta["ss_network_dim"] == str(R)
    assert with_meta["ss_network_alpha"] == plain_meta["ss_network_alpha"] == str(float(2 * R))
