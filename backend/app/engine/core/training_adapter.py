"""The training adapter: a frozen LoRA live in every training step, off for every sample.

A distilled model trained with plain flow-matching MSE drifts from what it was distilled to
(the step and guidance distillation decay; outputs darken and flatten). One remedy is a
second LoRA, trained once on the model's own outputs, that absorbs that drift: it is live in
every training forward pass, so the user's LoRA only has to learn the concept, and it is off
for every preview and validation sample, so those show the user's LoRA on the plain model,
which is what the file the user takes away will do.

A definition names its adapter in ``distillation.training_adapter`` (repo, revision, file,
sha256); a job uses it when its distillation handling is ``adapter`` or ``both``
(``core/definitions.resolve_distillation``). The shared mechanism, any family can use:

* :func:`fetch_verified` downloads the pinned file once through the shared Hugging Face path
  and checks its sha256 (``setup()``, before any weight loads);
* :func:`family_key_map` finds the family's own native -> training key map, a
  ``training_adapter_from_native`` function in the family package's ``lora_keys`` module;
* :func:`attach` adds the adapter as a second named PEFT adapter, frozen, both adapters
  active (it works on a quantized base the same way the user's LoRA does: PEFT dispatches
  both the same way);
* :func:`adapter_off` is the one context manager every preview and validation sample runs
  under; it turns the adapter off and back on in its ``finally``, and leaves every
  ``requires_grad`` as it found it;
* the adapter is never merged, never saved (the savers read only :data:`USER_ADAPTER`; a
  resumable checkpoint leaves it out, :func:`resumable_save_kwargs`) and
  never trainable, so the optimizer, EMA and the adaptive controller never see it.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)

#: PEFT's name for the user's LoRA (``get_peft_model``'s default adapter).
USER_ADAPTER = "default"
#: The frozen training adapter's PEFT name.
TRAINING_ADAPTER = "training_adapter"

#: The function a family's ``lora_keys`` module provides: the adapter file's tensors ->
#: ``{training-model module path: (lora_A, lora_B)}``, the scale folded into ``lora_B``.
KEY_MAP_NAME = "training_adapter_from_native"

KeyMap = Callable[[Mapping[str, torch.Tensor]], dict[str, tuple[torch.Tensor, torch.Tensor]]]


# ── The pinned file ──────────────────────────────────────────────────────────


def hub_uri(spec: Any) -> str:
    """The shared resolver's URI for the pinned file: ``huggingface:<repo>@<revision>:<file>``."""
    return f"huggingface:{spec.repo}@{spec.revision}:{spec.file}"


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_verified(spec: Any, *, local_files_only: bool = False) -> str:
    """Download the definition's training adapter (once; the HF cache keeps it) and return its
    local path. Raises ``FileNotFoundError`` when it cannot be had and ``ValueError`` when its
    sha256 is not the pinned one, both naming the file."""
    from app.engine.utils.model_utils import ModelPathResolver

    uri = hub_uri(spec)
    path = ModelPathResolver.resolve(uri, local_files_only=local_files_only)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"training adapter {uri}: no file at {path!r}")
    actual = sha256_of(path)
    if actual != spec.sha256:
        raise ValueError(
            f"training adapter {uri}: sha256 {actual} does not match the pinned {spec.sha256}; "
            "the file was not used"
        )
    logger.info("training_adapter_verified", uri=uri, path=path, sha256=actual)
    return path


def family_key_map(definition: Any) -> KeyMap:
    """The family's native -> training key map (``<family package>.lora_keys.
    training_adapter_from_native``). Raises ``ValueError`` naming the family when it has none."""
    from app.engine.models.registry import registry

    try:
        family_cls = registry.get_family_class(definition.family)
    except ValueError:  # a process that has not discovered the families yet
        registry.discover_families()
        family_cls = registry.get_family_class(definition.family)
    package = family_cls.__module__.rsplit(".", 1)[0]
    try:
        module = importlib.import_module(f"{package}.lora_keys")
    except ImportError:
        module = None
    key_map = getattr(module, KEY_MAP_NAME, None)
    if not callable(key_map):
        raise ValueError(
            f"{definition.family}: the definition names a training adapter, but the family has no "
            f"{package}.lora_keys.{KEY_MAP_NAME} to convert it to the training model's modules"
        )
    return key_map


def load_pairs(path: str, key_map: KeyMap) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The adapter file through the family's key map, in fp32."""
    from safetensors.torch import load_file

    pairs = key_map(load_file(path))
    if not pairs:
        raise ValueError(f"training adapter {path}: the family's key map found no LoRA pair")
    return {m: (a.float(), b.float()) for m, (a, b) in pairs.items()}


# ── The PEFT side ────────────────────────────────────────────────────────────


def _tuner_layers(model: torch.nn.Module) -> list[Any]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [m for m in model.modules() if isinstance(m, BaseTunerLayer)]


def has_training_adapter(model: torch.nn.Module | None) -> bool:
    if model is None:
        return False
    config = getattr(model, "peft_config", None)
    return isinstance(config, dict) and TRAINING_ADAPTER in config


def resumable_save_kwargs(model: torch.nn.Module) -> dict[str, Any]:
    """The ``save_pretrained`` arguments for a resumable checkpoint of ``model``: with the
    training adapter attached, every adapter but it; otherwise none, so the call is as before.

    The adapter is never part of a checkpoint: a resumed run attaches it again from its pinned,
    verified file, and under block swap its frozen weights hold no storage between runs, so
    reading them for a save would read freed memory."""
    if not has_training_adapter(model):
        return {}
    return {"selected_adapters": [name for name in model.peft_config if name != TRAINING_ADAPTER]}


def adapter_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Every parameter of the training adapter, by name."""
    marker = f".{TRAINING_ADAPTER}."
    return [(n, p) for n, p in model.named_parameters() if marker in n]


def freeze(model: torch.nn.Module) -> int:
    """Set every training-adapter parameter to ``requires_grad=False``; returns how many."""
    params = adapter_parameters(model)
    for _, p in params:
        p.requires_grad_(False)
    return len(params)


def _set_active(model: torch.nn.Module, names: list[str]) -> None:
    """Make ``names`` the active adapters of every LoRA layer, leaving ``requires_grad`` as it
    was (PEFT's ``set_adapter`` also flips it, which would undo targeted training)."""
    grads = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for layer in _tuner_layers(model):
            layer.set_adapter(names)
    finally:
        for p, flag in grads:
            p.requires_grad_(flag)


def attach(model: torch.nn.Module, pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Add the adapter to the PEFT-wrapped ``model`` as :data:`TRAINING_ADAPTER`, load its
    weights, freeze it and make both adapters active. Every pair must land on a module of the
    model with matching shapes (0 unmatched, 0 unused). Returns the number of modules."""
    from peft import LoraConfig

    if not hasattr(model, "add_adapter") or not isinstance(getattr(model, "peft_config", None), dict):
        raise ValueError("training adapter: the model is not a PEFT model; attach it after the user's LoRA")
    if has_training_adapter(model):
        raise ValueError("training adapter: already attached")
    ranks = {a.shape[0] for a, _ in pairs.values()}
    if len(ranks) != 1:
        raise ValueError(f"training adapter: one rank per adapter is supported, got {sorted(ranks)}")
    rank = ranks.pop()
    inner = model.base_model.model
    missing = []
    for module in pairs:
        try:
            target = inner.get_submodule(module)
        except AttributeError:
            target = None
        if not isinstance(target, torch.nn.Module):
            missing.append(module)
    if missing:
        raise ValueError(f"training adapter: {len(missing)} modules are not in the model: {sorted(missing)[:4]}")

    # Scale 1: the file's scale is folded into lora_B (alpha = r).
    config = LoraConfig(r=rank, lora_alpha=rank, target_modules=sorted(pairs), lora_dropout=0.0, bias="none")
    model.add_adapter(TRAINING_ADAPTER, config)

    loaded = set()
    with torch.no_grad():
        for name, layer in inner.named_modules():
            lora_a = getattr(layer, "lora_A", None)
            if lora_a is None or TRAINING_ADAPTER not in lora_a:
                continue
            if name not in pairs:
                raise ValueError(f"training adapter: PEFT adapted {name!r}, which the file does not name")
            a, b = pairs[name]
            weight_a = layer.lora_A[TRAINING_ADAPTER].weight
            weight_b = layer.lora_B[TRAINING_ADAPTER].weight
            if tuple(weight_a.shape) != tuple(a.shape) or tuple(weight_b.shape) != tuple(b.shape):
                raise ValueError(
                    f"training adapter: {name} has A{tuple(a.shape)} B{tuple(b.shape)}, the model's layer "
                    f"A{tuple(weight_a.shape)} B{tuple(weight_b.shape)}"
                )
            weight_a.copy_(a.to(weight_a.dtype))
            weight_b.copy_(b.to(weight_b.dtype))
            loaded.add(name)
    unused = sorted(set(pairs) - loaded)
    if unused:
        raise ValueError(f"training adapter: {len(unused)} pairs were not loaded: {unused[:4]}")
    freeze(model)
    _set_active(model, [USER_ADAPTER, TRAINING_ADAPTER])
    logger.info("training_adapter_attached", modules=len(loaded), rank=rank)
    return len(loaded)


@contextmanager
def adapter_off(model: torch.nn.Module | None) -> Iterator[None]:
    """Run the block with the training adapter off (only the user's LoRA active), and turn it
    back on afterwards, also when the block raises. A model without the adapter is untouched."""
    if not has_training_adapter(model):
        yield
        return
    _set_active(model, [USER_ADAPTER])
    try:
        yield
    finally:
        _set_active(model, [USER_ADAPTER, TRAINING_ADAPTER])
