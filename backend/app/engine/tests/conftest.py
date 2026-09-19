"""Make ``backend/`` importable when these engine tests are collected.

Mirrors ``backend/tests/conftest.py``'s sys.path insert. Without this, running
or collecting ``backend/app/engine/tests`` raises ``ModuleNotFoundError: No
module named 'app'`` because the package root (``backend/``) is not on the path.
No-op when ``backend/`` is already importable.

Also hosts the shared adaptive-layer-targeting test doubles. They live HERE and
not in a test module because pytest runs with ``--import-mode=importlib`` and
this directory has no ``__init__.py``, so ``app.engine.tests.<module>`` is not
importable — cross-test-module helper imports would fail at collection.
"""

import os
import sys
from typing import Any

import pytest
import torch
import torch.nn as nn

_BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


class FakeLogWriter:
    """In-memory stand-in for ``JobLogWriter``.

    Records every message as a ``(msg_type, data)`` tuple so tests can assert on
    what the UI would actually have seen, including the shorthands — a warning
    that never reached the log channel is a silent failure.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def emit(self, msg_type: str, data: Any) -> None:
        self.events.append((msg_type, data))

    def warning(self, message: str) -> None:
        self.emit("warning", message)

    def log(self, message: str) -> None:
        self.emit("log", message)


class _TinyBlock(nn.Module):
    """One attention-ish block: the two module names PEFT will wrap."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(16, 16)
        self.to_v = nn.Linear(16, 16)

    def forward(self, x):
        return self.to_v(torch.relu(self.to_q(x)))


class _TinyModel(nn.Module):
    """``blocks.<i>.to_{q,v}`` — mirrors real DiT module paths, so the
    controller's name handling is exercised against production naming."""

    def __init__(self, n_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_TinyBlock() for _ in range(n_blocks))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.fixture
def fake_log_writer() -> FakeLogWriter:
    return FakeLogWriter()


@pytest.fixture
def make_peft_tiny():
    """Factory: ``(n_blocks=4) -> peft-wrapped tiny model``.

    Real PEFT wrapping (not a hand-rolled fake) so module/parameter names match
    production exactly: ``base_model.model.blocks.0.to_q.lora_A.default.weight``.
    """

    def _make(n_blocks: int = 4):
        # Imported inside the fixture: this conftest is shared by every engine
        # test, and a peft import failure must not break unrelated collection.
        from peft import LoraConfig, get_peft_model

        return get_peft_model(
            _TinyModel(n_blocks),
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]),
        )

    return _make


@pytest.fixture
def make_adaptive_controller(tmp_path, make_peft_tiny):
    """Factory: ``(**cfg_kwargs) -> (model, controller, writer)``.

    Requests ``tmp_path`` itself (function-scoped, so it is the same directory a
    test's own ``tmp_path`` yields) and writes the run history there.
    """

    def _make(**cfg_kwargs):
        from app.engine.components.adaptive_targeting import AdaptiveTargetingController
        from app.engine.models.adaptive import AdaptiveTargetingConfig

        model = make_peft_tiny(4)
        config = AdaptiveTargetingConfig(
            **{
                "warmup_pct": 0.1,
                "interval_steps": 10,
                "min_active_pct": 0.13,
                # The short test interval forces a short probe window too:
                # AdaptiveTargetingConfig rejects probe_steps >= interval_steps,
                # and the default probe_steps is sized for production intervals.
                "probe_steps": 5,
                **cfg_kwargs,
            }
        )
        writer = FakeLogWriter()
        controller = AdaptiveTargetingController(
            model=model,
            config=config,
            total_steps=100,
            log_writer=writer,
            output_dir=str(tmp_path),
        )
        return model, controller, writer

    return _make


@pytest.fixture
def train_step():
    """Factory: ``(model, hot_modules) -> None`` — one backward + manual update.

    Only modules whose parameter name contains one of ``hot_modules`` receive a
    large update, which is what produces measurable heat; every other module's
    LoRA weights are left byte-identical so its windowed delta is exactly zero.
    Frozen parameters are skipped, so a module the controller froze can never
    accidentally regain heat.
    """

    def _step(model, hot_modules) -> None:
        model(torch.randn(2, 16)).sum().backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_" not in name or not param.requires_grad:
                    continue
                if any(hot in name for hot in hot_modules):
                    param += torch.randn_like(param) * 0.5
        model.zero_grad(set_to_none=True)

    return _step


@pytest.fixture
def force_update():
    """Factory: ``(model, module_substr, scale=0.5) -> None``.

    Moves a module's LoRA weights REGARDLESS of ``requires_grad`` — what a probe
    window's temporary un-freeze looks like from the heat metric's point of
    view. ``train_step`` deliberately skips frozen params, so this is the only
    way a test can put heat on a module the controller has already frozen (and
    therefore the only way to prove monotonicity non-circularly).
    """

    def _force(model, module_substr: str, scale: float = 0.5) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_" in name and module_substr in name:
                    param.data += torch.randn_like(param) * scale

    return _force


# Tiny, divisibility-respecting minimax_h3 config for the INSTALLED diffusers
# class (0.40.0). Real checkpoint values live in the definition YAML; these only
# exercise code paths on CPU. Relationships preserved from the real config:
#   - rope: 2 * 3 * rope_freq_dim (12) <= attention_head_dim (16);
#   - time-embedding funnel freq_dim -> time_embed_hidden_dim -> time_embed_dim
#     shrinks at each stage as 256 -> 5376 -> 2688 does (hidden > final);
#   - in_channels / audio_in_channels / patch_size are the REAL values (they
#     must match the two VAEs' latent widths and the packing.py patchify).
TINY_H3_TRANSFORMER_KWARGS: dict[str, Any] = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 16,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 32,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 16,
    "freq_dim": 16,
    "time_embed_hidden_dim": 16,
    "time_embed_dim": 8,
    "rope_freq_dim": 2,
    "rope_theta": 10000.0,
}


@pytest.fixture
def build_tiny_transformer():
    """Factory: ``() -> diffusers MiniMaxH3Transformer3DModel`` (tiny, fp32, CPU,
    seeded). Plan row 1.4; shared by the convention suite's tests 9 and 10."""

    def _build():
        # Imported inside the fixture: this conftest is shared by every engine
        # test, and a diffusers import failure must not break collection.
        from diffusers import MiniMaxH3Transformer3DModel

        torch.manual_seed(0)
        return MiniMaxH3Transformer3DModel(**TINY_H3_TRANSFORMER_KWARGS)

    return _build


@pytest.fixture
def lora_params():
    """Factory: ``(model, module_substr) -> list[Parameter]`` of LoRA params."""

    def _params(model, module_substr: str):
        return [
            p
            for n, p in model.named_parameters()
            if module_substr in n and ".lora_" in n
        ]

    return _params
