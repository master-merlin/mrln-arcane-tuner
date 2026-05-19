"""Audit R-TENSOR-10 — driver precision spec follows loaded model dtype.

Symmetrical to the sampler dtype fix in 287b840: when a driver builds a
``PrecisionSpec`` for the training loop, ``autocast_dtype`` must come
from the actually-loaded model parameters (typically bf16) and not from
the ``mixed_precision`` config string (defaults to ``"fp16"``).  Otherwise
``torch.autocast(dtype=fp16)`` wrapping a bf16 transformer silently
re-promotes every op and accumulates precision drift.

The config string is still respected for fp32-loaded families (e.g. SDXL
with genuine AMP), and continues to gate ``use_amp`` / ``grad_scaler_enabled``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.core.interfaces import IModelDriver
from app.engine.core.layer_manifest import PrecisionSpec


class _NoopDriver(IModelDriver):
    """Smallest possible IModelDriver — only ``get_primary_model`` is exercised."""

    def __init__(self, model: nn.Module | None):
        super().__init__()
        self._model = model

    # --- Phase 1 ---
    def assign_components(self, components: dict[str, Any]) -> None: ...
    def get_components(self) -> dict[str, Any]:
        return {}

    def get_primary_model(self) -> nn.Module:
        return self._model  # type: ignore[return-value]

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {}

    def get_lora_targets(self) -> list[str]:
        return []

    def init_scheduler(self) -> Any:
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    # --- Phase 2 ---
    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        return None

    # --- Phase 4 ---
    def get_te_lora_targets(self) -> list[str]:
        return []

    # --- Phase 5 ---
    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return noisy_input

    # --- Phase 6 ---
    def get_saver(self) -> Any:
        return None


class TestDriverPrecisionSpecFollowsModel:
    """The training-side dtype must come from the loaded model, not the config."""

    def test_bf16_model_overrides_fp16_config(self):
        """Model loaded in bf16 + config 'fp16' -> autocast must be bf16, not fp16."""
        model = nn.Linear(4, 4).to(torch.bfloat16)
        driver = _NoopDriver(model)

        spec = driver.get_precision_spec("fp16")

        assert spec.autocast_dtype == torch.bfloat16, (
            "autocast_dtype must follow the loaded model (bf16), not the config string (fp16). "
            "Otherwise torch.autocast silently re-promotes per-op and drifts precision."
        )
        # AMP-enable / grad-scaler flags still come from the config string.
        assert spec.use_amp is True
        # Pin the current behavior: when the config says fp16 + non-adaptive, we
        # still emit grad_scaler_enabled=True even though we resolved autocast to
        # bf16. GradScaler is a no-op for bf16 autocast (bf16 doesn't underflow),
        # so this is cosmetically misleading but harmless. R-TENSOR-10 only
        # contracts on the dtype; suppressing the scaler when the resolved
        # dtype is bf16 is a separate (deferred) follow-up. If the deferral
        # is ever fixed, this assert will fail loudly and force a re-think.
        assert spec.grad_scaler_enabled is True

    def test_bf16_model_overrides_fp16_config_bf16_branch(self):
        """Model loaded in bf16 + config 'bf16' -> autocast bf16, no scaler (regression check)."""
        model = nn.Linear(4, 4).to(torch.bfloat16)
        driver = _NoopDriver(model)

        spec = driver.get_precision_spec("bf16")

        assert spec.autocast_dtype == torch.bfloat16
        assert spec.use_amp is True
        assert spec.grad_scaler_enabled is False

    def test_fp32_model_honors_fp16_config_for_genuine_amp(self):
        """Model loaded in fp32 (SDXL) + config 'fp16' -> autocast fp16 + scaler on."""
        model = nn.Linear(4, 4).to(torch.float32)
        driver = _NoopDriver(model)

        spec = driver.get_precision_spec("fp16")

        # SDXL-style genuine AMP: fp32 params with fp16 autocast and GradScaler
        # MUST be preserved.  This is the legitimate use of the config string.
        assert spec.autocast_dtype == torch.float16
        assert spec.use_amp is True
        assert spec.grad_scaler_enabled is True

    def test_fp32_model_fp32_config_unchanged(self):
        """Model loaded in fp32 + config 'fp32' -> no autocast, no scaler."""
        model = nn.Linear(4, 4).to(torch.float32)
        driver = _NoopDriver(model)

        spec = driver.get_precision_spec("fp32")

        assert spec.autocast_dtype == torch.float32
        assert spec.use_amp is False
        assert spec.grad_scaler_enabled is False

    def test_adaptive_optimizer_still_disables_scaler(self):
        """Adaptive optimizer flag must still suppress GradScaler even after model-dtype resolution."""
        model = nn.Linear(4, 4).to(torch.float32)
        driver = _NoopDriver(model)

        spec = driver.get_precision_spec("fp16", is_adaptive_optimizer=True)

        assert spec.autocast_dtype == torch.float16
        assert spec.grad_scaler_enabled is False

    def test_from_config_back_compat_without_model_dtype(self):
        """Direct PrecisionSpec.from_config() calls without model_dtype keep legacy behavior."""
        spec = PrecisionSpec.from_config("fp16")
        assert spec.autocast_dtype == torch.float16
        spec = PrecisionSpec.from_config("bf16")
        assert spec.autocast_dtype == torch.bfloat16
