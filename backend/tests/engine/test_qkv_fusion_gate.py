"""QKV-projection fusion is gated per family.

Regression for the LTX-2 training crash:

    torch.cat([to_q.weight, to_k.weight, to_v.weight])
    RuntimeError: Expected size 4096 but got size 2048 for tensor number 1

``_fuse_qkv_projections`` fuses separate Q/K/V linears into a single ``to_qkv``
before PEFT.  This is correct ONLY for families whose LoRA targets reference the
fused module (FLUX.2).  LTX-2's attention has audio<->video cross-modal bridges
whose query dim (4096) differs from their key/value dim (2048); diffusers'
``fuse_projections`` mishandles those (it never sets ``is_cross_attention`` on
``LTX2Attention``) and crashes.  LTX-2 also targets the *unfused* ``to_q/k/v``,
so it must not fuse at all.
"""

import structlog
import torch

from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
from app.engine.models.families.flux2.driver import Flux2Driver
from app.engine.models.families.ltx2.driver import Ltx2Driver


class _FuseModule(torch.nn.Module):
    """Stand-in attention module that records whether it was fused."""

    def __init__(self) -> None:
        super().__init__()
        self._supports_qkv_fusion = True
        self.fused_projections = False
        self.fused_called = False

    def fuse_projections(self) -> None:
        self.fused_called = True
        self.fused_projections = True


class _Model(torch.nn.Module):
    def __init__(self, attn: _FuseModule) -> None:
        super().__init__()
        self.attn = attn


class _Driver:
    def __init__(self, should_fuse: bool) -> None:
        self._should_fuse = should_fuse

    def should_fuse_qkv_projections(self) -> bool:
        return self._should_fuse


class _Harness(PipelineOptimizationMixin):
    """Minimal host exposing just what ``_fuse_qkv_projections`` touches."""

    def __init__(self, driver: _Driver, model: _Model) -> None:
        self.driver = driver
        self._model = model
        self.logger = structlog.get_logger("test")

    def _get_primary_model(self) -> _Model:
        return self._model


def test_fusion_skipped_when_driver_opts_out():
    attn = _FuseModule()
    harness = _Harness(_Driver(should_fuse=False), _Model(attn))

    harness._fuse_qkv_projections()

    assert attn.fused_called is False


def test_fusion_runs_when_driver_opts_in():
    attn = _FuseModule()
    harness = _Harness(_Driver(should_fuse=True), _Model(attn))

    harness._fuse_qkv_projections()

    assert attn.fused_called is True


def test_ltx2_driver_does_not_fuse_qkv():
    drv = object.__new__(Ltx2Driver)
    assert drv.should_fuse_qkv_projections() is False


def test_flux2_driver_fuses_qkv():
    drv = object.__new__(Flux2Driver)
    assert drv.should_fuse_qkv_projections() is True
