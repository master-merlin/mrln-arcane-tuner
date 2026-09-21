"""minimax_h3 block swap actually ENGAGES (plan row 4.1, GATE-5's ladder).

GATE-5's first ladder rung died in `prepare_for_training` with
`KeyError: 'approx_vram_mb'`: the H3 driver hands the definition's
`block_topology` to the base pipeline verbatim, the YAML groups carried no
`approx_vram_mb` (the `IModelDriver.get_block_topology` contract), and
`_configure_block_swapping` indexes it for its log line — AFTER it has already
swapped the blocks. `supports_block_swap: True` was never exercised.

The seam under test is the real `PipelineOptimizationMixin
._configure_block_swapping` over the real H3 driver topology and the installed
diffusers transformer (tiny, CPU). Nothing in it is stubbed.
"""

from __future__ import annotations

import pytest
import structlog
import torch

from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
from app.engine.tests.test_minimax_h3_definitions import DEF_IDS, _load
from app.engine.tests.test_minimax_h3_driver import _driver
from app.engine.tests.test_minimax_h3_transformer import build_tiny_transformer

needs_pinning = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="BlockSwappingManager pins host memory (needs a CUDA runtime)"
)


class _Host(PipelineOptimizationMixin):
    def __init__(self, model: torch.nn.Module, config: dict) -> None:
        self.driver = _driver()
        self.config = config
        self.device = torch.device("cpu")
        self.logger = structlog.get_logger("test")
        self._model = model

    def _get_primary_model(self) -> torch.nn.Module:
        return self._model


def _hooked(blocks) -> list[int]:
    return [i for i, b in enumerate(blocks) if b._forward_pre_hooks and b._forward_hooks]


def test_every_h3_topology_group_carries_a_positive_approx_vram_mb():
    # The IModelDriver.get_block_topology contract: name, attr_path, count, approx_vram_mb.
    for def_id in DEF_IDS:
        for group in _load(def_id)["block_topology"]:
            value = group.get("approx_vram_mb")
            assert isinstance(value, int) and value > 0, f"{def_id} group {group['name']}: approx_vram_mb={value!r}"


@needs_pinning
def test_block_swap_engages_on_the_main_blocks():
    model = build_tiny_transformer()
    host = _Host(model, {"block_swap_config": {"transformer_blocks": 50}})
    try:
        host._configure_block_swapping()
        assert _hooked(model.transformer_blocks) == [0]  # round(2 * 50 / 100) = 1, the first block
        assert [len(m.blocks) for m in host._block_swap_managers] == [1]
    finally:
        for manager in getattr(host, "_block_swap_managers", []):
            manager.remove()


@needs_pinning
def test_a_topology_without_approx_vram_mb_still_swaps():
    # 18 shipped definitions carry no approx_vram_mb in YAML; the log line must not end the run.
    model = build_tiny_transformer()
    host = _Host(model, {"block_swap_config": {"transformer_blocks": 100}})
    host.driver.get_block_topology = lambda: [
        {"name": "transformer_blocks", "attr_path": "transformer_blocks", "count": 2}
    ]
    try:
        host._configure_block_swapping()
        assert _hooked(model.transformer_blocks) == [0, 1]
    finally:
        for manager in getattr(host, "_block_swap_managers", []):
            manager.remove()


@needs_pinning
def test_block_swap_reaches_the_nested_refiner_group():
    # `token_refiner.refiner_blocks` is a DOTTED attr_path: a plain getattr answers None
    # and the group is skipped without a word.
    model = build_tiny_transformer()
    host = _Host(model, {"block_swap_config": {"token_refiner_blocks": 100}})
    try:
        host._configure_block_swapping()
        assert _hooked(model.token_refiner.refiner_blocks) == [0]
        assert [len(m.blocks) for m in host._block_swap_managers] == [1]
    finally:
        for manager in getattr(host, "_block_swap_managers", []):
            manager.remove()


@needs_pinning
def test_a_requested_swap_group_that_does_not_resolve_says_so():
    # Shared core: other families keep running, but the skip is no longer silent.
    model = build_tiny_transformer()
    host = _Host(model, {"block_swap_config": {"token_refiner_blocks": 100}})
    host.driver.get_block_topology = lambda: [
        {"name": "token_refiner_blocks", "attr_path": "refiner_blocks", "count": 1, "approx_vram_mb": 1}
    ]
    with structlog.testing.capture_logs() as logs:
        host._configure_block_swapping()
    assert not getattr(host, "_block_swap_managers", [])
    unresolved = [e for e in logs if e["event"] == "block_swap_group_unresolved"]
    assert unresolved and unresolved[0]["attr_path"] == "refiner_blocks" and unresolved[0]["log_level"] == "warning"
