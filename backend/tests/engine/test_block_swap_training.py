"""A block-swapped LoRA run trains like an unswapped one, and saves the VRAM the estimate credits.

Block swapping used to move every param of a swapped block to pinned host memory after its forward,
LoRA included. The first optimizer step then mixed the LoRA param on the host with its grad on the
card, and a LayerNorm's weight saved for backward was read from the host. The manager now keeps every
param on its device and frees a frozen weight's card memory in place between the times its block
runs. These tests hold it to three invariants, each checked against an unswapped twin on the card:

- I1: a trainable param never changes device because of the swap; after every optimizer step the
  param, its grad and its optimizer state share one device (a host ``step`` counter excepted).
- I2: the swapped run's trainable params equal the unswapped twin's, bitwise, after every step.
- I3: a swapped block's frozen weights hold card memory only while the block works: none after a
  step or a no-grad forward, the one running block during forward, at most 2 blocks inside backward.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

if __name__ != "__main__":
    pytestmark = [
        pytest.mark.xdist_group("gpu"),
        pytest.mark.skipif(not torch.cuda.is_available(), reason="block swapping frees card memory: needs a card"),
    ]

from app.engine.core.optimization.block_swapping import BlockSwappingManager  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── The rig: blocks whose backward reads frozen weights the way WAN's blocks do ──────────────────


class Block(nn.Module):
    """A ``Linear`` (LoRA target), an RMSNorm-style multiply by a weight, and a LayerNorm.

    Autograd saves the multiply's weight and the LayerNorm's weight for backward, so a frozen weight
    that is not where autograd left it fails or misleads backward, not just forward.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.w = nn.Parameter(torch.full((width,), 1.5))
        self.ln = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.lin(x) * self.w) + x


class Net(nn.Module):
    def __init__(self, n_blocks: int, width: int, ckpt: str | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(n_blocks)])
        self.ckpt = ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        for block in self.blocks:
            if self.ckpt is None or not self.training or not torch.is_grad_enabled():
                x = block(x)
            else:
                x = checkpoint(block, x, use_reentrant=self.ckpt == "reentrant")
        return x


def build(
    n_blocks: int = 4,
    width: int = 64,
    ckpt: str | None = None,
    lora_blocks: Iterable[int] | None = None,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """A PEFT model of ``Net``; LoRA on every block's ``lin`` unless ``lora_blocks`` names some."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    net = Net(n_blocks, width, ckpt)
    targets = ["lin"] if lora_blocks is None else [f"blocks.{i}.lin" for i in lora_blocks]
    model = get_peft_model(net, LoraConfig(r=4, lora_alpha=4, target_modules=targets)).to(device)
    gen = torch.Generator().manual_seed(seed + 1)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.copy_(torch.randn(param.shape, generator=gen))
    return model


def blocks_of(model: nn.Module) -> nn.ModuleList:
    return model.base_model.model.blocks


def batch(width: int, seed: int = 7, device: str = "cuda", requires_grad: bool = False) -> torch.Tensor:
    gen = torch.Generator(device).manual_seed(seed)
    return torch.randn(4, width, device=device, generator=gen).requires_grad_(requires_grad)


def trainable(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def swap(model: nn.Module, which: Iterable[int], device: str | torch.device = "cuda") -> BlockSwappingManager:
    blocks = blocks_of(model)
    manager = BlockSwappingManager([blocks[i] for i in which], device=torch.device(device))
    manager.apply()
    return manager


# ── The invariants, each at its observation points ──────────────────────────────────────────────


def check_i1(model: nn.Module, optimizer: torch.optim.Optimizer, devices: dict[int, torch.device]) -> None:
    """After a step: each trainable param is where it started, with its grad and state beside it."""
    for param in trainable(model):
        assert param.device == devices[id(param)], f"a trainable param moved: {devices[id(param)]} -> {param.device}"
        if param.grad is not None:
            assert param.grad.device == param.device, f"grad on {param.grad.device}, param on {param.device}"
        for key, value in optimizer.state.get(param, {}).items():
            if not torch.is_tensor(value) or (key == "step" and value.device.type == "cpu"):
                continue
            assert value.device == param.device, f"state {key!r} on {value.device}, param on {param.device}"


def check_i3_idle(managers: Iterable[BlockSwappingManager]) -> None:
    """After a step or a no-grad forward: no swapped block holds any card memory for its weights."""
    for manager in managers:
        assert manager.seated_bytes() > 0, "nothing was seated: the check would pass vacuously"
        assert manager.filled_blocks() == [], f"{len(manager.filled_blocks())} swapped blocks still filled"


class Observer:
    """Records how many swapped blocks are filled at each forward pre-hook and each backward refill."""

    def __init__(self, managers: list[BlockSwappingManager]) -> None:
        self.managers = managers
        self.in_forward: list[int] = []
        self.in_backward: list[int] = []
        self._handles = []
        for manager in managers:
            for block in manager.blocks:
                # Registered after the manager's own pre-hook, so it runs once the block is filled.
                self._handles.append(block.register_forward_pre_hook(self._on_pre))
            original = manager._reload

            def reload(block: nn.Module, _original: Callable = original) -> None:
                _original(block)
                self.in_backward.append(self.filled())

            manager._reload = reload

    def filled(self) -> int:
        return sum(len(m.filled_blocks()) for m in self.managers)

    def _on_pre(self, _module: nn.Module, _args: tuple) -> None:
        if torch._C._current_graph_task_id() != -1:
            self.in_backward.append(self.filled())
        else:
            self.in_forward.append(self.filled())


def check_i3_working(observer: Observer) -> None:
    assert observer.in_forward and all(n == 1 for n in observer.in_forward), observer.in_forward
    assert all(n <= 2 for n in observer.in_backward), observer.in_backward


def run(
    model: nn.Module,
    steps: int,
    managers: list[BlockSwappingManager] | None = None,
    make_optimizer: Callable[[list[nn.Parameter]], torch.optim.Optimizer] | None = None,
    between: Callable[[int], None] | None = None,
    x: torch.Tensor | None = None,
) -> list[list[torch.Tensor]]:
    """Train ``steps`` steps; check I1 and I3 after each; return every trainable param after each."""
    managers = managers or []
    params = trainable(model)
    devices = {id(p): p.device for p in params}
    optimizer = (make_optimizer or (lambda ps: torch.optim.AdamW(ps, lr=1e-2)))(params)
    width = blocks_of(model)[0].lin.in_features
    x = batch(width) if x is None else x
    observer = Observer(managers) if managers else None
    snapshots = []
    for step in range(steps):
        loss = model(x).pow(2).mean()
        loss.backward()
        optimizer.step()
        if managers:
            check_i1(model, optimizer, devices)
            check_i3_idle(managers)
        optimizer.zero_grad(set_to_none=True)
        snapshots.append([p.detach().cpu().clone() for p in params])
        if between is not None:
            between(step)
    if observer is not None:
        check_i3_working(observer)
    return snapshots


def assert_bitwise_equal(reference: list[list[torch.Tensor]], swapped: list[list[torch.Tensor]]) -> None:
    assert len(reference) == len(swapped)
    for step, (ref, got) in enumerate(zip(reference, swapped)):
        for i, (a, b) in enumerate(zip(ref, got)):
            assert torch.equal(a, b), f"step {step}, trainable param {i}: max diff {(a - b).abs().max().item()}"
    if len(reference) > 1:
        # A run that never updates would pass by standing still.
        assert not all(torch.equal(a, b) for a, b in zip(reference[0], reference[-1])), "the params never changed"


# ── Task 1: a swapped block trains ─────────────────────────────────────────────────────────────


def test_a_swapped_lora_block_takes_an_optimizer_step():
    # The card's defect: the first step mixed a host LoRA param with its card grad.
    # Plain Linears only, so backward gets through and the step is what meets the mixed devices.
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
    model = get_peft_model(net, LoraConfig(r=4, target_modules=["0", "1"])).to("cuda")
    manager = BlockSwappingManager([model.base_model.model[0]], device=torch.device("cuda"))
    manager.apply()
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-2)
    model(batch(64)).pow(2).mean().backward()
    optimizer.step()
    for param in trainable(model):
        assert param.device == param.grad.device == optimizer.state[param]["exp_avg"].device


def test_a_swapped_layernorm_weight_is_read_on_the_card_in_backward():
    # A block with no LoRA at all: backward still reads its LayerNorm and multiply weights.
    model = build(lora_blocks=[3])
    manager = swap(model, [0, 1, 2])
    x = batch(64, requires_grad=True)
    model(x).pow(2).mean().backward()
    assert x.grad is not None and x.grad.device.type == "cuda"
    check_i3_idle([manager])


def test_a_swapped_run_equals_the_unswapped_run_bitwise_after_every_step():
    reference = run(build(), 4)
    model = build()
    swapped = run(model, 4, [swap(model, [0, 1, 2])])
    assert_bitwise_equal(reference, swapped)


def test_a_trainable_param_keeps_its_device_and_frozen_ones_stay_on_the_card():
    model = build()
    manager = swap(model, [0, 1])
    run(model, 2, [manager])
    for param in blocks_of(model)[0].parameters():
        assert param.device.type == "cuda"


def test_a_no_grad_forward_frees_every_swapped_block():
    model = build()
    manager = swap(model, [0, 1, 2])
    with torch.no_grad():
        model(batch(64))
    check_i3_idle([manager])


def test_a_buffer_stays_resident():
    model = build()
    blocks_of(model)[0].register_buffer("counter", torch.zeros(3, device="cuda"))
    manager = swap(model, [0])
    with torch.no_grad():
        model(batch(64))
    assert blocks_of(model)[0].counter.untyped_storage().nbytes() == 12
    check_i3_idle([manager])


def test_a_param_frozen_after_the_swap_is_seated_and_one_made_trainable_is_left_alone():
    model = build()
    manager = swap(model, [0])
    block = blocks_of(model)[0]
    before = manager.seated_bytes()
    block.ln.weight.requires_grad_(True)  # trained from now on: must never be freed
    with torch.no_grad():
        model(batch(64))
    assert block.ln.weight.untyped_storage().nbytes() == block.ln.weight.numel() * 4
    assert manager.seated_bytes() == before - block.ln.weight.numel() * 4


def test_the_estimate_credits_what_a_swapped_block_frees():
    # The credit is per block, the seated bytes of its frozen weights.
    model = build(n_blocks=4, width=256)
    manager = swap(model, [0, 1, 2, 3])
    one_block = sum(p.numel() * p.element_size() for p in blocks_of(model)[0].parameters() if not p.requires_grad)
    assert manager.seated_bytes() == 4 * one_block


# ── The VRAM saving, measured: each twin in its own process ──────────────────────────────────────

VRAM_BLOCKS = 8
VRAM_WIDTH = 2048
VRAM_SWAPPED = 6


def _vram_child(n_swapped: int) -> dict[str, float]:
    """One twin: built, warmed up with a full step, then measured over one more step."""
    model = build(n_blocks=VRAM_BLOCKS, width=VRAM_WIDTH)
    manager = swap(model, range(n_swapped)) if n_swapped else None
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-3)
    x = batch(VRAM_WIDTH)

    def step() -> float:
        loss = model(x).pow(2).mean()
        after_forward = torch.cuda.memory_allocated()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return after_forward

    step()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    after_forward = step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    block = blocks_of(model)[0]
    return {
        "after_forward": after_forward,
        "peak": peak,
        "seated": manager.seated_bytes() if manager else 0,
        "block_frozen": sum(p.numel() * p.element_size() for p in block.parameters() if not p.requires_grad),
    }


def _measure(n_swapped: int) -> dict[str, float]:
    env = {**os.environ, "PYTHONPATH": BACKEND}
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--vram-child", str(n_swapped)],
        capture_output=True, text=True, cwd=BACKEND, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_vram_the_swap_frees_its_blocks_in_training_and_the_estimate_matches():
    from app.engine.utils.vram_estimator import _block_swap_credit_mb

    plain = _measure(0)
    swapped = _measure(VRAM_SWAPPED)
    seated, block = swapped["seated"], swapped["block_frozen"]
    assert seated == VRAM_SWAPPED * block

    saved_after_forward = plain["after_forward"] - swapped["after_forward"]
    saved_at_peak = plain["peak"] - swapped["peak"]
    assert saved_after_forward >= 0.9 * seated, (saved_after_forward, seated)
    assert saved_at_peak >= 0.9 * (seated - 2 * block), (saved_at_peak, seated, block)

    definition = SimpleNamespace(
        block_topology=[{"name": "blocks", "count": VRAM_BLOCKS, "approx_vram_mb": block / 2**20}]
    )
    config = {"block_swap_config": {"blocks": round(100 * VRAM_SWAPPED / VRAM_BLOCKS)}}
    credit_mb = _block_swap_credit_mb(definition, config, 1.0)
    measured_mb = saved_after_forward / 2**20
    assert abs(credit_mb - measured_mb) <= 0.1 * measured_mb, (credit_mb, measured_mb)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--vram-child":
    print(json.dumps(_vram_child(int(sys.argv[2]))))
