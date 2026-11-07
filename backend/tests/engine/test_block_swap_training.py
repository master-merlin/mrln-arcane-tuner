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
    # root=model, as the pipeline passes it: a child shared with an unswapped block stays whole.
    manager = BlockSwappingManager([blocks[i] for i in which], device=torch.device(device), root=model)
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

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        for manager in self.managers:
            manager.__dict__.pop("_reload", None)

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


def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    """One step; SophiaH takes the trainer's Hessian estimate, as the trainer hands it."""
    kwargs: dict = {}
    if getattr(optimizer, "hessian_period", None) is not None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        kwargs["hessian"] = [None if p.grad is None else p.grad.abs() for p in params]
    optimizer.step(**kwargs)


def run(
    model: nn.Module,
    steps: int,
    managers: list[BlockSwappingManager] | None = None,
    make_optimizer: Callable[[list[nn.Parameter]], torch.optim.Optimizer] | None = None,
    between: Callable[[int], None] | None = None,
    x: torch.Tensor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[list[torch.Tensor]]:
    """Train ``steps`` steps; check I1 and I3 after each; return every trainable param after each."""
    managers = managers or []
    params = trainable(model)
    devices = {id(p): p.device for p in params}
    if optimizer is None:
        optimizer = (make_optimizer or (lambda ps: torch.optim.AdamW(ps, lr=1e-2)))(params)
    width = blocks_of(model)[0].lin.in_features
    x = batch(width) if x is None else x
    observer = Observer(managers) if managers else None
    snapshots = []
    for step in range(steps):
        loss = model(x).pow(2).mean()
        loss.backward()
        optimizer_step(optimizer)
        if managers:
            check_i1(model, optimizer, devices)
            check_i3_idle(managers)
        optimizer.zero_grad(set_to_none=True)
        snapshots.append([p.detach().cpu().clone() for p in params])
        if between is not None:
            between(step)
    if observer is not None:
        observer.close()
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


def test_an_adapter_frozen_after_the_swap_keeps_its_values_readable():
    # Targeted training and adaptive targeting turn adapters off after the swap is set up; the
    # LoRA save, EMA and the adaptive probe then read them outside any forward.
    model = build()
    manager = swap(model, [0])
    block = blocks_of(model)[0]
    adapters = [p for name, p in block.named_parameters() if "lora_" in name]
    for param in adapters:
        param.requires_grad_(False)
    expected = [p.detach().cpu().clone() for p in adapters]
    for _ in range(2):
        with torch.no_grad():
            model(batch(64))
        model(batch(64)).pow(2).mean().backward()
    for param, value in zip(adapters, expected, strict=True):
        assert param.untyped_storage().nbytes() == param.numel() * param.element_size()
        assert torch.equal(param.detach().cpu(), value)
    check_i3_idle([manager])


def test_a_param_made_trainable_after_the_swap_is_left_alone():
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


# ── Task 2: gradient checkpointing, moves and remove ─────────────────────────────────────────────


@pytest.mark.parametrize("ckpt", ["nonreentrant", "reentrant"])
def test_a_checkpointed_swapped_run_equals_the_unswapped_run_bitwise(ckpt):
    # A reentrant checkpoint only builds a graph when its input needs a grad.
    x = batch(64, requires_grad=ckpt == "reentrant")
    reference = run(build(ckpt=ckpt), 4, x=x)
    model = build(ckpt=ckpt)
    swapped = run(model, 4, [swap(model, [0, 1, 2])], x=x)
    assert_bitwise_equal(reference, swapped)


@pytest.mark.parametrize("ckpt", [None, "nonreentrant", "reentrant"])
def test_backward_holds_at_most_two_swapped_blocks_and_ends_with_none(ckpt):
    x = batch(64, requires_grad=ckpt == "reentrant")
    model = build(n_blocks=6, ckpt=ckpt)
    manager = swap(model, range(6))
    observer = Observer([manager])
    model(x).pow(2).mean().backward()
    assert observer.in_backward, "nothing was observed inside backward"
    assert max(observer.in_backward) <= 2, observer.in_backward
    assert manager.filled_blocks() == []


def test_a_model_moved_off_the_card_and_back_trains_bitwise_like_the_unswapped_one():
    def round_trip(model: nn.Module) -> Callable[[int], None]:
        def between(step: int) -> None:
            if step == 0:
                model.to("cpu")
                model.to("cuda")
        return between

    plain = build()
    reference = run(plain, 4, between=round_trip(plain))
    model = build()
    manager = swap(model, [0, 1, 2])
    swapped = run(model, 4, [manager], between=round_trip(model))
    assert_bitwise_equal(reference, swapped)


def test_a_swapped_block_moved_off_the_card_is_whole_there_and_seated_again_on_return():
    model = build()
    original = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    manager = swap(model, [0, 1])
    seated = manager.seated_bytes()
    model.to("cpu")
    assert manager.seated_bytes() == 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.device.type == "cpu" and torch.equal(param.detach(), original[name]), name
    model.to("cuda")
    assert manager.seated_bytes() == seated
    check_i3_idle([manager])


def test_no_grad_forwards_between_steps_keep_the_run_bitwise_equal():
    def sample(model: nn.Module) -> Callable[[int], None]:
        def between(_step: int) -> None:
            with torch.no_grad():
                model(batch(64, seed=11))
        return between

    plain = build()
    reference = run(plain, 4, between=sample(plain))
    model = build()
    manager = swap(model, [0, 1, 2])
    swapped = run(model, 4, [manager], between=lambda step: (sample(model)(step), check_i3_idle([manager])))
    assert_bitwise_equal(reference, swapped)


def test_remove_gives_back_the_original_frozen_weights_and_the_model_still_trains():
    model = build()
    original = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    manager = swap(model, [0, 1, 2])
    run(model, 2, [manager])
    manager.remove()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.untyped_storage().nbytes() > 0, name
            assert torch.equal(param.detach().cpu(), original[name]), name
    assert all("_apply" not in block.__dict__ for block in blocks_of(model))
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in blocks_of(model))
    run(model, 1)


# ── Task 3: every optimizer, fresh and resumed ──────────────────────────────────────────────────


def _every_optimizer() -> dict[str, Callable[[list[nn.Parameter]], torch.optim.Optimizer]]:
    from app.engine.factories.optimizer import OptimizerFactory

    made = {key: (lambda ps, key=key: OptimizerFactory.create(key, ps, lr=1e-3))
            for key in OptimizerFactory.SUPPORTED_OPTIMIZERS}
    made["AdamW-capturable"] = lambda ps: torch.optim.AdamW(ps, lr=1e-3, capturable=True)
    made["AdamW-fused"] = lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True)
    made["SophiaG-capturable"] = lambda ps: OptimizerFactory.create(
        "SophiaG", ps, lr=1e-3, config={"sophia_capturable": True}
    )
    return made


EVERY_OPTIMIZER = _every_optimizer()
FRESH_STEPS, RESUMED_STEPS = 8, 3
# A LoRA A of rank 4 at this width has 4096 elements: AdamW8bit's min_8bit_size, so its state is 8-bit.
OPTIMIZER_WIDTH = 1024


def _trainable_names(model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _fresh_then_resumed(key: str, swapped: list[int]) -> list[list[torch.Tensor]]:
    """8 steps, a save through a file, then the restore the checkpoint manager does (a load, a
    remap by name, ``load_state_dict``) into a new model and optimizer, and 3 more steps."""
    import io

    from app.engine.core.optimization.optimizer_remap import remap_optimizer_state

    make = EVERY_OPTIMIZER[key]
    model = build(width=OPTIMIZER_WIDTH)
    managers = [swap(model, swapped)] if swapped else []
    optimizer = make(trainable(model))
    snapshots = run(model, FRESH_STEPS, managers, optimizer=optimizer)
    if "8bit" in key:
        dtypes = {v.dtype for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)}
        assert torch.uint8 in dtypes, f"{key} made no 8-bit state at this size: {dtypes}"
    buffer = io.BytesIO()
    torch.save(
        {"optimizer": optimizer.state_dict(), "names": _trainable_names(model),
         "params": [p.detach().clone() for p in trainable(model)]},
        buffer,
    )
    for manager in managers:
        manager.remove()
    del model, optimizer, managers

    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    model = build(width=OPTIMIZER_WIDTH)  # the base weights a resume loads again; LoRA values from the save
    with torch.no_grad():
        for param, value in zip(trainable(model), saved["params"], strict=True):
            param.copy_(value)
    managers = [swap(model, swapped)] if swapped else []
    optimizer = make(trainable(model))
    state, unmapped = remap_optimizer_state(saved["optimizer"], saved["names"], _trainable_names(model))
    assert unmapped == []
    optimizer.load_state_dict(state)
    return snapshots + run(model, RESUMED_STEPS, managers, optimizer=optimizer)


@pytest.mark.parametrize("key", list(EVERY_OPTIMIZER))
def test_every_optimizer_trains_a_swapped_run_bitwise_like_the_unswapped_one_across_a_resume(key):
    reference = _fresh_then_resumed(key, [])
    swapped = _fresh_then_resumed(key, [0, 1, 2])
    assert_bitwise_equal(reference, swapped)


# ── Task 4: quantized weights ───────────────────────────────────────────────────────────────────


class QuantBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width, dtype=torch.bfloat16)
        self.ln = nn.LayerNorm(width, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.lin(x)) + x


class QuantNet(nn.Module):
    def __init__(self, n_blocks: int = 4, width: int = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([QuantBlock(width) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _schemes() -> list[tuple[str, str]]:
    if not torch.cuda.is_available():
        return []
    from app.engine.factories.quantization import QuantizationFactory

    return [(backend, scheme) for backend, schemes in QuantizationFactory.get_supported_capabilities().items()
            for scheme in schemes]


def _quantized_run(backend: str, scheme: str, swapped: bool) -> list[list[torch.Tensor]]:
    """Setup's order: frozen, quantized on the host, onto the card, LoRA, then the swap."""
    from peft import LoraConfig, get_peft_model

    from app.engine.factories.quantization import QuantizationFactory

    torch.manual_seed(0)
    net = QuantNet()
    for param in net.parameters():
        param.requires_grad_(False)
    net = QuantizationFactory.quantize(net, scheme, backend_name=backend).to("cuda")
    model = get_peft_model(net, LoraConfig(r=4, lora_alpha=4, target_modules=["lin"]))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param.dtype))
    managers = [swap(model, [0, 1, 2])] if swapped else []
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(2))
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-3)
    snapshots = []
    for _ in range(3):
        model(x).float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if managers:
            check_i3_idle(managers)
        snapshots.append([p.detach().float().cpu().clone() for p in trainable(model)])
    return snapshots


@pytest.mark.parametrize(("backend", "scheme"), _schemes(), ids=lambda v: str(v))
def test_every_quantized_scheme_that_trains_unswapped_trains_swapped_within_its_own_run_to_run_bound(
    backend, scheme
):
    try:
        first = _quantized_run(backend, scheme, False)
        second = _quantized_run(backend, scheme, False)
    except Exception as error:  # noqa: BLE001 - the scheme's own failure, with no swap at all
        pytest.skip(f"{backend}/{scheme} fails LoRA training with no swap (pre-existing): {str(error)[:160]}")
    swapped = _quantized_run(backend, scheme, True)
    for step, (a, b, got) in enumerate(zip(first, second, swapped, strict=True)):
        for i, (x, y, z) in enumerate(zip(a, b, got, strict=True)):
            bound = (x - y).abs().max().item()
            seen = (x - z).abs().max().item()
            assert seen <= bound, f"step {step}, param {i}: swapped differs by {seen}, unswapped runs by {bound}"


# ── Graphs, aliases, failures and placements the chain of blocks above does not have ──────────────


class TwoStreamBlock(Block):
    """Returns ``(h, side)``, where ``side`` is one no later block reads:

    - ``passthrough``: its second input, unchanged;
    - ``branch``: a second output only the loss reads;
    - ``leaf``: a leaf tensor it was handed outside its inputs. Autograd accumulates a leaf's grad
      at top priority, so the hook on this output fires at the start of backward.
    """

    carried: torch.Tensor | None = None

    def __init__(self, width: int, side: str) -> None:
        super().__init__(width)
        self.side = side

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = super().forward(x)
        if self.side == "leaf":
            return h, self.carried
        return h, (y if self.side == "passthrough" else h * 2.0)


class TwoStreamNet(nn.Module):
    def __init__(self, n_blocks: int, width: int, side: str, ckpt: str | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TwoStreamBlock(width, side) for _ in range(n_blocks)])
        self.ckpt = ckpt

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        sides = []
        for block in self.blocks:
            block.carried = y
            # A leaf handed in as an input as well would get its grad only once every block is done.
            second = y.detach() if block.side == "leaf" else y
            if self.ckpt is None or not torch.is_grad_enabled():
                x, side = block(x, second)
            else:
                x, side = checkpoint(block, x, second, use_reentrant=self.ckpt == "reentrant")
            sides.append(side)
        return x.pow(2).mean() + sum(s.pow(2).mean() for s in sides)


def build_two_stream(
    n_blocks: int, width: int, side: str, ckpt: str | None = None, lora_blocks: Iterable[int] | None = None
) -> nn.Module:
    """``build``'s PEFT model and LoRA values, over blocks with a second output."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    targets = ["lin"] if lora_blocks is None else [f"blocks.{i}.lin" for i in lora_blocks]
    model = get_peft_model(TwoStreamNet(n_blocks, width, side, ckpt), LoraConfig(r=4, target_modules=targets))
    model = model.to("cuda")
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen))
    return model


@pytest.mark.parametrize("side", ["passthrough", "branch", "leaf"])
def test_a_block_with_an_output_no_later_block_reads_trains_bitwise_like_unswapped(side):
    # An output whose grad is ready before the block's own backward must not end the block before it.
    def train(swapped: bool) -> list[list[torch.Tensor]]:
        m = build_two_stream(4, 64, side)
        managers = [swap(m, [0, 1, 2, 3])] if swapped else []
        optimizer = torch.optim.AdamW(trainable(m), lr=1e-2)
        x, y = batch(64, requires_grad=True), batch(64, seed=9, requires_grad=True)
        out = []
        for _ in range(3):
            m(x, y).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if managers:
                check_i3_idle(managers)
            out.append([p.detach().cpu().clone() for p in trainable(m)])
        return out

    assert_bitwise_equal(train(False), train(True))


def test_a_leaf_a_block_hands_back_collects_no_hook_across_steps():
    # A hook on a leaf outlives the graph: one per step would pile up and all fire every backward.
    model = build_two_stream(3, 64, "leaf")
    manager = swap(model, [0, 1, 2])
    y = batch(64, seed=9, requires_grad=True)
    for _ in range(3):
        model(batch(64), y).backward()
    assert not y._backward_hooks, f"{len(y._backward_hooks)} hooks left on the leaf"
    check_i3_idle([manager])


def test_a_frozen_param_that_shares_storage_with_a_trainable_param_or_a_buffer_leaves_it_whole():
    model = build()
    block = blocks_of(model)[0]
    block.alias = nn.Parameter(block.lin.weight.detach())  # trainable, same storage as a frozen weight
    block.register_buffer("w_copy", block.w.detach())       # a buffer over a frozen weight's storage
    manager = swap(model, [0, 1])
    with torch.no_grad():
        model(batch(64))
    assert block.alias.untyped_storage().nbytes() == block.alias.numel() * 4
    assert block.w_copy.untyped_storage().nbytes() == block.w_copy.numel() * 4
    check_i3_idle([manager])


def test_a_frozen_weight_tied_to_a_module_outside_the_swapped_blocks_stays_whole():
    model = build()
    blocks = blocks_of(model)
    outside = nn.Linear(64, 64).to("cuda")
    outside.weight = blocks[0].lin.base_layer.weight  # tied: the head reads it after the blocks ran
    manager = BlockSwappingManager([blocks[0], blocks[1]], device=torch.device("cuda"), root=nn.ModuleList([model, outside]))
    manager.apply()
    with torch.no_grad():
        model(batch(64))
    assert outside.weight.untyped_storage().nbytes() == outside.weight.numel() * 4  # before any read of it
    assert torch.equal(outside.weight.detach().cpu(), blocks[0].lin.base_layer.weight.detach().cpu())
    check_i3_idle([manager])


def test_a_forward_that_raises_leaves_no_swapped_block_filled():
    model = build()
    manager = swap(model, [0, 1, 2])
    failing = blocks_of(model)[2]

    def boom(_module: nn.Module, _args: tuple) -> None:
        raise ValueError("out of memory, say")

    handle = failing.register_forward_pre_hook(boom)
    for grad in (False, True):
        with torch.set_grad_enabled(grad), pytest.raises(ValueError):
            model(batch(64))
        assert manager.filled_blocks() == [], f"grad={grad}: {len(manager.filled_blocks())} left filled"
    handle.remove()
    run(model, 2, [manager])


def test_two_swapped_blocks_sharing_one_child_module_train_bitwise_like_unswapped():
    def shared(model: nn.Module) -> nn.Module:
        blocks = blocks_of(model)
        blocks[1].ln = blocks[0].ln  # one LayerNorm, run by both blocks
        return model

    reference = run(shared(build()), 3)
    model = shared(build())
    swapped = run(model, 3, [swap(model, [0, 1, 2])])
    assert_bitwise_equal(reference, swapped)
    assert blocks_of(model)[0].ln.weight.untyped_storage().nbytes() > 0, "the shared child is never freed"


def test_applying_the_swap_to_host_blocks_never_holds_more_than_one_extra_block_on_the_card():
    # A model that fits only with the swap must not need every block on the card first.
    model = build(n_blocks=6, width=1024, device="cpu")
    one_block = sum(p.numel() * p.element_size() for p in blocks_of(model)[0].parameters() if not p.requires_grad)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    manager = swap(model, range(6))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    trainable_bytes = sum(p.numel() * p.element_size() for p in trainable(model))
    assert peak <= 1.5 * one_block + trainable_bytes, (peak, one_block)
    assert manager.seated_bytes() == 6 * one_block


def test_a_block_still_on_the_host_when_the_swap_is_applied_trains_on_the_card():
    reference = run(build(), 3)
    model = build(device="cpu")
    manager = swap(model, range(4))
    assert manager.seated_bytes() > 0
    assert all(p.device.type == "cuda" for p in model.parameters())
    swapped = run(model, 3, [manager])
    assert_bitwise_equal(reference, swapped)


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
