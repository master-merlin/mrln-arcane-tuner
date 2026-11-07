"""Block swapping's three invariants, each over 50 seeded random configurations.

A draw picks a graph shape (a chain of blocks; blocks that also return their second input
unchanged; blocks with a second output only the loss reads; blocks that return a leaf handed to
them outside their inputs, whose grad backward makes first), 2 to 6 blocks, a random non-empty
swapped subset, a checkpoint kind (none, non-reentrant, reentrant), LoRA on a random non-empty
subset of blocks, and 2 to 4 steps with a random action after each: nothing, a no-grad forward, or
a ``.to("cpu")``/``.to("cuda")`` round trip. The swapped run is compared with an unswapped twin
that takes the same actions.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from tests.engine.test_block_swap_training import (
    Observer,
    batch,
    build,
    build_two_stream,
    check_i3_idle,
    swap,
    trainable,
)

pytestmark = [
    pytest.mark.xdist_group("gpu"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="block swapping frees card memory: needs a card"),
]

DRAWS = 50
WIDTH = 32


@dataclass(frozen=True)
class Draw:
    shape: str  # "chain", "passthrough" or "branch"
    n_blocks: int
    swapped: tuple[int, ...]
    ckpt: str | None
    lora: tuple[int, ...]
    actions: tuple[str, ...]  # one per step: "none", "sample" or "move"
    shared_ln: bool = False   # the last block runs the first block's LayerNorm too


def draw(index: int) -> Draw:
    rng = random.Random(18400 + index)
    shape = rng.choice(["chain", "passthrough", "branch", "leaf"])
    n = rng.randint(2, 6)
    swapped = tuple(sorted(rng.sample(range(n), rng.randint(1, n))))
    lora = tuple(sorted(rng.sample(range(n), rng.randint(1, n))))
    ckpt = rng.choice([None, "nonreentrant", "reentrant"])
    if shape == "leaf" and ckpt == "reentrant":
        # A reentrant checkpoint hands a leaf it did not take as input back as its own output, so
        # its recompute backpropagates into itself without end, swap or no swap.
        ckpt = "nonreentrant"
    actions = tuple(rng.choice(["none", "sample", "move"]) for _ in range(rng.randint(2, 4)))
    shared_ln = rng.random() < 0.3
    return Draw(shape, n, swapped, ckpt, lora, actions, shared_ln)


def _model(d: Draw) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """The model and its loss for one input; a two-stream model gets its second input too."""
    if d.shape == "chain":
        model = build(n_blocks=d.n_blocks, width=WIDTH, ckpt=d.ckpt, lora_blocks=d.lora)
        loss_of: Callable[[torch.Tensor], torch.Tensor] = lambda x: model(x).pow(2).mean()  # noqa: E731
    else:
        model = build_two_stream(d.n_blocks, WIDTH, d.shape, ckpt=d.ckpt, lora_blocks=d.lora)
        y = batch(WIDTH, seed=9, requires_grad=True)
        loss_of = lambda x: model(x, y)  # noqa: E731
    if d.shared_ln:
        blocks = model.base_model.model.blocks
        blocks[-1].ln = blocks[0].ln
    return model, loss_of


@dataclass
class Outcome:
    snapshots: list[list[torch.Tensor]]
    devices_ok: list[str]          # I1 breaches, described
    idle_breaches: list[str]       # I3 breaches at the idle observation points
    in_forward: list[int]
    in_backward: list[int]


def play(d: Draw, swapped: bool) -> Outcome:
    model, loss_of = _model(d)
    managers = [swap(model, d.swapped)] if swapped else []
    observer = Observer(managers) if managers else None
    params = trainable(model)
    start = {id(p): p.device for p in params}
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = batch(WIDTH, requires_grad=d.ckpt == "reentrant")
    out = Outcome([], [], [], [], [])

    def idle(where: str) -> None:
        for manager in managers:
            if manager.filled_blocks():
                out.idle_breaches.append(f"{where}: {len(manager.filled_blocks())} filled")

    actions: dict[str, Callable[[], None]] = {
        "none": lambda: None,
        "sample": lambda: _sample(loss_of),
        "move": lambda: (model.to("cpu"), model.to("cuda")),
    }
    for step, action in enumerate(d.actions):
        loss_of(x).backward()
        optimizer.step()
        for param in params:
            if param.device != start[id(param)]:
                out.devices_ok.append(f"step {step}: param moved to {param.device}")
            if param.grad is not None and param.grad.device != param.device:
                out.devices_ok.append(f"step {step}: grad on {param.grad.device}, param on {param.device}")
            for key, value in optimizer.state[param].items():
                if torch.is_tensor(value) and not (key == "step" and value.device.type == "cpu"):
                    if value.device != param.device:
                        out.devices_ok.append(f"step {step}: state {key} on {value.device}")
        idle(f"after step {step}")
        optimizer.zero_grad(set_to_none=True)
        out.snapshots.append([p.detach().cpu().clone() for p in params])
        actions[action]()
        idle(f"after {action} at step {step}")
    if observer is not None:
        observer.close()
        out.in_forward, out.in_backward = observer.in_forward, observer.in_backward
    if managers:
        check_i3_idle(managers)  # non-vacuous: something was seated
    return out


def _sample(loss_of: Callable[[torch.Tensor], torch.Tensor]) -> None:
    with torch.no_grad():
        loss_of(batch(WIDTH, seed=11))


def test_i1_a_trainable_param_its_grad_and_its_state_share_its_device_in_every_draw():
    for index in range(DRAWS):
        d = draw(index)
        breaches = play(d, swapped=True).devices_ok
        assert not breaches, f"draw {index} {d}: {breaches[:3]}"


def test_i2_the_swapped_run_equals_the_unswapped_twin_bitwise_in_every_draw():
    for index in range(DRAWS):
        d = draw(index)
        reference, got = play(d, swapped=False).snapshots, play(d, swapped=True).snapshots
        for step, (ref, swp) in enumerate(zip(reference, got, strict=True)):
            for i, (a, b) in enumerate(zip(ref, swp, strict=True)):
                assert torch.equal(a, b), f"draw {index} {d}: step {step}, param {i}"
        assert not all(torch.equal(a, b) for a, b in zip(reference[0], reference[-1])), f"draw {index}: never moved"


def test_i3_swapped_weights_hold_card_memory_only_while_their_block_works_in_every_draw():
    for index in range(DRAWS):
        d = draw(index)
        out = play(d, swapped=True)
        assert not out.idle_breaches, f"draw {index} {d}: {out.idle_breaches[:3]}"
        assert out.in_forward and all(n == 1 for n in out.in_forward), f"draw {index} {d}: {out.in_forward}"
        if d.shape == "chain":
            # A block is filled from its refill until its own backward is done: in a chain, the next
            # block's refill comes after that, so at most this block and the one before it.
            assert all(n <= 2 for n in out.in_backward), f"draw {index} {d}: {out.in_backward}"


def test_the_draws_cover_every_kind():
    draws = [draw(i) for i in range(DRAWS)]
    assert {d.shape for d in draws} == {"chain", "passthrough", "branch", "leaf"}
    assert any(d.shared_ln and {0, d.n_blocks - 1} <= set(d.swapped) for d in draws), "two swapped blocks share one"
    assert {d.ckpt for d in draws} == {None, "nonreentrant", "reentrant"}
    assert {a for d in draws for a in d.actions} == {"none", "sample", "move"}
    assert {d.n_blocks for d in draws} == {2, 3, 4, 5, 6}
    assert any(set(d.lora) != set(range(d.n_blocks)) for d in draws), "LoRA on some blocks only"
