"""A restored expert's optimizer state goes where a run that never stopped has it.

An optimizer makes its state lazily, on the device its parameter has then, so in a swap-mode run
it's on the card, and it stays there while the expert is off it. ``load_state_dict`` instead casts it
to the device the parameter has now. The first half runs anywhere, on tensors that record their
moves. The second half runs every optimizer the product offers on a card, as it has no other way to
say where it keeps what.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.engine.core.optimization.optimizer_remap import place_optimizer_state
from app.engine.factories.optimizer import OptimizerFactory


class _Recorded(torch.Tensor):
    """Says where it was sent, without a card to send it to."""

    moved_to: str | None = None

    def to(self, *args: Any, **kwargs: Any) -> _Recorded:
        assert not kwargs and len(args) == 1
        self.moved_to = str(args[0])
        return self


def _recorded() -> _Recorded:
    return torch.zeros(2).as_subclass(_Recorded)


def _optimizer(**group: Any) -> tuple[torch.optim.Optimizer, nn.Parameter]:
    param = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.param_groups[0].update(group)
    return optimizer, param


def _moved(optimizer: torch.optim.Optimizer, param: nn.Parameter) -> dict[str, str | None]:
    return {key: value.moved_to for key, value in optimizer.state[param].items() if isinstance(value, _Recorded)}


def test_every_tensor_moves_but_the_step_a_plain_group_keeps_off_the_card():
    optimizer, param = _optimizer()
    optimizer.state[param] = {"exp_avg": _recorded(), "exp_avg_sq": _recorded(), "p0": _recorded(), "step": _recorded()}

    place_optimizer_state(optimizer, [param], "cuda:0")

    assert _moved(optimizer, param) == {"exp_avg": "cuda:0", "exp_avg_sq": "cuda:0", "p0": "cuda:0", "step": None}


@pytest.mark.parametrize("flag", ["capturable", "fused"])
def test_a_group_that_keeps_its_step_on_the_card_gets_it_moved(flag):
    optimizer, param = _optimizer(**{flag: True})
    optimizer.state[param] = {"exp_avg": _recorded(), "step": _recorded()}

    place_optimizer_state(optimizer, [param], "cuda:0")

    assert _moved(optimizer, param) == {"exp_avg": "cuda:0", "step": "cuda:0"}


def test_only_the_given_params_move():
    given = nn.Parameter(torch.zeros(2))
    other = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([given, other], lr=0.1)
    optimizer.state[given] = {"exp_avg": _recorded()}
    optimizer.state[other] = {"exp_avg": _recorded()}

    place_optimizer_state(optimizer, [given], "cuda:0")

    assert _moved(optimizer, given) == {"exp_avg": "cuda:0"}
    assert _moved(optimizer, other) == {"exp_avg": None}


def test_a_param_in_another_group_is_read_under_that_groups_flags():
    plain = nn.Parameter(torch.zeros(2))
    capturing = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([{"params": [plain]}, {"params": [capturing], "capturable": True}], lr=0.1)
    optimizer.state[plain] = {"step": _recorded()}
    optimizer.state[capturing] = {"step": _recorded()}

    place_optimizer_state(optimizer, [plain, capturing], "cuda:0")

    assert _moved(optimizer, plain) == {"step": None}
    assert _moved(optimizer, capturing) == {"step": "cuda:0"}


def test_it_assumes_no_shape_of_state():
    """8-bit's int step, Adafactor's float, StableAdamW's ``None``, and containers as torch reads them."""
    optimizer, param = _optimizer()
    nested, in_list, in_tuple = _recorded(), _recorded(), _recorded()
    optimizer.state[param] = {
        "step": 3,
        "RMS": 0.5,
        "kahan_comp": None,
        "name": "x",
        "nested": {"state1": nested, "step": _recorded()},
        "list": [in_list, 1],
        "tuple": (in_tuple, None),
    }

    place_optimizer_state(optimizer, [param], "cuda:0")

    state = optimizer.state[param]
    assert (state["step"], state["RMS"], state["kahan_comp"], state["name"]) == (3, 0.5, None, "x")
    assert nested.moved_to == in_list.moved_to == in_tuple.moved_to == "cuda:0"
    assert state["nested"]["step"].moved_to is None
    assert state["list"][1] == 1 and state["tuple"][1] is None
    assert isinstance(state["list"], list) and isinstance(state["tuple"], tuple)


def test_a_param_that_never_stepped_gets_no_state():
    """Empty state would keep an optimizer's lazy start from running, and ``state`` is a defaultdict."""
    optimizer, param = _optimizer()

    place_optimizer_state(optimizer, [param], "cuda:0")

    assert param not in optimizer.state


def test_an_empty_list_of_params_and_a_real_optimizer_on_the_cpu_change_nothing():
    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    param.sum().backward()
    optimizer.step()
    before = {key: value.clone() for key, value in optimizer.state[param].items()}

    place_optimizer_state(optimizer, [], "cpu")
    place_optimizer_state(optimizer, [param], "cpu")

    assert all(torch.equal(optimizer.state[param][key], value) for key, value in before.items())


# ── Every optimizer offered, on a card ──


CARD = torch.device("cuda:0") if torch.cuda.is_available() else None
needs_a_card = pytest.mark.skipif(CARD is None, reason="needs a card to move anything")

VARIANTS = {
    "AdamW-capturable": lambda params: torch.optim.AdamW(params, lr=1e-3, capturable=True),
    "AdamW-fused": lambda params: torch.optim.AdamW(params, lr=1e-3, fused=True),
    "SophiaG-capturable": lambda params: OptimizerFactory.create(
        "SophiaG", params, lr=1e-3, config={"sophia_capturable": True}
    ),
}
EVERY_OPTIMIZER = [*OptimizerFactory.SUPPORTED_OPTIMIZERS, *VARIANTS]


def _build(key: str, params: list[nn.Parameter]) -> torch.optim.Optimizer:
    if key in VARIANTS:
        return VARIANTS[key](params)
    return OptimizerFactory.create(key, params, lr=1e-3)


def _experts() -> list[nn.Module]:
    torch.manual_seed(0)
    return [nn.Sequential(nn.Linear(4, 4)), nn.Sequential(nn.Linear(4, 4))]


def _step(optimizer: torch.optim.Optimizer, expert: nn.Module) -> None:
    expert(torch.ones(2, 4, device=CARD)).pow(2).mean().backward()
    # SophiaH takes the trainer's estimate, with an entry only for what this window reached.
    kwargs: dict[str, Any] = {}
    if getattr(optimizer, "hessian_period", None) is not None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        kwargs["hessian"] = [None if p.grad is None else p.grad.abs() for p in params]
    optimizer.step(**kwargs)
    optimizer.zero_grad(set_to_none=True)


def _places(optimizer: torch.optim.Optimizer, expert: nn.Module) -> dict[str, set[str]]:
    places: dict[str, set[str]] = {}
    for param in expert.parameters():
        for key, value in optimizer.state[param].items():
            if torch.is_tensor(value):
                places.setdefault(key, set()).add(value.device.type)
    return places


def _saved_after_stepping_both(key: str) -> tuple[dict[str, Any], list[dict[str, set[str]]]]:
    """The state as a checkpoint reads it, and its places in the run that made it."""
    experts = [expert.to(CARD) for expert in _experts()]
    optimizer = _build(key, [p for expert in experts for p in expert.parameters()])
    for expert in (experts[0], experts[1], experts[0], experts[1]):
        _step(optimizer, expert)
    places = [_places(optimizer, expert) for expert in experts]
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True), places


def _restored(key: str) -> tuple[torch.optim.Optimizer, list[nn.Module], list[dict[str, set[str]]]]:
    """A resume: one expert on the card, the other off it, and the state as the load casts it."""
    saved, fresh_places = _saved_after_stepping_both(key)
    experts = _experts()
    experts[0].to(CARD)
    optimizer = _build(key, [p for expert in experts for p in expert.parameters()])
    optimizer.load_state_dict(saved)
    return optimizer, experts, fresh_places


@needs_a_card
@pytest.mark.xdist_group("gpu")
@pytest.mark.parametrize("key", EVERY_OPTIMIZER)
def test_restored_state_is_where_the_run_that_made_it_has_it(key):
    optimizer, experts, fresh_places = _restored(key)
    assert "cpu" in {place for places in _places(optimizer, experts[1]).values() for place in places}, (
        "the load left the other expert's state off the card, or there's nothing to show here"
    )

    place_optimizer_state(optimizer, experts[1].parameters(), CARD)

    assert [_places(optimizer, expert) for expert in experts] == fresh_places


# Each fails with or without a resume, and with or without this placement. See the marks' reasons.
_ALSO_FAILS_WITHOUT_RESTORED_PLACEMENT = {
    "Adafactor": (
        "any family's resume on Adafactor: the load casts its bf16 exp_avg to the parameter's "
        "float32, and its next step refuses the mix"
    ),
    "ProdigyPlusSF": (
        "a swap-mode run that never stopped, too: it keeps its running d on group['params'][0]'s "
        "device, which leaves the card with the first expert"
    ),
}


@needs_a_card
@pytest.mark.xdist_group("gpu")
@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            key,
            marks=[pytest.mark.xfail(reason=_ALSO_FAILS_WITHOUT_RESTORED_PLACEMENT[key], raises=RuntimeError, strict=True)]
            if key in _ALSO_FAILS_WITHOUT_RESTORED_PLACEMENT
            else [],
        )
        for key in EVERY_OPTIMIZER
    ],
)
def test_restored_state_steps_across_switches(key):
    optimizer, experts, _ = _restored(key)
    place_optimizer_state(optimizer, experts[1].parameters(), CARD)
    active = 0
    for step in range(6):
        _step(optimizer, experts[active])
        if step % 2 == 1:
            experts[active].to("cpu")
            active = 1 - active
            experts[active].to(CARD)
