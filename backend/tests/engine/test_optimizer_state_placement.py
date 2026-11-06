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


# ── A load gives back the state as it was saved ──


def _through_a_file(state_dict: dict[str, Any]) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def _stepped(key: str, model: nn.Module, steps: int = 2) -> torch.optim.Optimizer:
    optimizer = OptimizerFactory.create(key, model.parameters(), lr=1e-3)
    for _ in range(steps):
        model(torch.ones(2, 4)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _tensors(entry: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in entry.items() if torch.is_tensor(value)}


def _loaded_beside_saved(key: str) -> tuple[torch.optim.Optimizer, nn.Module, list[dict[str, torch.Tensor]]]:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    saved = _through_a_file(_stepped(key, model).state_dict())
    expected = [_tensors(saved["state"][index]) for index in range(len(list(model.parameters())))]
    fresh = OptimizerFactory.create(key, model.parameters(), lr=1e-3)
    fresh.load_state_dict(saved)
    return fresh, model, expected


def _as_loaded(optimizer: torch.optim.Optimizer, model: nn.Module) -> list[dict[str, torch.Tensor]]:
    return [_tensors(optimizer.state[param]) for param in model.parameters()]


def _same(left: list[dict[str, torch.Tensor]], right: list[dict[str, torch.Tensor]]) -> bool:
    return [sorted(entry) for entry in left] == [sorted(entry) for entry in right] and all(
        a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]) for a, b in zip(left, right) for key in a
    )


def test_a_load_gives_back_adafactors_state_as_saved_and_its_next_step_runs():
    optimizer, model, saved = _loaded_beside_saved("Adafactor")
    assert any(value.dtype == torch.bfloat16 for entry in saved for value in entry.values()), (
        "Adafactor keeps no bf16 state here, so this shows nothing"
    )

    assert _same(_as_loaded(optimizer, model), saved)

    model(torch.ones(2, 4)).pow(2).mean().backward()
    optimizer.step()


def test_a_load_gives_back_state_as_saved_where_the_params_dtype_is_narrower():
    param = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = OptimizerFactory.create("AdamW", [param], lr=1e-3)
    wide = torch.full((4,), 1 + 2**-12, dtype=torch.float32)
    assert not torch.equal(wide.to(torch.bfloat16).float(), wide), "bf16 holds this value, so this shows nothing"
    saved = {
        "state": {0: {"step": torch.tensor(1.0), "exp_avg": wide.clone(), "exp_avg_sq": wide.clone()}},
        "param_groups": optimizer.state_dict()["param_groups"],
    }

    optimizer.load_state_dict(saved)

    for key in ("exp_avg", "exp_avg_sq"):
        assert optimizer.state[param][key].dtype == torch.float32
        assert torch.equal(optimizer.state[param][key], wide)


def test_a_load_of_state_that_has_its_params_dtype_is_as_before():
    optimizer, model, saved = _loaded_beside_saved("AdamW")

    assert _same(_as_loaded(optimizer, model), saved)


# ── ProdigyPlusSF keeps its running d with the params that step ──
#
# A swapped-out expert's params are off the card while the other expert steps on it. A param on
# ``meta`` without a gradient stands in for one here, swapped in after the build, so the
# accumulators start on the CPU and never have to be copied off ``meta``.


def _cpu_param(with_grad: bool) -> nn.Parameter:
    param = nn.Parameter(torch.ones(2))
    if with_grad:
        param.grad = torch.ones(2)
    return param


def _prodigy(groups: list[list[nn.Parameter]], split: bool = True) -> torch.optim.Optimizer:
    return OptimizerFactory.create(
        "ProdigyPlusSF", [{"params": params} for params in groups], lr=1.0, config={"ppsf_split_groups": split}
    )


def _swapped_out(optimizer: torch.optim.Optimizer, group: int) -> None:
    optimizer.param_groups[group]["params"][0] = nn.Parameter(torch.zeros(2, device="meta"))


def _running_places(optimizer: torch.optim.Optimizer, group: dict[str, Any]) -> set[str]:
    return {value.device.type for value in optimizer.get_running_values_for_group(group)}


def test_running_values_of_one_group_follow_the_params_that_step():
    optimizer = _prodigy([[_cpu_param(False), _cpu_param(True)]])
    _swapped_out(optimizer, 0)

    assert _running_places(optimizer, optimizer.param_groups[0]) == {"cpu"}


def test_running_values_of_split_groups_follow_each_groups_own_params_that_step():
    optimizer = _prodigy([[_cpu_param(False), _cpu_param(True)], [_cpu_param(False), _cpu_param(True)]], split=True)
    _swapped_out(optimizer, 0)
    _swapped_out(optimizer, 1)

    for group in optimizer.param_groups:
        numerator, denom = optimizer.get_running_values_for_group(group)
        assert {numerator.device.type, denom.device.type} == {"cpu"}
        assert numerator is group["running_d_numerator"] and denom is group["running_d_denom"]


def test_shared_running_values_follow_a_later_groups_params_that_step():
    optimizer = _prodigy([[_cpu_param(False)], [_cpu_param(True)]], split=False)
    _swapped_out(optimizer, 0)

    numerator, denom = optimizer.get_running_values_for_group(optimizer.param_groups[1])

    assert {numerator.device.type, denom.device.type} == {"cpu"}
    first = optimizer.param_groups[0]
    assert numerator is first["running_d_numerator"] and denom is first["running_d_denom"]


def test_running_values_without_any_gradient_follow_the_first_param_as_before():
    optimizer = _prodigy([[_cpu_param(False), _cpu_param(False)]])
    _swapped_out(optimizer, 0)

    assert _running_places(optimizer, optimizer.param_groups[0]) == {"meta"}


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


@needs_a_card
@pytest.mark.xdist_group("gpu")
@pytest.mark.parametrize("key", EVERY_OPTIMIZER)
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
