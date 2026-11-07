"""A dual-expert resume continues the router and gives each expert its own weights.

Everything runs through the real ``_resume_if_needed`` and the real ``CheckpointManager``, on a bare
trainer around a real driver, the router ``_build_router`` builds and two small PEFT experts. A
"run" is the part of the loop that matters here: the expert of a step is whatever the driver has
active when the step starts, and ``on_optimizer_step(step)`` follows the step. The resumed run starts
where ``train()`` starts it, and ``max_train_steps`` is set when it starts, as ``train()`` sets it.

The helpers here also serve ``test_bernini_r_router_resume.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from app.engine.components.checkpoints import CheckpointManager, checkpoint_resumable
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import HIGH, LOW
from app.engine.models.families.wan22.trainer import Wan22Trainer

CPU = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")
MAX_STEPS = 100
ROUTER_FILE = "router_state.pt"
SAVED_VALUE = {HIGH: 3.0, LOW: 7.0}


class _Wan22Defn:
    id = "wan22_t2v"
    architecture_params = {"mode": "t2v", "te.max_length": 512, "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []
    defaults: dict = {"num_frames": 81}


@dataclass(frozen=True)
class Family:
    trainer: type
    driver: type
    definition: type


WAN22 = Family(Wan22Trainer, Wan22Driver, _Wan22Defn)


class RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, event, **kw):
        pass

    def debug(self, event, **kw):
        pass

    def error(self, event, **kw):
        pass

    def named(self, event: str) -> list[dict]:
        return [kw for ev, kw in self.warnings if ev == event]


def _tiny_expert(fill: float) -> PeftModel:
    pm = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
    set_lora(pm, fill)
    return pm


def set_lora(model: PeftModel, fill: float) -> None:
    with torch.no_grad():
        model.base_model.model[0].lora_B["default"].weight.fill_(fill)


def lora_value(model: PeftModel) -> float:
    weight = model.base_model.model[0].lora_B["default"].weight
    assert weight.min() == weight.max()
    return float(weight[0, 0].detach())


def make_trainer(family: Family, config: dict | None = None, *, out: Path, expert_mode: str = "both") -> Any:
    """A trainer as ``setup`` leaves it before the resume: built, experts wrapped, nothing trained."""
    t = object.__new__(family.trainer)
    t.device = CPU
    t.config = {
        "timestep_sampling": "uniform",
        "expert_switch_interval": 1,
        "expert_swap_mode": "resident",
        "expert_mode": expert_mode,
        "max_train_steps": MAX_STEPS,
        "lora_name": "t",
        "resolutions": [512],
        "seed": 7,
        **(config or {}),
    }
    t.definition = family.definition()
    t.driver = family.driver(t.definition, CPU)
    t.driver.configure_expert_mode(expert_mode)
    t.expert_mode = expert_mode
    t.global_step = 0
    t.logger = RecordingLogger()
    t.components = {}
    t.text_cache = {}
    t.optimizer = t.lr_scheduler = t.scaler = t.ema_handler = None
    t.logger_component = SimpleNamespace(elapsed_offset=0.0, step_offset=0)
    t.checkpoint_manager = CheckpointManager(str(out))
    t._build_router()
    t.driver.transformer_high = _tiny_expert(0.0) if expert_mode != LOW else None
    t.driver.transformer_low = _tiny_expert(0.0) if expert_mode != HIGH else None
    t.driver._set_active(t.driver.active_expert)
    return t


def run_steps(t: Any, start: int, stop: int) -> dict[int, str]:
    t.max_train_steps = int(t.config["max_train_steps"])
    expert_of: dict[int, str] = {}
    for step in range(start, stop):
        t.global_step = step
        expert_of[step] = t.driver.active_expert
        t.driver.on_optimizer_step(step)
    return expert_of


def save(t: Any, step: int) -> Path:
    folder = t.checkpoint_manager.save_checkpoint(
        step=step, components=t._build_trainable_components(), config=dict(t.config)
    )
    return Path(folder)


def resume(t: Any, folder: Path) -> int:
    """``_resume_if_needed``, and the step ``train()`` then starts from."""
    t.config["resume_from_checkpoint"] = str(folder)
    t._resume_if_needed()
    return t.global_step + 1 if t.global_step > 0 else 0


def strip_file(folder: Path, name: str) -> None:
    """The folder as it would be had the file never been written."""
    (folder / name).unlink(missing_ok=True)
    manifest_path = folder / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop(name, None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def seed_starting_on(family: Family, expert: str, tmp_path: Path) -> int:
    for seed in range(400):
        t = make_trainer(family, {"seed": seed, "timestep_sampling": "uniform"}, out=tmp_path / "probe")
        if t.driver.active_expert == expert:
            return seed
    raise AssertionError(f"no seed starts on {expert}")


def step_where_next_differs_from_first(expert_of: dict[int, str]) -> int:
    """A step whose next expert isn't block 0's, so a resume left on block 0's is seen."""
    for step in range(2, len(expert_of) - 1):
        if expert_of[step + 1] != expert_of[0]:
            return step
    raise AssertionError(f"one expert all the way: {expert_of}")


def router_as_built_then_hook(family: Family, config: dict, step: int, out: Path, **kw) -> tuple[torch.Tensor, dict]:
    """What a newly built router holds once ``on_optimizer_step(step)`` has run on it."""
    twin = make_trainer(family, config, out=out, **kw)
    twin.global_step = step
    twin.max_train_steps = MAX_STEPS
    twin.driver.on_optimizer_step(step)
    return twin.expert_router._step_rng.get_state(), dict(twin.expert_router._block_decisions)


def assert_router_equals(router: Any, built: tuple[torch.Tensor, dict]) -> None:
    assert torch.equal(router._step_rng.get_state(), built[0])
    assert router._block_decisions == built[1]


# ── The checks both families run ──


def through_assignment(t: Any) -> Any:
    """Then the load's real ``assign_components``, which makes HIGH active whatever the router chose."""
    high, low = t.driver.transformer_high, t.driver.transformer_low
    t.driver.assign_components({"unet": high, "unet_low": low})
    return t


def controller_model(t: Any) -> Any:
    """The model ``_configure_adaptive_targeting`` hands the controller, as it would at setup."""
    from unittest.mock import MagicMock, patch

    seen: dict[str, Any] = {}

    class _Controller:
        enabled = False

        def __init__(self, model, **_kw):
            seen["model"] = model

    t.config["adaptive_targeting"] = True
    t._log_writer = MagicMock()
    with patch("app.engine.components.adaptive_targeting.AdaptiveTargetingController", _Controller):
        t._configure_adaptive_targeting()
    return seen["model"]


def check_fresh_run_starts_on_the_routers_expert(family: Family, tmp_path: Path, *, startup: str) -> None:
    config = {"seed": seed_starting_on(family, startup, tmp_path), "timestep_sampling": "uniform"}
    uninterrupted = run_steps(make_trainer(family, config, out=tmp_path / "ref"), 0, MAX_STEPS)
    assert uninterrupted[0] == startup

    fresh = through_assignment(make_trainer(family, config, out=tmp_path / "run"))
    fresh._resume_if_needed()  # no checkpoint: the fresh start

    assert fresh.driver.active_expert == startup
    assert fresh.driver.get_primary_model() is fresh.driver._expert_model(startup)
    assert run_steps(fresh, 0, MAX_STEPS) == uninterrupted


def check_adaptive_targeting_keeps_its_expert_across_a_resume(family: Family, tmp_path: Path) -> None:
    """Fresh and resumed, the controller is on the expert the load made active (HIGH)."""
    config = {"seed": seed_starting_on(family, LOW, tmp_path), "timestep_sampling": "uniform"}
    fresh = through_assignment(make_trainer(family, config, out=tmp_path / "run"))
    fresh._resume_if_needed()
    assert controller_model(fresh) is fresh.driver.transformer_high

    step = step_where_next_differs_from_first(run_steps(fresh, 0, MAX_STEPS))
    first = through_assignment(make_trainer(family, config, out=tmp_path / "run"))
    first._resume_if_needed()
    run_steps(first, 0, step + 1)
    folder = save(first, step)

    resumed = through_assignment(make_trainer(family, config, out=tmp_path / "run"))
    resume(resumed, folder)
    assert controller_model(resumed) is resumed.driver.transformer_high


BAD_AND_ODD_SEEDS =["abc", "12.5", [1], {"a": 1}, None, "", 12.7, "7", 7]


def check_router_reads_the_run_seed(family: Family, tmp_path: Path, seed: Any) -> None:
    """The router is built on the seed the run itself applies, and 0 when it applies none."""
    from app.engine.components.checkpoints import _run_seed

    t = make_trainer(family, {"seed": seed}, out=tmp_path / "run")

    assert t.expert_router._seed == (_run_seed({"seed": seed}) or 0)


def check_decisions_continue(
    family: Family, tmp_path: Path, *, mode: str, swap: str, interval: int, soft_stop: bool
) -> None:
    config = {"timestep_sampling": mode, "expert_swap_mode": swap, "expert_switch_interval": interval}
    uninterrupted = run_steps(make_trainer(family, config, out=tmp_path / "ref"), 0, MAX_STEPS)
    step = step_where_next_differs_from_first(uninterrupted)

    first = make_trainer(family, config, out=tmp_path / "run")
    if soft_stop:
        # The signal check at the top of ``step``: that step hasn't run, so the stop
        # records the last one that did, and the resume runs ``step`` itself.
        run_steps(first, 0, step)
        first.global_step = step
        folder = save(first, step - 1)
        expected_start = step
    else:
        run_steps(first, 0, step + 1)
        folder = save(first, step)
        expected_start = step + 1

    resumed = make_trainer(family, config, out=tmp_path / "run")
    start = resume(resumed, folder)

    assert start == expected_start
    assert resumed.driver.active_expert == uninterrupted[start], "the first step after the resume"
    active = resumed.driver.active_expert
    assert resumed.driver.get_primary_model() is resumed.driver._expert_model(active)
    assert run_steps(resumed, start, MAX_STEPS) == {s: uninterrupted[s] for s in range(start, MAX_STEPS)}
    assert resumed.logger.warnings == []


def record_expert_moves(driver: Any) -> dict[str, str]:
    """``driver.device`` becomes the card and each expert's ``.to`` is recorded: nothing moves."""
    driver.device = torch.device("cuda:0")
    placed: dict[str, str] = {}
    for name in (HIGH, LOW):
        model = driver._expert_model(name)
        if model is None:
            continue

        def _to(target, name=name, model=model):
            placed[name] = str(target)
            return model

        model.to = _to
    return placed


def check_right_expert_is_resident(family: Family, tmp_path: Path, *, swap: str) -> None:
    config = {"expert_swap_mode": swap}
    uninterrupted = run_steps(make_trainer(family, config, out=tmp_path / "ref"), 0, MAX_STEPS)
    step = step_where_next_differs_from_first(uninterrupted)
    first = make_trainer(family, config, out=tmp_path / "run")
    run_steps(first, 0, step + 1)
    folder = save(first, step)

    resumed = make_trainer(family, config, out=tmp_path / "run")
    driver = resumed.driver
    placed = record_expert_moves(driver)
    driver.configure_swap_mode(swap)
    driver.place_experts_for_start()  # as _configure_optimization does, before the resume

    resume(resumed, folder)

    active = uninterrupted[step + 1]
    other = LOW if active == HIGH else HIGH
    assert driver.active_expert == active
    assert placed[active] == "cuda:0"
    assert placed[other] == ("cpu" if swap == "swap" else "cuda:0")


def check_own_weights(
    family: Family, tmp_path: Path, *, startup: str, at_save: str, router_file: bool
) -> None:
    config = {"seed": seed_starting_on(family, startup, tmp_path), "timestep_sampling": "uniform"}
    first = make_trainer(family, config, out=tmp_path / "run")
    assert first.driver.active_expert == startup
    set_lora(first.driver.transformer_high, SAVED_VALUE[HIGH])
    set_lora(first.driver.transformer_low, SAVED_VALUE[LOW])
    step = 0
    run_steps(first, step, 1)
    while first.driver.active_expert != at_save:
        step += 1
        run_steps(first, step, step + 1)
    folder = save(first, max(step, 1))
    if not router_file:
        strip_file(folder, ROUTER_FILE)

    resumed = make_trainer(family, config, out=tmp_path / "run")
    assert resumed.driver.active_expert == startup
    resume(resumed, folder)

    assert lora_value(resumed.driver.transformer_high) == SAVED_VALUE[HIGH]
    assert lora_value(resumed.driver.transformer_low) == SAVED_VALUE[LOW]
    assert resumed.logger.named("resume_partial_adapter_restore") == []


def check_save_writes_the_router_and_readers_accept_it(family: Family, tmp_path: Path) -> None:
    t = make_trainer(family, out=tmp_path / "run")
    comps = t._build_trainable_components()
    assert comps["router_state"] is t.expert_router

    # _name_optimizer_params, _resume_if_needed's loop and the initial log all walk these.
    params = [p for p in t.driver.transformer_high.parameters() if p.requires_grad]
    assert all(name.startswith("unet") for name in t._name_optimizer_params(params))
    t.checkpoint_manager._write_training_log(
        step=0, components=comps, optimizer=None, config=t.config, is_final=False,
        elapsed_time=0.0, lora_filename="(not yet saved)",
    )
    log = json.loads((tmp_path / "run" / "training_log.json").read_text(encoding="utf-8"))
    assert "router_state" not in log["parameter_counts"]

    # The real save, with the family's real saver.
    t.checkpoint_manager.saver = t.driver.get_saver()
    run_steps(t, 0, 3)
    folder = save(t, 2)
    assert t.checkpoint_manager.last_lora_path is not None

    assert (folder / ROUTER_FILE).is_file()
    manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][ROUTER_FILE] == (folder / ROUTER_FILE).stat().st_size
    assert checkpoint_resumable(folder) == (True, "")
    saved = torch.load(folder / ROUTER_FILE, map_location="cpu", weights_only=True)
    assert saved["block_decisions"] == t.expert_router._block_decisions


def check_missing_file_warns_once(family: Family, tmp_path: Path) -> None:
    first = make_trainer(family, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)
    strip_file(folder, ROUTER_FILE)

    resumed = make_trainer(family, out=tmp_path / "run")
    resume(resumed, folder)

    assert len(resumed.logger.named("router_state_missing_on_resume")) == 1
    built = router_as_built_then_hook(family, {}, 5, tmp_path / "twin")
    assert_router_equals(resumed.expert_router, built)


def check_mismatch_leaves_the_built_router(
    family: Family, tmp_path: Path, *, setting: str, changed: dict, expert_mode: str = "both"
) -> None:
    first = make_trainer(family, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)

    resumed = make_trainer(family, changed, out=tmp_path / "run", expert_mode=expert_mode)
    resume(resumed, folder)

    warned = resumed.logger.named("router_state_mismatch_on_resume")
    assert len(warned) == 1 and warned[0]["settings"] == [setting]
    built = router_as_built_then_hook(family, changed, 5, tmp_path / "twin", expert_mode=expert_mode)
    assert_router_equals(resumed.expert_router, built)


def check_today_layout_resumes(family: Family, tmp_path: Path) -> None:
    """Written by a process that passes the router's plain dict and saves no streams."""
    first = make_trainer(family, out=tmp_path / "run")
    set_lora(first.driver.transformer_high, SAVED_VALUE[HIGH])
    set_lora(first.driver.transformer_low, SAVED_VALUE[LOW])
    comps = first._build_trainable_components()
    comps["router_state"] = first.expert_router.state_dict()
    folder = Path(first.checkpoint_manager.save_checkpoint(step=5, components=comps, config=dict(first.config)))
    strip_file(folder, "rng_state.pt")
    assert not (folder / ROUTER_FILE).exists()
    assert checkpoint_resumable(folder) == (True, "")

    resumed = make_trainer(family, out=tmp_path / "run")
    resume(resumed, folder)

    assert lora_value(resumed.driver.transformer_high) == SAVED_VALUE[HIGH]
    assert lora_value(resumed.driver.transformer_low) == SAVED_VALUE[LOW]
    assert resumed.global_step == 5
    assert len(resumed.logger.named("router_state_missing_on_resume")) == 1


def check_only_a_unet_folder_loads_into_the_active_expert(family: Family, tmp_path: Path, *, startup: str) -> None:
    config = {"seed": seed_starting_on(family, startup, tmp_path), "timestep_sampling": "uniform"}
    only = _tiny_expert(SAVED_VALUE[HIGH])
    folder = Path(
        CheckpointManager(str(tmp_path / "run")).save_checkpoint(step=5, components={"unet": only}, config=dict(config))
    )

    resumed = make_trainer(family, config, out=tmp_path / "run")
    active_before = resumed.driver.get_primary_model()
    other = resumed.driver._expert_model(LOW if startup == HIGH else HIGH)
    resume(resumed, folder)

    assert lora_value(active_before) == SAVED_VALUE[HIGH]
    assert lora_value(other) == 0.0
    missing = resumed.logger.named("resume_partial_adapter_restore")
    assert len(missing) == 1 and missing[0]["missing"] == [f"unet_{LOW if startup == HIGH else HIGH}"]


# ── The optimizer's state, on the card and off it ──


def resume_point(expert_of: dict[int, str]) -> tuple[int, int]:
    """``(save, stop)``: both experts have stepped by ``save``, ``save + 1`` isn't block 0's expert,
    and the steps up to ``stop`` cross at least two switches."""
    first = expert_of[0]
    for save_step in range(1, MAX_STEPS - 1):
        if first_other_expert_ran(expert_of, save_step) and expert_of[save_step + 1] != first:
            switches = 0
            for step in range(save_step + 2, MAX_STEPS):
                switches += expert_of[step] != expert_of[step - 1]
                if switches == 2:
                    return save_step, step + 1
    raise AssertionError(f"no such point: {expert_of}")


def first_other_expert_ran(expert_of: dict[int, str], upto: int) -> bool:
    return len({expert_of[s] for s in range(upto + 1)}) == 2


def make_training_trainer(family: Family, config: dict, *, out: Path) -> Any:
    """``make_trainer`` on the card, with the real optimizer, scheduler, scaler and start placement.

    The seed makes two calls build the same frozen base layers: a checkpoint holds only the adapters.
    """
    torch.manual_seed(1234)
    t = make_trainer(
        family,
        {"optimizer_type": "AdamW", "learning_rate": 1e-2, "mixed_precision": "no", "lr_scheduler": "constant", **config},
        out=out,
    )
    t.device = t.driver.device = CUDA_DEVICE
    set_lora(t.driver.transformer_high, 0.3)
    set_lora(t.driver.transformer_low, 0.7)
    t._configure_optimization(MAX_STEPS)
    return t


def train_steps(t: Any, start: int, stop: int) -> list[str]:
    """Forward, backward, optimizer and hook, in the loop's order. The experts differ in their weights."""
    t.max_train_steps = MAX_STEPS
    experts: list[str] = []
    for step in range(start, stop):
        t.global_step = step
        experts.append(t.driver.active_expert)
        model = t.driver.get_primary_model()
        model(torch.ones(2, 4, device=CUDA_DEVICE)).pow(2).mean().backward()
        t.optimizer.step()
        t.lr_scheduler.step()
        t.optimizer.zero_grad(set_to_none=True)
        t.driver.on_optimizer_step(step)
    return experts


def expert_params(t: Any, expert: str) -> list[torch.nn.Parameter]:
    return [p for p in t.driver._expert_model(expert).parameters() if p.requires_grad]


def optimizer_places(t: Any) -> dict[str, dict[str, set[str]]]:
    """Each expert's state: key to the device types its tensors are on."""
    places: dict[str, dict[str, set[str]]] = {}
    for expert in (HIGH, LOW):
        places[expert] = {}
        for param in expert_params(t, expert):
            for key, value in t.optimizer.state[param].items():
                places[expert].setdefault(key, set()).add(value.device.type)
    return places


def optimizer_values(t: Any) -> dict[str, list[dict[str, torch.Tensor]]]:
    return {
        expert: [
            {key: value.detach().clone().cpu() for key, value in t.optimizer.state[param].items()}
            for param in expert_params(t, expert)
        ]
        for expert in (HIGH, LOW)
    }


def weights(t: Any, expert: str) -> list[torch.Tensor]:
    return [p.detach().clone().cpu() for p in expert_params(t, expert)]


def check_real_steps_across_switches(family: Family, tmp_path: Path, *, swap: str, startup: str) -> None:
    """A resumed run steps across two switches on its restored moments, as the uninterrupted run does."""
    config = {"seed": seed_starting_on(family, startup, tmp_path), "expert_swap_mode": swap}
    save_step, stop = resume_point(run_steps(make_trainer(family, config, out=tmp_path / "plan"), 0, MAX_STEPS))

    first = make_training_trainer(family, config, out=tmp_path / "run")
    assert first.driver.active_expert == startup
    train_steps(first, 0, save_step + 1)
    saved_places = optimizer_places(first)
    saved_values = optimizer_values(first)
    folder = Path(
        first.checkpoint_manager.save_checkpoint(
            step=save_step,
            components=first._build_trainable_components(),
            optimizer=first.optimizer,
            scheduler=first.lr_scheduler,
            scaler=first.scaler,
            config=dict(first.config),
            optimizer_param_names=first._optimizer_param_names,
        )
    )
    expected_experts = train_steps(first, save_step + 1, stop)
    switches = sum(a != b for a, b in zip(expected_experts, expected_experts[1:]))
    assert switches >= 2

    resumed = make_training_trainer(family, config, out=tmp_path / "run")
    assert resumed.driver.active_expert == startup
    assert resume(resumed, folder) == save_step + 1
    assert resumed.driver.active_expert != startup, "the first step needs the other expert"

    assert optimizer_places(resumed) == saved_places, "where the uninterrupted run has its state"
    assert saved_places[startup]["step"] == {"cpu"} and saved_places[startup]["exp_avg"] == {"cuda"}
    restored = optimizer_values(resumed)
    for expert in (HIGH, LOW):
        for got, saved in zip(restored[expert], saved_values[expert]):
            assert got.keys() == saved.keys()
            assert all(torch.equal(got[key], saved[key]) for key in got), "restored, not begun again"

    assert train_steps(resumed, save_step + 1, stop) == expected_experts

    stepped = {expert: expected_experts.count(expert) for expert in (HIGH, LOW)}
    assert all(stepped.values()), expected_experts
    after = optimizer_values(resumed)
    for expert in (HIGH, LOW):
        for got, was in zip(after[expert], restored[expert]):
            assert got["step"] == was["step"] + stepped[expert]
            assert not torch.equal(got["exp_avg_sq"], was["exp_avg_sq"]), "this expert really stepped"
        # Begun-again moments would take other steps, so the weights would part from the reference.
        for got_weight, reference in zip(weights(resumed, expert), weights(first, expert)):
            assert torch.allclose(got_weight, reference, rtol=1e-5, atol=1e-7)
    assert optimizer_places(resumed) == saved_places


def check_placement_is_asked_for(
    family: Family, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, swap: str, managed: str | None, expected: set[str]
) -> None:
    """Without a card: which experts' state the resume sends to the training device."""
    import app.engine.core.pipeline.pipeline_optimization as pipeline_optimization

    config = {"expert_swap_mode": swap}
    first = make_trainer(family, config, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)

    asked: dict[str, tuple[Any, str, Any]] = {}
    resumed = make_trainer(family, config, out=tmp_path / "run")
    resumed.optimizer = torch.optim.AdamW(resumed._collect_expert_params(), lr=0.01)
    names = {HIGH: resumed.driver.transformer_high, LOW: resumed.driver.transformer_low}

    def _record(optimizer, params, device):
        params = list(params)
        expert = next(e for e, model in names.items() if params[0] is next(model.parameters()))
        asked[expert] = (optimizer, str(device), params)

    monkeypatch.setattr(pipeline_optimization, "place_optimizer_state", _record, raising=False)
    record_expert_moves(resumed.driver)
    resumed.driver.block_swap_active_expert = managed
    resume(resumed, folder)

    assert set(asked) == expected
    for expert, (optimizer, device, params) in asked.items():
        assert optimizer is resumed.optimizer
        assert device == "cuda:0"
        assert params == list(names[expert].parameters())


def check_single_expert_places_nothing(family: Family, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap mode needs two experts: a pinned job resolves it to resident."""
    import app.engine.core.pipeline.pipeline_optimization as pipeline_optimization

    calls: list[Any] = []
    monkeypatch.setattr(pipeline_optimization, "place_optimizer_state", lambda *a: calls.append(a), raising=False)
    config = {"expert_swap_mode": "swap"}
    first = make_trainer(family, config, out=tmp_path / "run", expert_mode=HIGH)
    run_steps(first, 0, 3)
    folder = save(first, 2)

    resumed = make_trainer(family, config, out=tmp_path / "run", expert_mode=HIGH)
    resumed.optimizer = torch.optim.AdamW(resumed._collect_expert_params(), lr=0.01)
    assert resumed.driver.swap_mode == "resident"
    resume(resumed, folder)

    assert calls == []


# ── Acceptance 6: the decisions ──


@pytest.mark.parametrize("swap", ["resident", "swap"])
@pytest.mark.parametrize(
    ("mode", "interval", "soft_stop"),
    [("uniform", 1, False), ("radc", 1, False), ("uniform", 3, False), ("radc", 1, True), ("radc", 3, True)],
    ids=["uniform", "radc", "interval3", "radc-soft-stop", "radc-interval3-soft-stop"],
)
def test_a_resumed_run_takes_the_decisions_of_an_uninterrupted_one(tmp_path, swap, mode, interval, soft_stop):
    check_decisions_continue(WAN22, tmp_path, mode=mode, swap=swap, interval=interval, soft_stop=soft_stop)


@pytest.mark.parametrize("swap", ["resident", "swap"])
def test_the_right_expert_is_active_and_resident_at_the_first_step(tmp_path, swap):
    check_right_expert_is_resident(WAN22, tmp_path, swap=swap)


def test_a_checkpoint_of_step_0_activates_the_expert_of_step_0(tmp_path):
    """``train()`` runs step 0 again after such a checkpoint, not step 1."""
    uninterrupted: dict[int, str] = {}
    for seed in range(50):
        config = {"seed": seed, "timestep_sampling": "radc"}
        uninterrupted = run_steps(make_trainer(WAN22, config, out=tmp_path / "ref"), 0, MAX_STEPS)
        if uninterrupted[0] != uninterrupted[1]:
            break
    assert uninterrupted[0] != uninterrupted[1]

    first = make_trainer(WAN22, config, out=tmp_path / "run")
    run_steps(first, 0, 1)
    folder = save(first, 0)
    resumed = make_trainer(WAN22, config, out=tmp_path / "run")
    start = resume(resumed, folder)

    assert start == 0
    assert run_steps(resumed, start, MAX_STEPS) == uninterrupted


def test_the_save_path_writes_the_router_and_its_readers_accept_it(tmp_path):
    check_save_writes_the_router_and_readers_accept_it(WAN22, tmp_path)


def test_a_checkpoint_without_the_file_warns_once(tmp_path):
    check_missing_file_warns_once(WAN22, tmp_path)


@pytest.mark.parametrize(
    ("setting", "changed", "expert_mode"),
    [
        ("seed", {"seed": 8}, "both"),
        ("switch_interval", {"expert_switch_interval": 3}, "both"),
        ("pinned_expert", {}, "high"),
    ],
    ids=["seed", "interval", "pinned"],
)
def test_a_mismatched_setting_leaves_the_built_router(tmp_path, setting, changed, expert_mode):
    check_mismatch_leaves_the_built_router(WAN22, tmp_path, setting=setting, changed=changed, expert_mode=expert_mode)


@pytest.mark.parametrize("value", ["42", 42], ids=["string", "int"])
def test_a_seed_read_the_same_way_is_unchanged(tmp_path, value):
    first = make_trainer(WAN22, {"seed": 42}, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)

    resumed = make_trainer(WAN22, {"seed": value}, out=tmp_path / "run")
    resume(resumed, folder)

    assert resumed.logger.named("router_state_mismatch_on_resume") == []
    assert 6 in resumed.expert_router._block_decisions
    assert resumed.expert_router._block_decisions == first.expert_router._block_decisions


def test_a_config_with_no_seed_key_is_unchanged(tmp_path):
    first = make_trainer(WAN22, {"seed": 42}, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)

    resumed = make_trainer(WAN22, out=tmp_path / "run")
    del resumed.config["seed"]
    resumed._build_router()
    resumed.driver._set_active(resumed.driver.active_expert)
    resume(resumed, folder)

    assert resumed.logger.named("router_state_mismatch_on_resume") == []
    assert resumed.expert_router._seed == 42
    assert resumed.expert_router._block_decisions == first.expert_router._block_decisions


@pytest.mark.parametrize("seed", BAD_AND_ODD_SEEDS, ids=repr)
def test_the_router_reads_the_seed_the_run_reads(tmp_path, seed):
    check_router_reads_the_run_seed(WAN22, tmp_path, seed)


@pytest.mark.parametrize("startup", [HIGH, LOW])
def test_a_fresh_run_starts_on_the_routers_expert(tmp_path, startup):
    check_fresh_run_starts_on_the_routers_expert(WAN22, tmp_path, startup=startup)


def test_adaptive_targeting_keeps_its_expert_across_a_resume(tmp_path):
    check_adaptive_targeting_keeps_its_expert_across_a_resume(WAN22, tmp_path)


@pytest.mark.parametrize("value", [None, "", "not-a-seed"], ids=["none", "empty", "invalid"])
def test_no_valid_seed_where_one_was_saved_is_changed(tmp_path, value):
    first = make_trainer(WAN22, {"seed": 42}, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)

    resumed = make_trainer(WAN22, {"seed": 0}, out=tmp_path / "run")
    resumed.config["seed"] = value  # the job's router is seed 0, as for any seed the run ignores
    resume(resumed, folder)

    warned = resumed.logger.named("router_state_mismatch_on_resume")
    assert len(warned) == 1 and warned[0]["settings"] == ["seed"]
    assert resumed.expert_router._seed == 0


@pytest.mark.parametrize("value", [None, ""], ids=["none", "empty"])
def test_an_unseeded_job_continues_its_router(tmp_path, value):
    """An unseeded router is built on seed 0, so the file's 0 is this same router."""
    first = make_trainer(WAN22, {"seed": value}, out=tmp_path / "run")
    uninterrupted = run_steps(make_trainer(WAN22, {"seed": value}, out=tmp_path / "ref"), 0, MAX_STEPS)
    run_steps(first, 0, 6)
    folder = save(first, 5)

    resumed = make_trainer(WAN22, {"seed": value}, out=tmp_path / "run")
    start = resume(resumed, folder)

    assert resumed.logger.warnings == []
    assert run_steps(resumed, start, MAX_STEPS) == {s: uninterrupted[s] for s in range(start, MAX_STEPS)}


def test_an_unreadable_router_state_warns_and_leaves_the_built_router(tmp_path):
    first = make_trainer(WAN22, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)
    path = folder / ROUTER_FILE
    torch.save({"seed": 7, "pinned_expert": None, "switch_interval": 1, "rng_state": torch.zeros(3)}, path)
    manifest_path = folder / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][ROUTER_FILE] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = make_trainer(WAN22, out=tmp_path / "run")
    resume(resumed, folder)

    assert len(resumed.logger.named("router_state_unreadable_on_resume")) == 1
    assert_router_equals(resumed.expert_router, router_as_built_then_hook(WAN22, {}, 5, tmp_path / "twin"))


def test_a_changed_distribution_keeps_the_estimate_of_the_new_config(tmp_path):
    """The saved ``p_high`` belongs to the old settings, and only ``radc`` would ever refresh it."""
    first = make_trainer(WAN22, {"timestep_sampling": "logit_normal"}, out=tmp_path / "run")
    run_steps(first, 0, 6)
    folder = save(first, 5)
    saved = first.expert_router.p_high

    resumed = make_trainer(WAN22, {"timestep_sampling": "uniform"}, out=tmp_path / "run")
    built = resumed.expert_router.p_high
    assert abs(built - saved) > 0.01
    resume(resumed, folder)
    assert resumed.expert_router.p_high == built

    same = make_trainer(WAN22, {"timestep_sampling": "logit_normal"}, out=tmp_path / "run")
    resume(same, folder)
    assert same.expert_router.p_high == saved


def test_adaptive_targeting_stays_on_the_expert_it_was_built_on(tmp_path, monkeypatch):
    """The controller's universe is one expert: a resume doesn't hand it the other's."""
    import app.engine.components.adaptive_targeting as adaptive

    seen: list[Any] = []

    class _Controller:
        enabled = False

        def __init__(self, *, model, **kw):
            seen.append(model)

    monkeypatch.setattr(adaptive, "AdaptiveTargetingController", _Controller)

    uninterrupted = run_steps(make_trainer(WAN22, out=tmp_path / "ref"), 0, MAX_STEPS)
    step = step_where_next_differs_from_first(uninterrupted)
    first = make_trainer(WAN22, out=tmp_path / "run")
    run_steps(first, 0, step + 1)
    folder = save(first, step)

    resumed = make_trainer(WAN22, {"adaptive_targeting": True}, out=tmp_path / "run")
    resumed._log_writer = object()
    startup = resumed.driver.get_primary_model()
    resume(resumed, folder)
    assert resumed.driver.get_primary_model() is not startup

    resumed._configure_adaptive_targeting()

    assert seen == [startup]


# ── Acceptance 6a and 7: the weights ──


@pytest.mark.parametrize("router_file", [True, False], ids=["router-file", "no-router-file"])
@pytest.mark.parametrize("at_save", [HIGH, LOW])
@pytest.mark.parametrize("startup", [HIGH, LOW])
def test_each_expert_ends_with_its_own_weights(tmp_path, startup, at_save, router_file):
    check_own_weights(WAN22, tmp_path, startup=startup, at_save=at_save, router_file=router_file)


@pytest.mark.parametrize("startup", [HIGH, LOW])
def test_only_a_unet_folder_loads_into_the_active_expert(tmp_path, startup):
    check_only_a_unet_folder_loads_into_the_active_expert(WAN22, tmp_path, startup=startup)


def test_a_folder_in_todays_layout_resumes(tmp_path):
    check_today_layout_resumes(WAN22, tmp_path)


# ── The optimizer's state ──


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card to move anything")
@pytest.mark.parametrize("startup", [HIGH, LOW])
@pytest.mark.parametrize("swap", ["swap", "resident"])
def test_a_resumed_run_steps_across_switches_on_its_restored_moments(tmp_path, swap, startup):
    check_real_steps_across_switches(WAN22, tmp_path, swap=swap, startup=startup)


@pytest.mark.parametrize(
    ("swap", "managed", "expected"),
    [
        ("swap", None, {HIGH, LOW}),
        ("resident", None, set()),
        # A block-swapped expert's state is where each of its parameters steps, already.
        ("swap", HIGH, {LOW}),
        ("swap", LOW, {HIGH}),
    ],
    ids=["swap", "resident", "swap-high-block-swapped", "swap-low-block-swapped"],
)
def test_the_resume_places_the_state_of_the_experts_swap_mode_moves(tmp_path, monkeypatch, swap, managed, expected):
    check_placement_is_asked_for(WAN22, tmp_path, monkeypatch, swap=swap, managed=managed, expected=expected)


def test_a_single_expert_job_places_nothing(tmp_path, monkeypatch):
    check_single_expert_places_nothing(WAN22, tmp_path, monkeypatch)


def test_a_family_without_a_router_places_nothing(tmp_path, monkeypatch):
    """The whole of it hangs off the router, which only a dual-expert family builds."""
    import app.engine.core.pipeline.pipeline_optimization as pipeline_optimization

    calls: list[Any] = []
    monkeypatch.setattr(pipeline_optimization, "place_optimizer_state", lambda *a: calls.append(a), raising=False)
    t = make_trainer(WAN22, {"expert_swap_mode": "swap"}, out=tmp_path / "run")
    t.optimizer = torch.optim.AdamW(t._collect_expert_params(), lr=0.01)
    t.expert_router = None
    t._resume_expert_router(str(tmp_path))

    assert calls == []


def test_a_pinned_job_loads_its_own_expert_from_a_dual_checkpoint(tmp_path):
    """The saved active may be the other expert; the pinned job asks for its own by name."""
    config = {"seed": seed_starting_on(WAN22, HIGH, tmp_path), "timestep_sampling": "uniform"}
    first = make_trainer(WAN22, config, out=tmp_path / "run")
    set_lora(first.driver.transformer_high, SAVED_VALUE[HIGH])
    set_lora(first.driver.transformer_low, SAVED_VALUE[LOW])
    assert first.driver.active_expert == HIGH
    folder = save(first, 5)

    pinned = make_trainer(WAN22, config, out=tmp_path / "run", expert_mode=LOW)
    resume(pinned, folder)

    assert lora_value(pinned.driver.transformer_low) == SAVED_VALUE[LOW]
