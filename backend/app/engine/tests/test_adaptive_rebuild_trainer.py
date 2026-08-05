"""Rebuild action, trainer side: trigger, deferred emission, resume persistence.

``action: "rebuild"`` freezes in place exactly like ``freeze`` and, when the
active set has shrunk enough, additionally asks the train loop to save a
checkpoint and exit so the backend can relaunch the SAME job over only the kept
params (spec §5). The narrowing is of what the OPTIMIZER knows about — every
LoRA module stays instantiated and every checkpoint keeps them all.

Shared fixtures (``make_adaptive_controller``, ``make_peft_tiny``,
``train_step``, ``fake_log_writer``) live in this directory's ``conftest.py``.
"""

import inspect
import json
import os
from types import SimpleNamespace

import structlog
import torch

from app.engine.components.checkpoints import CheckpointManager
from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


def _rebuild_cfg(**overrides) -> dict:
    return {
        "action": "rebuild",
        "rebuild_min_shrink_pct": 20.0,
        "energy_threshold": 0.90,
        **overrides,
    }


def _adapt(writer) -> list[dict]:
    return [data for msg_type, data in writer.events if msg_type == "adapt"]


def _rebuild_events(writer) -> list[dict]:
    return [event for event in _adapt(writer) if event["kind"] == "rebuild_request"]


# ── controller: the rebuild trigger ──────────────────────────────────────


def test_rebuild_requested_when_shrink_threshold_met(
    make_adaptive_controller, train_step
):
    model, ctl, writer = make_adaptive_controller(**_rebuild_cfg())
    result = None
    for step in range(1, 21):  # the loop stops ON the event step
        train_step(model, ["blocks.0.to_q"])
        result = ctl.on_optimizer_step(step) or result
    assert result == "rebuild_request"
    assert ctl.active_count < 8  # it froze in place first, like freeze mode

    # Emission is DEFERRED: the event carries the directory the backend
    # relaunches from, and that checkpoint does not exist yet.
    assert not _rebuild_events(writer)

    ctl.notify_rebuild_checkpoint("checkpoint-000020")
    events = _rebuild_events(writer)
    assert len(events) == 1
    assert events[0]["checkpoint_dir"] == "checkpoint-000020"
    assert events[0]["keep_patterns"] == ctl.keep_patterns()
    assert events[0]["step"] == 20
    assert ctl.rebuild_count == 1


def test_no_rebuild_below_shrink_threshold(make_adaptive_controller, train_step):
    model, ctl, writer = make_adaptive_controller(
        **_rebuild_cfg(rebuild_min_shrink_pct=95.0, min_active_pct=0.5)
    )
    results = set()
    for step in range(1, 41):  # events at 20, 30, 40
        train_step(model, ["blocks.0.to_q"])
        results.add(ctl.on_optimizer_step(step))
    assert "rebuild_request" not in results
    assert not _rebuild_events(writer)


def test_rebuild_mode_with_no_shrink_behaves_exactly_like_freeze(
    make_adaptive_controller, train_step
):
    """A rebuild-mode run whose keep-set never shrinks must be
    indistinguishable from a freeze-mode one — same events, same flags, no
    restart. Every module keeps learning, so no event can narrow anything."""
    model, rebuild_ctl, rebuild_writer = make_adaptive_controller(
        **_rebuild_cfg(energy_threshold=1.0)
    )
    model2, freeze_ctl, freeze_writer = make_adaptive_controller(energy_threshold=1.0)
    every = [f"blocks.{i}.{proj}" for i in range(4) for proj in ("to_q", "to_v")]
    for step in range(1, 41):
        train_step(model, every)
        train_step(model2, every)
        assert rebuild_ctl.on_optimizer_step(step) is None
        freeze_ctl.on_optimizer_step(step)

    assert rebuild_ctl.active_count == freeze_ctl.active_count == 8
    assert [e["kind"] for e in _adapt(rebuild_writer)] == [
        e["kind"] for e in _adapt(freeze_writer)
    ]
    assert rebuild_ctl.rebuild_count == 0


def test_freeze_action_never_requests_a_rebuild(make_adaptive_controller, train_step):
    model, ctl, writer = make_adaptive_controller(energy_threshold=0.90)
    results = set()
    for step in range(1, 41):
        train_step(model, ["blocks.0.to_q"])
        results.add(ctl.on_optimizer_step(step))
    assert ctl.active_count < 8  # it did narrow — the check is not vacuous
    assert results == {None}
    assert not _rebuild_events(writer)


def test_rebuild_cap_stops_requests_but_run_continues(
    make_adaptive_controller, train_step
):
    """The 5-rebuild bound must never fail the run: freezing continues in
    place, only the optimizer-VRAM reclaim stops."""
    model, ctl, writer = make_adaptive_controller(**_rebuild_cfg())
    ctl.rebuild_count = 5
    results = set()
    for step in range(1, 41):
        train_step(model, ["blocks.0.to_q"])
        results.add(ctl.on_optimizer_step(step))

    assert results == {None}
    assert ctl.active_count < 8  # narrowing still happened
    assert ctl.enabled is True
    assert not _rebuild_events(writer)
    capped = [
        str(data)
        for msg_type, data in writer.events
        if msg_type == "log" and "rebuild" in str(data)
    ]
    assert len(capped) == 1  # said once, not once per event
    assert "5" in capped[0]


def test_second_rebuild_measures_shrink_from_the_last_rebuild(
    make_adaptive_controller, train_step
):
    """The baseline resets at every rebuild, so each restart has to earn its
    own shrink. Measuring from run start instead would fire a rebuild at every
    event once the first threshold was crossed."""
    model, ctl, _writer = make_adaptive_controller(
        **_rebuild_cfg(min_active_pct=0.13, heat_ema=0.0)
    )
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_v", "blocks.1.to_q", "blocks.3.to_v"])
        ctl.on_optimizer_step(step)
    assert ctl.rebuild_count == 1
    first_active = ctl.active_count

    # The SAME modules keep learning: nothing new can be frozen, so there is no
    # further shrink and no second request.
    for step in range(21, 31):
        train_step(model, ["blocks.0.to_v", "blocks.1.to_q", "blocks.3.to_v"])
        assert ctl.on_optimizer_step(step) is None
    assert ctl.active_count == first_active
    assert ctl.rebuild_count == 1


def test_notify_without_a_pending_request_emits_nothing(
    make_adaptive_controller, train_step
):
    """A stray notify must not fabricate a restart: the backend acts on this
    event by relaunching the job."""
    model, ctl, writer = make_adaptive_controller(**_rebuild_cfg())
    ctl.notify_rebuild_checkpoint("checkpoint-000020")
    assert not _rebuild_events(writer)

    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    ctl.notify_rebuild_checkpoint("checkpoint-000020")
    ctl.notify_rebuild_checkpoint("checkpoint-000020")  # replay of the same one
    assert len(_rebuild_events(writer)) == 1


def test_rebuild_state_round_trip_restores_counts_and_baseline(
    make_adaptive_controller, train_step
):
    model, ctl, _writer = make_adaptive_controller(**_rebuild_cfg())
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert state["rebuild_count"] == 1
    assert state["params_at_last_rebuild"] > 0

    model2, resumed, _w2 = make_adaptive_controller(**_rebuild_cfg())
    resumed.restore_state(state, current_step=20)
    assert resumed.rebuild_count == 1
    assert (
        resumed.get_state()["params_at_last_rebuild"]
        == (state["params_at_last_rebuild"])
    )
    # The restored baseline is the post-rebuild one, so the next event needs a
    # fresh shrink of its own rather than re-firing on the shrink already spent.
    for step in range(21, 31):
        train_step(model2, ["blocks.0.to_q"])
        assert resumed.on_optimizer_step(step) is None


# ── checkpoint persistence ───────────────────────────────────────────────


def _trainable(model):
    return [(f"unet.{n}", p) for n, p in model.named_parameters() if p.requires_grad]


def _stepped_optimizer(params):
    opt = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    return opt


def test_checkpoint_carries_adaptive_state_and_param_names(tmp_path, make_peft_tiny):
    model = make_peft_tiny(2)
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    named = _trainable(model)
    path = manager.save_checkpoint(
        step=20,
        components={"unet": model},
        optimizer=_stepped_optimizer([p for _, p in named]),
        config={"lora_name": "t"},
        adaptive_state={"active_modules": ["blocks.0.to_q"], "rebuild_count": 1},
        optimizer_param_names=[n for n, _ in named],
    )
    assert os.path.basename(path) == "checkpoint-000020"

    with open(os.path.join(path, "training_state.json"), encoding="utf-8") as fh:
        state = json.load(fh)
    assert state["adaptive_targeting"]["rebuild_count"] == 1
    assert state["adaptive_targeting"]["active_modules"] == ["blocks.0.to_q"]

    names_path = os.path.join(path, "optimizer_param_names.json")
    with open(names_path, encoding="utf-8") as fh:
        assert json.load(fh) == [n for n, _ in named]
    assert not os.path.exists(names_path + ".tmp")  # atomic: no debris

    with open(os.path.join(path, "checkpoint_manifest.json"), encoding="utf-8") as fh:
        assert fh.read().count("optimizer_param_names.json") == 1


def test_checkpoint_keeps_every_lora_module_when_the_set_is_narrowed(
    tmp_path, make_peft_tiny
):
    """The invariant the whole rebuild design rests on: a frozen module's
    learned delta still shapes the forward pass, so it must be IN the
    checkpoint. Saving only the trainable subset would silently drop learned
    weights from the resume and from the final LoRA."""
    model = make_peft_tiny(2)
    for name, param in model.named_parameters():
        if ".lora_" in name and "blocks.0." in name:
            param.requires_grad_(False)
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    path = manager.save_checkpoint(
        step=20,
        components={"unet": model},
        config={"lora_name": "t"},
        adaptive_state={"active_modules": ["blocks.1.to_q"]},
    )
    from safetensors.torch import load_file

    saved = load_file(os.path.join(path, "unet", "adapter_model.safetensors"))
    assert any("blocks.0.to_q" in key for key in saved)
    assert any("blocks.1.to_q" in key for key in saved)


def test_feature_off_save_writes_neither_the_key_nor_the_names_file(
    tmp_path, make_peft_tiny
):
    """Zero-footprint contract: a run without adaptive targeting produces the
    exact checkpoint it produced before this feature existed."""
    model = make_peft_tiny(2)
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    path = manager.save_checkpoint(
        step=20, components={"unet": model}, config={"lora_name": "t"}
    )
    with open(os.path.join(path, "training_state.json"), encoding="utf-8") as fh:
        assert "adaptive_targeting" not in json.load(fh)
    assert not os.path.exists(os.path.join(path, "optimizer_param_names.json"))


def test_resume_over_a_narrowed_param_set_remaps_momentum_by_name(
    tmp_path, make_peft_tiny
):
    model = make_peft_tiny(2)
    named = _trainable(model)
    optimizer = _stepped_optimizer([p for _, p in named])
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    path = manager.save_checkpoint(
        step=10,
        components={"unet": model},
        optimizer=optimizer,
        config={"lora_name": "t"},
        optimizer_param_names=[n for n, _ in named],
    )

    # The restart: only blocks.1 survives, so the optimizer is built over a
    # SUBSET whose positions no longer line up with the saved state.
    kept = [(n, p) for n, p in named if "blocks.1." in n]
    assert 0 < len(kept) < len(named)
    expected = {n: optimizer.state[p]["exp_avg"].clone() for n, p in kept}
    narrowed = torch.optim.AdamW([p for _, p in kept], lr=1e-3)
    manager.load_checkpoint(
        path, optimizer=narrowed, optimizer_param_names=[n for n, _ in kept]
    )
    for position, (name, param) in enumerate(kept):
        restored = narrowed.state_dict()["state"][position]["exp_avg"]
        assert torch.equal(restored, expected[name])  # not zeros, not a neighbour
        assert restored.abs().sum() > 0
        assert narrowed.param_groups[0]["params"][position] is param


def test_plain_resume_loads_the_optimizer_state_verbatim(tmp_path, make_peft_tiny):
    """An ordinary (non-rebuild) resume must take the untouched path: same
    names, no remap, every moment restored exactly."""
    model = make_peft_tiny(2)
    named = _trainable(model)
    optimizer = _stepped_optimizer([p for _, p in named])
    before = {n: optimizer.state[p]["exp_avg"].clone() for n, p in named}
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    path = manager.save_checkpoint(
        step=10,
        components={"unet": model},
        optimizer=optimizer,
        config={"lora_name": "t"},
        optimizer_param_names=[n for n, _ in named],
    )

    resumed = torch.optim.AdamW([p for _, p in named], lr=1e-3)
    manager.load_checkpoint(
        path, optimizer=resumed, optimizer_param_names=[n for n, _ in named]
    )
    for position, (name, _param) in enumerate(named):
        assert torch.equal(
            resumed.state_dict()["state"][position]["exp_avg"], before[name]
        )


def test_resume_from_a_pre_feature_checkpoint_still_loads(tmp_path, make_peft_tiny):
    """Checkpoints written before this feature carry no name list; they must
    load exactly as they always did rather than degrade to fresh moments."""
    model = make_peft_tiny(2)
    named = _trainable(model)
    optimizer = _stepped_optimizer([p for _, p in named])
    before = optimizer.state[named[0][1]]["exp_avg"].clone()
    manager = CheckpointManager(output_dir=str(tmp_path / "run"))
    path = manager.save_checkpoint(
        step=10,
        components={"unet": model},
        optimizer=optimizer,
        config={"lora_name": "t"},
    )
    assert not os.path.exists(os.path.join(path, "optimizer_param_names.json"))

    resumed = torch.optim.AdamW([p for _, p in named], lr=1e-3)
    manager.load_checkpoint(
        path, optimizer=resumed, optimizer_param_names=[n for n, _ in named]
    )
    assert torch.equal(resumed.state_dict()["state"][0]["exp_avg"], before)


# ── pipeline wiring ──────────────────────────────────────────────────────


class _OptHost(PipelineOptimizationMixin):
    """Exposes exactly what ``_configure_optimization`` reads from a trainer."""

    def __init__(self, model, config=None, text_encoders=None):
        self.config = {
            "optimizer_type": "AdamW",
            "learning_rate": 1e-4,
            **(config or {}),
        }
        self._model = model
        self._text_encoders = text_encoders or {}
        self.logger = structlog.get_logger("test")
        self.driver = SimpleNamespace(
            get_precision_spec=lambda *a, **k: SimpleNamespace(
                autocast_dtype=torch.float32, use_amp=False, grad_scaler_enabled=False
            )
        )

    def _get_primary_model(self):
        return self._model

    def _get_text_encoders(self):
        return self._text_encoders


def test_optimizer_param_names_are_index_aligned_with_the_optimizer(make_peft_tiny):
    """The remap is positional-by-name: a list that does not line up with the
    optimizer's own param order would restore each param's moments onto its
    neighbour."""
    model = make_peft_tiny(2)
    host = _OptHost(model)
    host._configure_optimization(100)

    names = host._optimizer_param_names
    params = host.optimizer.param_groups[0]["params"]
    assert len(names) == len(params) > 0
    by_name = dict(zip(names, params))
    for name, param in model.named_parameters():
        if param.requires_grad:
            # PEFT's wrapper prefix is normalized away — see the compile test.
            assert by_name[f"unet.{name.replace('base_model.model.', '')}"] is param


def test_optimizer_param_names_are_compile_state_independent(make_peft_tiny):
    """The names are matched in a LATER process. If one of the two compiles the
    model and the other does not, an un-normalized list matches nothing and
    every param silently restarts from zeroed moments."""
    model = make_peft_tiny(2)
    plain = _OptHost(model)
    plain._configure_optimization(100)

    class _CompiledShim(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    compiled_model = _CompiledShim(make_peft_tiny(2))
    compiled = _OptHost(compiled_model)
    compiled._configure_optimization(100)

    assert compiled._optimizer_param_names == plain._optimizer_param_names
    assert not any("_orig_mod" in n for n in compiled._optimizer_param_names)


def test_params_of_a_component_that_cannot_enumerate_fall_back_to_the_model(
    make_peft_tiny,
):
    """A family may hand the saver a proxy that exposes no ``named_parameters``
    (hidream_o1 does, to avoid a 35 GB dump). Its params must still be named
    from the primary model rather than filled with placeholders."""
    model = make_peft_tiny(2)

    class _ProxyHost(_OptHost):
        def _build_trainable_components(self):
            return {"unet": SimpleNamespace(modules=lambda: iter(()))}

    host = _ProxyHost(model)
    host._configure_optimization(100)
    assert host._optimizer_param_names
    assert not any(n.startswith("<unnamed>") for n in host._optimizer_param_names)


def test_param_names_follow_a_family_override_that_supplies_its_own_params(
    make_peft_tiny,
):
    """Dual-expert trainers patch ``primary.parameters`` so the base collects
    BOTH experts' params. Re-walking the model for names would name a different
    set than the optimizer actually holds — and the remap would then hand every
    expert's moments to the wrong tensor."""
    high, low = make_peft_tiny(2), make_peft_tiny(2)
    both = [p for p in high.parameters() if p.requires_grad]
    both += [p for p in low.parameters() if p.requires_grad]

    class _DualHost(_OptHost):
        def _build_trainable_components(self):
            return {"unet": high, "unet_low": low}

    host = _DualHost(high)
    high.parameters = lambda *a, **k: iter(both)  # mirrors Wan22Trainer
    host._configure_optimization(100)

    names = host._optimizer_param_names
    params = host.optimizer.param_groups[0]["params"]
    assert len(names) == len(params) == len(both)
    assert any(n.startswith("unet_low.") for n in names)
    for name, param in zip(names, params):
        assert not name.startswith("<")  # nothing fell back to a placeholder
        assert param is dict(zip(names, params))[name]


class _TrainHost(PipelineTrainMixin):
    """Exposes exactly what ``_save_rebuild_checkpoint`` reads from a trainer."""

    def __init__(self, tmp_path, model, controller=None, names=None):
        self.config = {"lora_name": "t"}
        self.checkpoint_manager = CheckpointManager(output_dir=str(tmp_path / "run"))
        self._model = model
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        self.lr_scheduler = None
        self.scaler = None
        self.ema_handler = None
        self.logger = structlog.get_logger("test")
        self.logger_component = SimpleNamespace(get_total_elapsed=lambda: 12.0)
        self.adaptive_controller = controller
        self._optimizer_param_names = names

    def _build_trainable_components(self):
        return {"unet": self._model}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None


def test_save_rebuild_checkpoint_returns_the_dir_and_persists_the_state(
    tmp_path, make_peft_tiny, make_adaptive_controller, train_step
):
    """End-to-end on the value Task 7 relaunches from: the directory name the
    ``adapt`` event advertises must be the directory that exists on disk, and
    it must hold the controller state the restart re-adopts."""
    model, ctl, _writer = make_adaptive_controller(**_rebuild_cfg())
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    host = _TrainHost(tmp_path, make_peft_tiny(2), controller=ctl, names=["unet.a"])
    path = host._save_rebuild_checkpoint(20)
    assert os.path.basename(path) == "checkpoint-000020"
    assert os.path.isdir(path)

    with open(os.path.join(path, "training_state.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["global_step"] == 20
    assert (
        saved["adaptive_targeting"]["active_modules"]
        == ctl.get_state()["active_modules"]
    )
    with open(os.path.join(path, "optimizer_param_names.json"), encoding="utf-8") as fh:
        assert json.load(fh) == ["unet.a"]


def test_save_rebuild_checkpoint_without_the_feature_is_a_plain_checkpoint(
    tmp_path, make_peft_tiny
):
    host = _TrainHost(tmp_path, make_peft_tiny(2))
    path = host._save_rebuild_checkpoint(5)
    with open(os.path.join(path, "training_state.json"), encoding="utf-8") as fh:
        assert "adaptive_targeting" not in json.load(fh)


# ── the wiring itself (a method nothing calls is not wired) ──────────────


def test_train_loop_acts_on_the_rebuild_request():
    src = inspect.getsource(PipelineTrainMixin.train)
    hook = src.index("adaptive_ctl.on_optimizer_step(step)")
    act = src.index('adaptive_action == "rebuild_request"')
    save = src.index("self._save_rebuild_checkpoint(step)")
    notify = src.index("notify_rebuild_checkpoint(")
    flag = src.index("self._rebuild_exit = True")
    # The event announces a checkpoint directory, so the checkpoint must exist
    # before it is emitted; the flag is what run_trainer and the block after
    # the loop read to skip the final save.
    assert hook < act < save < notify < flag
    assert "break" in src[flag : flag + 200]
    # Save AND announcement inside one try, with the exit flag outside it: if
    # the backend never learns about the rebuild, exiting would end the job
    # with no final LoRA and nothing to relaunch it.
    handler = src.index("except Exception", act)
    assert save < notify < handler < flag
    # Acted on AFTER the step's batch tensors are dropped: the save is a
    # VRAM-transient hot spot, and running it on top of a live step's
    # allocations is how a big model OOMs on the very save meant to save VRAM.
    assert src.index("            del (") < act
    # And before the periodic save, so a step that is both writes ONE checkpoint.
    assert act < src.index("save_every > 0 and step > 0")


def test_train_loop_skips_the_final_save_on_a_rebuild_exit():
    """A "final" LoRA and a completed job record would both be lies: the
    backend is about to relaunch this same job from the checkpoint."""
    src = inspect.getsource(PipelineTrainMixin.train)
    guard = src.rindex('getattr(self, "_rebuild_exit", False)')
    assert guard < src.rindex("is_final=True")
    assert guard < src.rindex("self._complete_job_history(")
    assert "return" in src[guard : src.rindex("is_final=True")]


def test_periodic_and_rebuild_saves_share_one_call():
    """One checkpoint-save invocation, so a normal periodic save keeps passing
    the adaptive state too and the two can never drift apart."""
    src = inspect.getsource(PipelineTrainMixin.train)
    periodic = src.index("save_every > 0 and step > 0")
    assert "self._save_rebuild_checkpoint(step)" in src[periodic : periodic + 900]


def test_run_trainer_consults_the_rebuild_flag():
    """The trainer entry point has to know a rebuild exit is a SUCCESS: it must
    read the flag after train() returns, and must not turn it into a failure
    exit that the backend would report as a crashed job."""
    path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "run_trainer.py",
    )
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    guard = src.index("_rebuild_exit")
    assert src.index("await trainer.train()") < guard
    assert "exit(1" not in src[guard : guard + 500]


class _AdaptiveHost(PipelineOptimizationMixin):
    """Exposes exactly what ``_configure_adaptive_targeting`` reads."""

    def __init__(self, config, model, output_dir, log_writer, global_step=0):
        self.config = config
        self._model = model
        self.logger = structlog.get_logger("test")
        self._log_writer = log_writer
        self.checkpoint_manager = SimpleNamespace(output_dir=str(output_dir))
        self.global_step = global_step

    def _get_primary_model(self):
        return self._model


def _write_resume_state(tmp_path, adaptive_state, step=500):
    ckpt = tmp_path / "checkpoint-000500"
    ckpt.mkdir()
    payload = {"global_step": step, "config": {}}
    if adaptive_state is not None:
        payload["adaptive_targeting"] = adaptive_state
    (ckpt / "training_state.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(ckpt)


def _adaptive_config(resume_dir=None, **sub):
    config = {
        "max_train_steps": 1000,
        "adaptive_targeting": True,
        "adaptive_targeting_config": {"interval_steps": 50, "probe_steps": 5, **sub},
    }
    if resume_dir:
        config["resume_from_checkpoint"] = resume_dir
    return config


def test_resume_re_adopts_the_narrowed_set_before_the_first_step(
    tmp_path, make_peft_tiny, fake_log_writer, make_adaptive_controller, train_step
):
    """Restored HERE and nowhere else: ``_resume_if_needed`` runs before the
    controller exists, so this is the only point at which both the checkpoint
    and a live controller are available. Without it every resume — and every
    rebuild restart — silently trains the modules the run had frozen."""
    source_model, ctl, _writer = make_adaptive_controller(**_rebuild_cfg())
    for step in range(1, 21):
        train_step(source_model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert 0 < len(state["active_modules"]) < 8
    assert state["rebuild_count"] == 1

    model = make_peft_tiny(4)
    host = _AdaptiveHost(
        _adaptive_config(_write_resume_state(tmp_path, state)),
        model,
        tmp_path,
        fake_log_writer,
        global_step=500,
    )
    host._configure_adaptive_targeting()

    restored = host.adaptive_controller
    assert restored is not None
    assert set(restored.get_state()["active_modules"]) == set(state["active_modules"])
    assert restored.rebuild_count == 1
    frozen = [
        name
        for name, param in model.named_parameters()
        if ".lora_" in name and not param.requires_grad
    ]
    assert frozen  # the freeze reached the real model, not just the bookkeeping
    for name in frozen:
        assert not any(module in name for module in state["active_modules"])
    # And the schedule is clamped to the resume point — a persisted next_event
    # of 30 would otherwise fire an event on step 501, over a one-step window.
    assert restored.get_state()["next_event"] >= 550


def test_resume_without_an_adaptive_section_starts_wide_open(
    tmp_path, make_peft_tiny, fake_log_writer
):
    host = _AdaptiveHost(
        _adaptive_config(_write_resume_state(tmp_path, None)),
        make_peft_tiny(4),
        tmp_path,
        fake_log_writer,
    )
    host._configure_adaptive_targeting()
    assert host.adaptive_controller.active_count == 8


def test_unreadable_resume_state_warns_and_keeps_training(
    tmp_path, make_peft_tiny, fake_log_writer
):
    """A corrupt state file must degrade to a fresh controller, never end a
    multi-hour job — and never in silence."""
    from structlog.testing import capture_logs

    ckpt = tmp_path / "checkpoint-000500"
    ckpt.mkdir()
    (ckpt / "training_state.json").write_text("{not json", encoding="utf-8")
    host = _AdaptiveHost(
        _adaptive_config(str(ckpt)), make_peft_tiny(4), tmp_path, fake_log_writer
    )
    with capture_logs() as logs:
        host._configure_adaptive_targeting()
    assert host.adaptive_controller is not None
    assert host.adaptive_controller.active_count == 8
    assert any(entry.get("event") == "adaptive_state_restore_failed" for entry in logs)


def test_resume_restore_runs_before_the_controller_is_published():
    """The train loop reads ``self.adaptive_controller``; publishing it before
    the restore would leave a window in which the loop could step a controller
    that still believes the whole universe is active."""
    src = inspect.getsource(PipelineOptimizationMixin._configure_adaptive_targeting)
    assert src.index("_restore_adaptive_state(") < src.index(
        "self.adaptive_controller = controller"
    )
