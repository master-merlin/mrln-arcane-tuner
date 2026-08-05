"""Pipeline glue for adaptive layer targeting: controller lifecycle and the
per-step curve metrics (spec §5, §6).

The host below exposes ONLY the attributes the real pipeline actually carries
(``_log_writer``, ``checkpoint_manager.output_dir``). A host mirroring invented
names would pass while production reads attributes that do not exist, so the
collaborator contracts it stands in for are pinned separately below, and the
call sites are pinned by source-scan guards — unit-testing the mixin method in
isolation cannot tell whether anything ever calls it.

Shared fixtures (``make_peft_tiny``, ``fake_log_writer``) live in this
directory's ``conftest.py``.
"""

import inspect
from types import SimpleNamespace

import structlog
import torch.nn as nn
from structlog.testing import capture_logs

from app.engine.components.checkpoints import CheckpointManager
from app.engine.components.job_log_writer import JobLogWriter
from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class _Host(PipelineOptimizationMixin):
    """Minimal stand-in exposing exactly what ``_configure_adaptive_targeting``
    reads from the real trainer — nothing more, so an attribute the pipeline
    does not have cannot be satisfied here."""

    def __init__(self, config, model, output_dir, log_writer):
        self.config = config
        self._model = model
        self.logger = structlog.get_logger("test")
        self._log_writer = log_writer
        # The real pipeline's output root lives on the checkpoint manager,
        # which _configure_managers builds before resume — and so before the
        # controller is created.
        self.checkpoint_manager = SimpleNamespace(output_dir=str(output_dir))

    def _get_primary_model(self):
        return self._model


def _on_config(**sub_config):
    return {
        "max_train_steps": 100,
        "adaptive_targeting": True,
        "adaptive_targeting_config": sub_config,
    }


def _warnings(writer):
    return [str(data) for msg_type, data in writer.events if msg_type == "warning"]


# ── controller creation gate ─────────────────────────────────────────────


def test_disabled_by_default(make_peft_tiny, tmp_path, fake_log_writer):
    host = _Host({"max_train_steps": 100}, make_peft_tiny(2), tmp_path, fake_log_writer)
    host._configure_adaptive_targeting()
    assert host.adaptive_controller is None


def test_enabled_creates_controller(make_peft_tiny, tmp_path, fake_log_writer):
    host = _Host(
        _on_config(interval_steps=50), make_peft_tiny(2), tmp_path, fake_log_writer
    )
    host._configure_adaptive_targeting()
    assert host.adaptive_controller is not None
    assert host.adaptive_controller.total_count == 4  # 2 blocks × (to_q, to_v)
    assert host.adaptive_controller.config.interval_steps == 50
    assert host.adaptive_controller.total_steps == 100
    assert host.adaptive_controller.output_dir == str(tmp_path)
    assert host.adaptive_controller.log_writer is fake_log_writer


def test_universe_excludes_layers_the_user_already_froze(
    make_peft_tiny, tmp_path, fake_log_writer
):
    """Ordering contract: the controller is built AFTER the manual
    ``targeted_layers`` freeze, so it narrows strictly within the user's
    selection. Built earlier, its universe would include modules the user
    turned off and a rebuild restart could re-enable them."""
    model = make_peft_tiny(2)
    for name, param in model.named_parameters():
        if ".lora_" in name and "blocks.0." in name:
            param.requires_grad_(False)
    host = _Host(_on_config(), model, tmp_path, fake_log_writer)
    host._configure_adaptive_targeting()
    assert host.adaptive_controller is not None
    assert host.adaptive_controller.total_count == 2  # only block 1 survived


def test_invalid_subconfig_disables_with_warning_not_crash(
    make_peft_tiny, tmp_path, fake_log_writer
):
    """A sub-config that slipped past creation-time validation must not kill a
    multi-hour run at prepare time (spec §7) — and must not vanish quietly."""
    host = _Host(
        _on_config(action="bogus"), make_peft_tiny(2), tmp_path, fake_log_writer
    )
    host._configure_adaptive_targeting()
    assert host.adaptive_controller is None
    assert any("adaptive_targeting" in w for w in _warnings(fake_log_writer))


def test_model_without_lora_disables_with_warning_not_crash(tmp_path, fake_log_writer):
    """The controller disables itself when there is nothing to manage; the
    pipeline must drop the reference rather than keep a dead controller that
    every later step would still consult."""
    host = _Host(
        _on_config(),
        nn.Sequential(nn.Linear(8, 8)),  # never PEFT-wrapped
        tmp_path,
        fake_log_writer,
    )
    host._configure_adaptive_targeting()
    assert host.adaptive_controller is None
    assert any("adaptive_targeting" in w for w in _warnings(fake_log_writer))


def test_missing_log_writer_disables_with_warning_not_crash(make_peft_tiny, tmp_path):
    """The job log is the controller's ONLY channel for its freeze decisions and
    its own failures. Without one it would steer the run invisibly, so the
    feature switches off — loudly, never by silently doing nothing."""
    with capture_logs() as logs:
        host = _Host(_on_config(), make_peft_tiny(2), tmp_path, None)
        host._configure_adaptive_targeting()
    assert host.adaptive_controller is None
    assert any(
        entry.get("event") == "adaptive_targeting_setup_failed"
        and entry.get("log_level") == "warning"
        for entry in logs
    )


# ── per-step curve metrics ───────────────────────────────────────────────


def test_step_extras_reports_counts_and_emits_total_once(
    make_peft_tiny, tmp_path, fake_log_writer
):
    host = _Host(_on_config(), make_peft_tiny(2), tmp_path, fake_log_writer)
    host._configure_adaptive_targeting()
    first = host._adaptive_step_extras()
    assert first == {"adaptive_active": 4, "adaptive_hot": 0, "adaptive_total": 4}
    second = host._adaptive_step_extras()
    # The denominator is emitted once per PROCESS: it never changes while the
    # process lives, so re-sending it every step is pure payload weight.
    assert "adaptive_total" not in second
    assert second == {"adaptive_active": 4, "adaptive_hot": 0}


def test_step_extras_empty_when_off(make_peft_tiny, tmp_path, fake_log_writer):
    """Zero-overhead contract: a feature-off run's step payload must be
    byte-identical to what it was before this feature existed."""
    host = _Host({"max_train_steps": 100}, make_peft_tiny(2), tmp_path, fake_log_writer)
    host._configure_adaptive_targeting()
    assert host._adaptive_step_extras() == {}


def test_step_extras_empty_before_prepare_ever_ran(
    make_peft_tiny, tmp_path, fake_log_writer
):
    """The train loop may run on a trainer whose prepare phase never set the
    attribute (partial harnesses, subclassed prepare) — the helper must degrade
    to "off" rather than raise mid-run."""
    host = _Host({"max_train_steps": 100}, make_peft_tiny(2), tmp_path, fake_log_writer)
    assert not hasattr(host, "adaptive_controller")
    assert host._adaptive_step_extras() == {}


# ── the wiring itself (a mixin method nothing calls is not wired) ────────


def test_prepare_for_training_creates_the_controller_after_resume():
    src = inspect.getsource(PipelineOptimizationMixin.prepare_for_training)
    assert "self._configure_adaptive_targeting()" in src
    # PEFT, the manual freeze, the checkpoint manager and resume must all have
    # happened first: the controller snapshots the FINAL trainable set and
    # writes its history into the checkpoint manager's output root.
    for earlier in (
        "self._apply_peft()",
        "self._configure_targeted_training()",
        "self._configure_managers(",
        "self._resume_if_needed()",
    ):
        assert src.index(earlier) < src.index("self._configure_adaptive_targeting()")


def test_train_loop_steps_the_controller_and_merges_the_extras():
    src = inspect.getsource(PipelineTrainMixin.train)
    # The counters must ride the SAME per-step payload as loss/lr so the UI can
    # chart the narrowing staircase against the loss curve: hooked after the
    # optimizer step, merged into the payload dict, emitted with it. rindex for
    # the emit — an earlier log_step serves the all-NaN-window skip path, which
    # never reaches the optimizer and so has no adaptive counters to report.
    hook = src.index("adaptive_ctl.on_optimizer_step(step)")
    payload_start = src.index("extra: dict[str, Any] = {}")
    merge = src.index("self._adaptive_step_extras()")
    emit = src.rindex("self.logger_component.log_step(")
    assert hook < payload_start < merge < emit


def test_setup_reads_the_pipelines_real_attribute_names():
    """Anti-fiction guard: the host above could be edited to match whatever
    production reads. These are the names the REAL collaborators expose (pinned
    by the test below)."""
    src = inspect.getsource(PipelineOptimizationMixin._configure_adaptive_targeting)
    assert '"_log_writer"' in src
    assert "self.checkpoint_manager.output_dir" in src


def test_real_collaborators_carry_the_attributes_the_host_stubs(tmp_path):
    manager = CheckpointManager(output_dir=str(tmp_path))
    assert manager.output_dir == str(tmp_path)
    writer = JobLogWriter(str(tmp_path))
    try:
        # The controller calls all three on the writer it is handed.
        for method in ("emit", "log", "warning"):
            assert callable(getattr(writer, method))
    finally:
        writer.close()
