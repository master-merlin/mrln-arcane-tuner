"""Integration tests: the capability allowlist wired at the job_manager choke
points (create_job AND update_job_config), alongside _apply_video_contract.

Uses the real registry so resolve_capabilities can build field_visibility.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.core.job import JobStatus
from app.core.job_manager import JobManager
from app.engine.models.registry import registry


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


IMAGE = "flux1-dev"
VIDEO = "wan2.1-t2v-1.3b"


def _cfg(**overrides) -> dict:
    base = {"output_dir": "outputs", "lora_name": "t", "definition_id": IMAGE}
    base.update(overrides)
    return base


def test_create_drops_gated_key_for_family():
    """A key gated OFF for the target family is dropped at create."""
    mgr = JobManager()
    job = mgr.create_job(IMAGE, _cfg(num_frames=81))
    assert "num_frames" not in job.config
    assert job.status == JobStatus.PENDING


def test_create_keeps_supported_key_for_family():
    """The SAME key passes through for a family that supports it."""
    mgr = JobManager()
    job = mgr.create_job(VIDEO, _cfg(definition_id=VIDEO, num_frames=81))
    assert job.config["num_frames"] == 81


def test_update_drops_gated_key():
    """update_job_config applies the same allowlist as create."""
    mgr = JobManager()
    job = mgr.create_job(IMAGE, _cfg())
    with patch(
        "app.core.db.repositories.job_repo.JobHistoryRepository"
    ) as Repo:
        Repo.return_value.get_by_id.return_value = {"id": job.id}
        mgr.update_job_config(job.id, _cfg(num_frames=81))
    assert "num_frames" not in job.config


def test_allowlist_runs_after_video_contract_preserving_hard_reject():
    """train_audio=True on a non-audio model is HARD-rejected by the video
    contract (not silently dropped) — the allowlist must not pre-empt it."""
    mgr = JobManager()
    with pytest.raises(ValueError):
        mgr.create_job(IMAGE, _cfg(train_audio=True))


def test_unknown_id_is_a_noop():
    """An unresolvable definition_id skips the allowlist (fail-open), leaving
    the config untouched — mirrors _apply_video_contract."""
    mgr = JobManager()
    job = mgr.create_job("does-not-exist", _cfg(definition_id="does-not-exist",
                                                num_frames=81))
    assert job.config["num_frames"] == 81


def test_dropped_keys_logged_at_info():
    mgr = JobManager()
    with patch("app.core.job_manager.logger") as log:
        mgr.create_job(IMAGE, _cfg(num_frames=81, target_fps=24))
    calls = [c for c in log.info.call_args_list
             if c.args and c.args[0] == "capability_allowlist_dropped"]
    assert len(calls) == 1
    assert set(calls[0].kwargs["dropped"]) == {"num_frames", "target_fps"}


# ── LANE-168 verify 1.01: a STORED config is filtered at job start ──────────
#
# A job row saved before a field became capability-gated (min_snr_gamma went
# SDXL-only in LANE-168) still carries the key. start_job launches the stored
# config, so without a sweep at start the trainer receives it and the savers
# write ss_min_snr_gamma for a weighting the family never applied.


class _RecordingPlugin:
    """Stands in for the training plugin: records the config it is handed and
    spawns nothing (the returned object has no ``pid``, so no tailer/watchdog)."""

    def __init__(self) -> None:
        self.launched: list[dict] = []

    def start_training(self, config):
        self.launched.append(dict(config))
        return object()


def _stored_job(mgr: JobManager, definition_id: str, **extra):
    """A pending job as it comes back from the DB — NOT through create_job, so
    no allowlist has touched its config."""
    from app.core.job import Job

    job = Job.create(definition_id, _cfg(definition_id=definition_id, **extra))
    job.config["job_id"] = job.id
    mgr._jobs[job.id] = job
    return job


def _start(mgr: JobManager, job_id: str):
    plugin = _RecordingPlugin()
    with patch(
        "app.core.job_manager.plugin_manager.get_plugin", return_value=plugin
    ), patch("app.core.db.repositories.job_repo.JobHistoryRepository") as Repo:
        mgr.start_job(job_id, clear_stale_signal=False, preflight=False)
    persisted = [c.args[1] for c in Repo.return_value.update_config.call_args_list]
    return plugin, persisted


def test_start_job_filters_a_stored_config_before_launch():
    from app.engine.utils.lora_metadata import kohya_config_metadata

    mgr = JobManager()
    job = _stored_job(mgr, "flux2-dev", min_snr_gamma=7.0)
    assert job.config["min_snr_gamma"] == 7.0  # the precondition: a stale row

    plugin, persisted = _start(mgr, job.id)

    assert len(plugin.launched) == 1
    launched = plugin.launched[0]
    assert "min_snr_gamma" not in launched
    assert "min_snr_gamma" not in job.config
    # The filtered config is written back to the record, exactly once.
    assert len(persisted) == 1
    assert "min_snr_gamma" not in persisted[0]
    # The metadata built from what the trainer received claims no Min-SNR.
    assert "ss_min_snr_gamma" not in kohya_config_metadata(launched)


def test_start_job_keeps_min_snr_for_sdxl_and_persists_nothing():
    from app.engine.utils.lora_metadata import kohya_config_metadata

    mgr = JobManager()
    job = _stored_job(mgr, "sdxl_base_1.0", min_snr_gamma=7.0)

    plugin, persisted = _start(mgr, job.id)

    assert plugin.launched[0]["min_snr_gamma"] == 7.0
    assert persisted == []  # nothing dropped -> the record is not rewritten
    assert kohya_config_metadata(plugin.launched[0])["ss_min_snr_gamma"] == "7.0"


def test_start_job_logs_the_stored_drop_once_as_a_warning():
    mgr = JobManager()
    job = _stored_job(mgr, "flux2-dev", min_snr_gamma=7.0)
    with patch("app.core.job_manager.logger") as log:
        _start(mgr, job.id)
    calls = [c for c in log.warning.call_args_list
             if c.args and c.args[0] == "stored_config_capability_filtered"]
    assert len(calls) == 1
    assert calls[0].kwargs["dropped"] == ["min_snr_gamma"]
    assert calls[0].kwargs["job_id"] == job.id
    # ...and not a second, INFO line for the same drop.
    assert not [c for c in log.info.call_args_list
                if c.args and c.args[0] == "capability_allowlist_dropped"]
