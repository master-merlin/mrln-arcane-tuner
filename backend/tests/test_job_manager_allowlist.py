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
