"""
Tests for the Job data model and JobStatus enum.
Covers: Pydantic serialization, factory method, status transitions.
"""

import time


from app.core.job import Job, JobStatus


# ── JobStatus Enum ──────────────────────────────────────────────────────


class TestJobStatus:
    """Tests for the JobStatus enum values."""

    def test_all_statuses_are_strings(self):
        """All JobStatus values should be plain strings."""
        for status in JobStatus:
            assert isinstance(status.value, str)

    def test_expected_statuses_exist(self):
        """Required lifecycle states should be defined."""
        expected = {"pending", "running", "paused", "completed", "failed", "stopped"}
        actual = {s.value for s in JobStatus}
        assert expected == actual

    def test_status_string_comparison(self):
        """JobStatus members should compare equal to their string values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"


# ── Job Model ───────────────────────────────────────────────────────────


class TestJobModel:
    """Tests for the Job Pydantic model."""

    def test_create_factory_sets_defaults(self):
        """Job.create should populate id, created_at, and PENDING status."""
        job = Job.create(plugin_id="std", config={"lr": 1e-4})

        assert job.plugin_id == "std"
        assert job.config == {"lr": 1e-4}
        assert job.status == JobStatus.PENDING
        assert job.id  # non-empty UUID string
        assert job.created_at > 0

    def test_create_generates_unique_ids(self):
        """Each call to Job.create should produce a unique id."""
        job1 = Job.create(plugin_id="std", config={})
        job2 = Job.create(plugin_id="std", config={})
        assert job1.id != job2.id

    def test_optional_fields_default_to_none(self):
        """started_at, finished_at, pid, error should default to None."""
        job = Job.create(plugin_id="std", config={})
        assert job.started_at is None
        assert job.finished_at is None
        assert job.pid is None
        assert job.error is None

    def test_logs_default_to_empty_list(self):
        """logs should default to an empty list."""
        job = Job.create(plugin_id="std", config={})
        assert job.logs == []

    def test_serialization_roundtrip(self):
        """model_dump / model_validate should roundtrip correctly."""
        job = Job.create(plugin_id="flux2", config={"rank": 16})
        data = job.model_dump()

        restored = Job.model_validate(data)

        assert restored.id == job.id
        assert restored.plugin_id == "flux2"
        assert restored.config["rank"] == 16
        assert restored.status == JobStatus.PENDING

    def test_json_serialization(self):
        """model_dump_json should produce valid JSON with status as string."""
        job = Job.create(plugin_id="std", config={})
        json_str = job.model_dump_json()
        assert '"pending"' in json_str

    def test_status_can_be_updated(self):
        """Job status field should be mutable for state transitions."""
        job = Job.create(plugin_id="std", config={})
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
