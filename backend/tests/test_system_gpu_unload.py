"""The user-triggered "free the GPU" control: `/api/system/gpu/{loaded,unload}`.

Reported by the user during UAT round 4 (LANE-42): nothing in the UI could
release VRAM on demand, so a caption/mask/score model sat resident until the
user happened to switch models or restart the backend.

What these tests pin, and why each one exists:

* **the unload has a measured effect** — the services' active-model keys go
  `None` and every plugin's `unload()` actually ran. A test that asserted the
  route returns 200 would pass against a route that does nothing at all
  (CONVENTIONS "Tests": a flag is not verified until you have seen its effect).
  On a machine with a GPU the residency test goes further and measures
  `torch.cuda.memory_allocated()` falling across the call.
* **the negative** — a service whose batch task is running is NOT unloaded.
  This is the failure the guard exists to prevent: freeing a model out from
  under a running batch crashes it mid-dataset.
* **reading is free** — `GET /gpu/loaded` must not construct a singleton (and
  therefore cannot load a model in order to answer whether one is loaded).
* **the wire shape** — the ids and fields are frozen public surface
  (ARCHITECTURE D2, reserved in ECOSYSTEM §6), so they are asserted literally.

The plugin stand-ins are `SimpleNamespace`-flavoured objects rather than the
real ones: `unload_gpu_plugins` needs only `.plugins` on the singleton and
`.unload()` on each plugin, and the seam under test is the ROUTE's decision to
unload — not the plugins themselves, which have their own tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.core.captioning.caption_service import CaptionService
from app.core.masking.masking_service import MaskingService
from app.core.scoring.scoring_service import ScoringService

_SERVICES = (
    (CaptionService, "_active_model_key", "caption"),
    (MaskingService, "_active_model_id", "masking"),
    (ScoringService, "_active_model_id", "scoring"),
)


class _FakePlugin:
    """Records that it was unloaded — the observable effect we assert on."""

    def __init__(self) -> None:
        self.unload_calls = 0

    def unload(self) -> None:
        self.unload_calls += 1


@pytest.fixture(autouse=True)
def _pristine_services():
    """Restore every service's class-level state.

    These are process-wide singletons: a test that leaves `_instance` or an
    active key set would make the NEXT test's "nothing is loaded" assertion
    pass or fail for the wrong reason.
    """
    saved = [(cls, cls._instance, getattr(cls, attr)) for cls, attr, _ in _SERVICES]
    CaptionService.cancel_idle_unload()
    for cls, attr, _ in _SERVICES:
        cls._instance = None
        setattr(cls, attr, None)
    yield
    for (cls, instance, key), (_, attr, _sid) in zip(saved, _SERVICES, strict=True):
        cls._instance = instance
        setattr(cls, attr, key)
    CaptionService.cancel_idle_unload()


def _load(cls, attr: str, key: str) -> _FakePlugin:
    """Put *cls* into the "a model is resident" state and return its plugin."""
    plugin = _FakePlugin()
    cls._instance = SimpleNamespace(plugins={"fake": plugin})
    setattr(cls, attr, key)
    return plugin


def _pretend_task_running(monkeypatch, task_type: str) -> None:
    """Make `gpu_batch_active` see one RUNNING task of *task_type*.

    Patched on `task_manager` itself rather than on a service, so the guard
    under test — the service's own check-then-act — still runs for real.
    """
    from app.core.tasks.task import TaskStatus
    from app.core.tasks.task_manager import task_manager

    monkeypatch.setattr(
        task_manager,
        "list",
        lambda *a, **k: [SimpleNamespace(type=task_type, status=TaskStatus.RUNNING)],
        raising=True,
    )


def _no_tasks_running(monkeypatch) -> None:
    from app.core.tasks.task_manager import task_manager

    monkeypatch.setattr(task_manager, "list", lambda *a, **k: [], raising=True)


# ── GET /gpu/loaded ───────────────────────────────────────────────────────


def test_loaded_reports_nothing_when_no_service_holds_a_model(client):
    r = client.get("/api/system/gpu/loaded")
    assert r.status_code == 200
    body = r.json()
    assert body["any_loaded"] is False
    assert [s["service"] for s in body["services"]] == ["caption", "masking", "scoring"]
    assert all(s["loaded"] is False and s["model"] is None for s in body["services"])


def test_loaded_reports_the_active_model_key_per_service(client):
    _load(CaptionService, "_active_model_key", "florence2:base")
    r = client.get("/api/system/gpu/loaded")
    body = r.json()
    assert body["any_loaded"] is True
    by_id = {s["service"]: s for s in body["services"]}
    assert by_id["caption"] == {
        "service": "caption",
        "label": "Captioning",
        "loaded": True,
        "model": "florence2:base",
    }
    assert by_id["masking"]["loaded"] is False
    assert by_id["scoring"]["loaded"] is False


def test_reading_loaded_state_never_constructs_a_singleton(client):
    """The observable effect of the "cheap read" requirement.

    `get_instance()` is the only thing that builds the plugin dict, and
    building it is what pulls the model wrappers in. If the route ever reached
    for the instance instead of the class attribute, `_instance` would be
    non-None after a plain GET.
    """
    assert client.get("/api/system/gpu/loaded").status_code == 200
    assert CaptionService._instance is None
    assert MaskingService._instance is None
    assert ScoringService._instance is None


# ── POST /gpu/unload — the measured effect ────────────────────────────────


def test_unload_clears_every_service_and_unloads_every_plugin(client, monkeypatch):
    _no_tasks_running(monkeypatch)
    plugins = {
        service_id: _load(cls, attr, f"{service_id}-model")
        for cls, attr, service_id in _SERVICES
    }

    r = client.post("/api/system/gpu/unload")
    assert r.status_code == 200
    body = r.json()

    # The effect, not the status code: keys cleared, plugins actually unloaded.
    for cls, attr, _service_id in _SERVICES:
        assert getattr(cls, attr) is None
    assert all(p.unload_calls == 1 for p in plugins.values())

    assert body["any_loaded"] is False
    assert sorted(body["unloaded"]) == ["caption", "masking", "scoring"]
    assert body["skipped"] == []
    assert all(s["loaded"] is False for s in body["services"])


def test_unload_reports_only_services_that_actually_held_a_model(client, monkeypatch):
    """`unloaded` is what was FREED, not what was asked.

    Without this, a UI toast would claim it freed three models on a machine
    where only one was resident.
    """
    _no_tasks_running(monkeypatch)
    _load(ScoringService, "_active_model_id", "hpsv2")

    body = client.post("/api/system/gpu/unload").json()
    assert body["unloaded"] == ["scoring"]
    assert body["any_loaded"] is False


def test_a_running_caption_batch_is_not_unloaded(client, monkeypatch):
    """THE negative: the batch keeps its model.

    A `caption_batch` owns the caption model for its whole run and frees it in
    its own `finally`. Pulling it out mid-run fails the rest of the dataset.
    """
    _pretend_task_running(monkeypatch, "caption_batch")
    caption_plugin = _load(CaptionService, "_active_model_key", "florence2:base")
    scoring_plugin = _load(ScoringService, "_active_model_id", "hpsv2")

    body = client.post("/api/system/gpu/unload").json()

    assert CaptionService._active_model_key == "florence2:base"
    assert caption_plugin.unload_calls == 0
    assert body["skipped"] == [
        {
            "service": "caption",
            "reason": "captioning is busy — a batch task is using the model",
        }
    ]
    assert body["unloaded"] == ["scoring"]
    assert scoring_plugin.unload_calls == 1
    # A skipped service is still reported as loaded, so the button stays.
    assert body["any_loaded"] is True
    assert next(s for s in body["services"] if s["service"] == "caption")["loaded"] is True


def test_a_running_mask_generate_batch_is_not_unloaded(client, monkeypatch):
    _pretend_task_running(monkeypatch, "mask_generate_batch")
    plugin = _load(MaskingService, "_active_model_id", "sam3")

    body = client.post("/api/system/gpu/unload").json()

    assert MaskingService._active_model_id == "sam3"
    assert plugin.unload_calls == 0
    assert [s["service"] for s in body["skipped"]] == ["masking"]


def test_a_running_rescan_does_not_lose_the_scoring_model(client, monkeypatch):
    """Scoring has no batch task of its own — it runs inside `rescan_batch`,
    which is why the guard names that type and not a `score_batch` that no
    longer exists."""
    _pretend_task_running(monkeypatch, "rescan_batch")
    plugin = _load(ScoringService, "_active_model_id", "hpsv2")

    body = client.post("/api/system/gpu/unload").json()

    assert ScoringService._active_model_id == "hpsv2"
    assert plugin.unload_calls == 0
    assert [s["service"] for s in body["skipped"]] == ["scoring"]


def test_a_mask_apply_batch_does_not_block_the_unload(client, monkeypatch):
    """`mask_apply_batch` composites already-written mask files
    (`MaskingService.mass_apply`) and never loads a plugin, so it must not hold
    the GPU hostage. This pins the exclusion the guard constant documents."""
    _pretend_task_running(monkeypatch, "mask_apply_batch")
    plugin = _load(MaskingService, "_active_model_id", "rembg")

    body = client.post("/api/system/gpu/unload").json()

    assert MaskingService._active_model_id is None
    assert plugin.unload_calls == 1
    assert body["skipped"] == []


def test_unloading_with_nothing_loaded_is_a_no_op_not_an_error(client, monkeypatch):
    _no_tasks_running(monkeypatch)
    body = client.post("/api/system/gpu/unload").json()
    assert body == {
        "any_loaded": False,
        "unloaded": [],
        "skipped": [],
        "services": [
            {"service": "caption", "label": "Captioning", "loaded": False, "model": None},
            {"service": "masking", "label": "Masking", "loaded": False, "model": None},
            {"service": "scoring", "label": "Scoring", "loaded": False, "model": None},
        ],
    }


def test_a_pending_caption_idle_unload_is_cancelled_by_the_global_unload(
    client, monkeypatch
):
    """The two release paths must not fight: after the user frees the GPU by
    hand there is nothing left for the idle timer to free, and a timer that
    survived would fire into whatever loaded next."""
    _no_tasks_running(monkeypatch)
    _load(CaptionService, "_active_model_key", "florence2:base")
    CaptionService.arm_idle_unload(30.0)
    assert CaptionService._idle_timer is not None

    client.post("/api/system/gpu/unload")
    assert CaptionService._idle_timer is None


# ── The GPU-only measurement ──────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_unload_actually_returns_vram_to_the_allocator(client, monkeypatch):
    """Measure the thing the user asked for: allocated VRAM falling.

    The fake plugin holds a real 64 MiB CUDA tensor and drops it in `unload()`,
    exactly as the model wrappers do; the route's `empty_cache()` then returns
    it. Asserting on `memory_allocated` is the only assertion here that could
    not be satisfied by a route that merely nulls a bookkeeping attribute.
    """
    _no_tasks_running(monkeypatch)

    class _VramPlugin:
        def __init__(self) -> None:
            self.buf = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device="cuda")

        def unload(self) -> None:
            self.buf = None

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    plugin = _VramPlugin()
    CaptionService._instance = SimpleNamespace(plugins={"fake": plugin})
    CaptionService._active_model_key = "vram-probe"

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    assert before - baseline >= 60 * 1024 * 1024, "probe tensor did not allocate"

    assert client.post("/api/system/gpu/unload").json()["any_loaded"] is False

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after <= baseline, f"VRAM not released: {after} > {baseline}"
