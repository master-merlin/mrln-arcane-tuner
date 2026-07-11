"""Regression test — boogu_image REAL-path scheduler wiring (GPU-UAT bug).

SYMPTOM (live run job_log.jsonl): every sampling attempt (step-0 baseline
and every periodic sample) failed identically:

    "boogu_image sampler: driver.scheduler is None — the LOADER-provided
    vendored scheduler must be assigned (assign_components()) before
    sampling; a fresh/stock scheduler would silently drop the checkpoint's
    shift config."

samples/ was empty. Training itself was healthy (loss fell, 1000 steps,
checkpoints written) — only sampling ever touches ``driver.scheduler``.
That error is the sampler's own deliberate fail-loud guard
(``sampler.py``'s ``_denoise_base``) — it correctly refused to build a
stock scheduler. The bug is upstream of the guard.

ROOT CAUSE — ``PipelineLoadingMixin.load_model()``
(``app/engine/core/pipeline/pipeline_loading.py``), in order:

    1. ``self.components = await self.loader.load(...)``
       — the REAL ``BooguImageLoader`` manifest includes a ``"scheduler"``
       key: the checkpoint's vendored, shift-config-carrying instance.
    2. ``self._assign_components()``
       — ``driver.assign_components(self.components)`` sets
       ``driver.scheduler`` = that real instance. Correct, but ONLY
       transiently (see step 4).
    3. ``self.scheduler = self.init_scheduler()``
       — ``self.init_scheduler()`` resolves through the TRAINER's MRO to
       ``PipelineBaseMixin.init_scheduler()`` (``pipeline_base.py:44``),
       whose default unconditionally returns ``None``. No family trainer
       (boogu_image included) overrides this hook to delegate to the
       driver's OWN ``init_scheduler()`` (``driver.py:321``, which DOES
       correctly read ``self._components.get("scheduler")`` — but nothing
       on the real path ever calls the driver's version; it is dead code
       outside ``test_boogu_image_driver.py``, which calls it directly).
    4. ``self.components["scheduler"] = self.scheduler``
       — clobbers ``self.components["scheduler"]`` with ``None``.

The corruption in ``self.components`` is latent until the NEXT
``self._assign_components()`` call. ``_quantize_text_encoders()``
(``pipeline_loading.py``) calls ``self._assign_components()``
UNCONDITIONALLY at the end — regardless of ``te_quantization`` — and it
runs on EVERY real job via ``prepare_for_training()`` ->
``_quantize_components()`` -> ``_quantize_text_encoders()``
(``pipeline_optimization.py:76`` -> ``pipeline_loading.py:263-264``).
That re-sync re-reads the now-poisoned ``self.components`` dict and sets
``driver.scheduler = None``, clobbering the correct value step 2 set.

Why only boogu_image: it is the ONLY family whose loader manifest declares
a ``"scheduler"`` ``ComponentSpec`` (verified:
``grep -r 'key="scheduler"' app/engine/models/families`` — one hit,
``boogu_image/loader.py``). Every other new-family driver's
``init_scheduler()`` genuinely returns ``None`` (flow matching needs no
external scheduler) — see ``dreamlite``, ``ovis_image``, ``longcat_image``,
``prx``, ``prx_pixel``, ``hunyuan_video15``, ``kandinsky5`` — so the same
clobbering is harmless there: ``components["scheduler"]`` never held
anything meaningful to lose, and their samplers never read
``driver.scheduler`` at all. ``krea2`` (the other family with a
driver-level ``init_scheduler()``) also always returns ``None`` for the
same reason. sdxl is the only OTHER family whose trainer overrides
``init_scheduler()`` — but standalone (constructs its own ``DDPMScheduler``
directly), not delegating to a loader-provided component.

Unit tests hid this because ``test_boogu_image_driver.py`` calls
``driver.assign_components()`` / ``driver.init_scheduler()`` DIRECTLY,
never exercising the trainer-level ``self.init_scheduler()`` MRO
resolution or the ``load_model()`` -> ``_quantize_text_encoders()``
sequence — the historical krea2 C1-C4 bug class ("only the real path
crashes").

THE FIX (``trainer.py``): ``BooguImageTrainer.init_scheduler()`` now
overrides the ``PipelineBaseMixin`` hook to delegate to
``self.driver.init_scheduler()`` — which already correctly reads
``self._components["scheduler"]`` — mirroring the sdxl precedent of a
trainer-level override, but reusing the driver's existing, already-correct
implementation instead of duplicating scheduler-construction logic.

This test walks the REAL setup path — real ``BooguImageTrainer``, real
``BooguImageDriver``, real ``_setup_family()``, real ``load_model()``,
real ``_quantize_text_encoders()`` — with ONLY the loader's network I/O
faked (``_FakeLoader`` stands in for ``BooguImageLoader.load()``'s HF
calls). Nothing calls ``driver.assign_components()`` or
``driver.init_scheduler()`` directly from test code.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.boogu_image.trainer import BooguImageTrainer


def _definition() -> MagicMock:
    d = MagicMock(spec=ModelDefinition)
    d.family = "boogu_image"
    # Deliberately NOT a registered ModelRegistry id — load_model()'s
    # enrich_definition()/get_definition() calls become no-ops (warn +
    # return), so this test needs no registry fixture setup while still
    # exercising the real enrichment call site.
    d.id = "boogu-image-scheduler-wiring-test"
    d.lora_targetable_modules = ["dummy"]
    d.architecture_params = {}
    d.defaults = {}
    return d


class _FakeLoader:
    """Stands in for ``BooguImageLoader``'s HF ``from_pretrained`` I/O only.

    Returns the same manifest KEY SET the real loader produces (unet / vae
    / text_encoder / processor / scheduler — see ``loader.py``'s
    ``get_component_manifest``). Everything that CONSUMES this dict
    (``_assign_components()``, the trainer's ``init_scheduler()`` MRO
    resolution, ``_quantize_text_encoders()``) is real, un-mocked
    ``BooguImageTrainer`` / ``BooguImageDriver`` code.
    """

    def __init__(self, scheduler_instance: object) -> None:
        self._scheduler_instance = scheduler_instance
        self.warnings: list[str] = []
        self._root_path = None

    async def load(self, definition, torch_dtype=None, initial_device=None):
        return {
            "unet": torch.nn.Linear(2, 2),
            "vae": None,
            "text_encoder": None,
            "processor": None,
            "scheduler": self._scheduler_instance,
        }


def _real_trainer_with_faked_loader_io(scheduler_instance: object) -> BooguImageTrainer:
    """Real trainer + real driver via the real ``_setup_family()`` override;
    only the loader's HF I/O boundary is swapped out."""
    t = object.__new__(BooguImageTrainer)
    t.device = torch.device("cpu")
    t.definition = _definition()
    t.config = {
        "timestep_sampling": "uniform",
        "te_quantization": "none",
        "train_text_encoder": False,
        "sample_every_n_steps": 1,
    }
    t.logger = MagicMock()
    t._log_writer = None

    # Real _setup_family() (BooguImageTrainer's own override) builds the
    # real loader + real driver pair.
    t._setup_family()
    # Swap only the loader's network-touching load() — the I/O boundary —
    # for a tiny fake. driver/trainer wiring code is untouched.
    t.loader = _FakeLoader(scheduler_instance)
    return t


class TestRealPathSchedulerWiring:
    """Load-bearing regression test — FAILS RED on current main."""

    def test_driver_scheduler_survives_real_load_and_quantize_sequence(
        self,
        monkeypatch,
    ) -> None:
        # torch.cuda.empty_cache() is called unconditionally at the tail of
        # _quantize_text_encoders(); stub it so this CPU-only test doesn't
        # depend on / initialize a CUDA context.
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

        vendored_scheduler = object()  # stand-in for the vendored
        # FlowMatchEulerDiscreteScheduler instance the real loader hands
        # back — identity-checked below, so any accidental fresh/stock
        # instance or None fails the assertion.
        trainer = _real_trainer_with_faked_loader_io(vendored_scheduler)

        asyncio.run(trainer.load_model())

        # Every real job calls prepare_for_training() ->
        # _quantize_components() -> _quantize_text_encoders(),
        # UNCONDITIONALLY re-syncing driver components from
        # self.components — regardless of te_quantization ("none" here).
        trainer._quantize_text_encoders()

        assert trainer.driver.scheduler is vendored_scheduler, (
            "driver.scheduler must be the LOADER-provided vendored "
            "instance after the real load_model() + "
            "_quantize_text_encoders() sequence — got something else "
            "(likely None), which is exactly the GPU-UAT sampling failure "
            "('boogu_image sampler: driver.scheduler is None')."
        )

        sampler = trainer._create_sampler()
        assert sampler is not None, (
            "sample_every_n_steps > 0 must yield a real BooguImageSampler"
        )
        # Mirrors BooguImageSampler._denoise_base's own fail-loud
        # precondition (sampler.py) without running a full denoise loop.
        assert sampler.pipeline.driver.scheduler is not None, (
            "the sampler's 'driver.scheduler is None' guard would trip "
            "here on the real path"
        )
        assert sampler.pipeline.driver.scheduler is vendored_scheduler
