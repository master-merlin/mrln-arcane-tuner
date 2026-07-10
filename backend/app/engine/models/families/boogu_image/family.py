"""Boogu-Image model family registration.

Task 2 scope (this file): family + capability registration ONLY. The moment
``family.py`` exists in this directory, ``registry.discover_families()``
imports it and ``registry.get_family_class("boogu_image")`` resolves — which
``resolve_capabilities()`` requires (see
``app/engine/core/archetypes.py::resolve_capabilities``) for the two
definitions this task ships to pass the suite-wide
``backend/tests/test_resolve_capabilities.py`` guard.

Loader/driver-logic/trainer/sampler/saver land in Tasks 3-7 (see
``.agent/workdir/sdd-boogu/task-2-brief.md``) — ``get_trainer_class`` is
intentionally a stub until a later task lands a real ``BooguImageTrainer``.
No ``capability_overrides`` are declared: the plain ``latent_diffusion``
archetype defaults (``has_vae=True``, ``has_external_te=True``,
``supports_train_te=False``) already match Boogu-Image's architecture
(FLUX.1-dev VAE + external Qwen3-VL mllm, no TE-LoRA training) — the same
choice krea2 makes.
"""

from app.engine.core.definitions import ModelFamily


class BooguImageFamily(ModelFamily):
    """Boogu-Image (Base / Turbo) implementation of the ModelFamily logic provider."""

    family_name = "boogu_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        raise NotImplementedError(
            "boogu_image trainer lands in a later task — see "
            ".agent/workdir/sdd-boogu/task-2-brief.md"
        )
