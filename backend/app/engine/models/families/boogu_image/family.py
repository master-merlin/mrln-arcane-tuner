"""Boogu-Image model family registration.

The moment ``family.py`` exists in this directory,
``registry.discover_families()`` imports it and
``registry.get_family_class("boogu_image")`` resolves — which
``resolve_capabilities()`` requires (see
``app/engine/core/archetypes.py::resolve_capabilities``) for the two
shipped definitions to pass the suite-wide
``backend/tests/test_resolve_capabilities.py`` guard.

``get_trainer_class`` returns the real ``BooguImageTrainer``, which wires
in the real loader/driver/sampler/saver itself (``_setup_family`` /
``_create_sampler`` in ``trainer.py``; ``get_saver`` on
``BooguImageDriver`` in ``driver.py``) — matching the krea2/dreamlite
precedent, where ``ModelFamily`` only ever exposes ``get_trainer_class``
and the trainer owns the rest of the component wiring. No
``capability_overrides`` are declared: the plain ``latent_diffusion``
archetype defaults (``has_vae=True``, ``has_external_te=True``,
``supports_train_te=False``) already match Boogu-Image's architecture
(FLUX.1-dev VAE + external Qwen3-VL mllm, no TE-LoRA training) — the same
choice krea2 makes.
"""

from app.engine.core.definitions import ModelFamily

from .trainer import BooguImageTrainer


class BooguImageFamily(ModelFamily):
    """Boogu-Image (Base / Turbo) implementation of the ModelFamily logic provider."""

    family_name = "boogu_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return BooguImageTrainer
