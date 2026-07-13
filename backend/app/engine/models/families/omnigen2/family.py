"""OmniGen2 model family registration.

The moment ``family.py`` exists in this directory,
``registry.discover_families()`` imports it and
``registry.get_family_class("omnigen2")`` resolves — which
``resolve_capabilities()`` requires for the shipped definition to pass the
suite-wide ``backend/tests/test_resolve_capabilities.py`` guard.

No ``capability_overrides``: the plain ``latent_diffusion`` archetype
defaults (``has_vae=True``, ``has_external_te=True``,
``supports_train_te=False``) match OmniGen2's architecture (FLUX.1-dev VAE
+ external Qwen2.5-VL mllm, no TE-LoRA) — the krea2/boogu_image choice.
"""

from app.engine.core.definitions import ModelFamily

from .trainer import OmniGen2Trainer


class OmniGen2Family(ModelFamily):
    """OmniGen2 implementation of the ModelFamily logic provider."""

    family_name = "omnigen2"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Edit definitions (control_inputs > 0 — the shipped omnigen2.yaml)
        # train on paired control images — dispatch to the Edit subclass
        # (boogu_image/qwen_image family-level dispatch precedent). Lazy
        # import so registry discovery stays light.
        if int(getattr(self.definition, "control_inputs", 0) or 0) > 0:
            from .trainer_edit import OmniGen2EditTrainer  # noqa: PLC0415

            return OmniGen2EditTrainer
        return OmniGen2Trainer
