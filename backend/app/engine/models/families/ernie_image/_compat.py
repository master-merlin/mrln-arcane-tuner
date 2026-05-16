"""Runtime shims that let ERNIE-Image's text encoder load on transformers 4.57.

Baidu shipped ``baidu/ERNIE-Image`` with a ``text_encoder/config.json``
exported from an unreleased ``transformers 5.2.0`` dev build.  Two
forward-looking pieces of that config break ``transformers 4.57``:

1. ``text_config.model_type == "ministral3"`` — the next-gen Ministral
   variant.  ``transformers 4.57`` only knows ``"ministral"`` (no "3"
   suffix) and raises ``KeyError('ministral3')`` when ``Mistral3Config``
   tries to build the inner text config.

2. ``text_config.sliding_window == null`` — in 5.x ``None`` means
   "no sliding window / full attention".  4.57's ``MinistralModel``
   forward pass requires a numeric value and crashes with
   ``Could not find a sliding_window argument in the config, or it
   is not set``.

Both fixes are surface-level: ``ministral3`` is architecturally
identical to ``ministral`` in 4.57 (same RoPE/YaRN, same layer counts,
same attention shapes — the only divergence is extra config knobs
silently ignored by ``__init__``).  We:

* Register a ``Ministral3Config`` subclass with ``model_type="ministral3"``
  and a constructor that fills ``sliding_window`` from
  ``max_position_embeddings`` whenever it arrives as ``None``.
* Register a ``Ministral3Model`` subclass whose ``config_class`` points
  at the new config so ``AutoModel.register`` accepts it.
* Wire both into ``AutoConfig`` / ``AutoModel`` with ``exist_ok=True``
  so re-importing is a no-op.

These registrations are global Transformers state but idempotent and
harmless to other families.  Import happens via the ERNIE-Image
``family.py`` so it only runs once the registry discovers this family
(or directly imports it during a job).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def _register_ministral3_aliases() -> None:
    """Idempotently teach transformers 4.x how to parse ``ministral3`` configs."""
    try:
        from transformers import AutoConfig, AutoModel
        from transformers.models.ministral.configuration_ministral import (
            MinistralConfig,
        )
        from transformers.models.ministral.modeling_ministral import (
            MinistralModel,
        )
    except ImportError as exc:
        logger.warning(
            "ernie_image_compat_imports_failed",
            error=str(exc),
            hint="Ministral may be unavailable in this transformers version",
        )
        return

    class Ministral3Config(MinistralConfig):
        model_type = "ministral3"

        def __init__(self, **kwargs):
            # transformers 5.x treats sliding_window=None as "full attention",
            # but 4.x's MinistralModel forward expects a numeric value.  Default
            # to max_position_embeddings so the window covers any realistic
            # caption length.
            if kwargs.get("sliding_window", "_unset") in (None, "_unset"):
                kwargs["sliding_window"] = kwargs.get(
                    "max_position_embeddings", 262144,
                )
            super().__init__(**kwargs)

    class Ministral3Model(MinistralModel):
        config_class = Ministral3Config

    AutoConfig.register("ministral3", Ministral3Config, exist_ok=True)
    AutoModel.register(Ministral3Config, Ministral3Model, exist_ok=True)
    logger.info("ernie_image_compat_registered", aliases=["ministral3"])


# Execute on module import.  Safe to call repeatedly — exist_ok=True makes
# the registration idempotent.
_register_ministral3_aliases()
