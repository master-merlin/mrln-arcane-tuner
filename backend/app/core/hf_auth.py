"""Apply Hugging Face authentication from settings / environment.

This is the single place that writes the HF auth environment variables that
``huggingface_hub`` (and transitively ``transformers`` / ``diffusers``) read on
every download — ``HF_TOKEN`` and the legacy ``HUGGING_FACE_HUB_TOKEN``.

Precedence: an **externally provided** token (set in the process environment
before startup, e.g. a RunPod pod's ``HF_TOKEN``) always wins. The in-app token
saved in Server → Models is the fallback, used only when no external token is
present. Captured once at import — before we start writing these vars ourselves
— so the external value can't be shadowed by our own writes and a settings-only
token can still be cleared later.

The token value is never logged.
"""
from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)

_HF_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Snapshot any externally-provided token ONCE, at import, before apply_hf_auth
# can overwrite os.environ.
_EXTERNAL_HF_TOKEN: str = (
    os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
).strip()


def apply_hf_auth(settings_token: str | None) -> str:
    """Write the effective HF token into the environment and return it.

    Effective token = external env token (wins) or the saved settings token.
    Clears both env vars when neither is present.
    """
    effective = _EXTERNAL_HF_TOKEN or (settings_token or "").strip()
    if effective:
        for var in _HF_ENV_VARS:
            os.environ[var] = effective
    else:
        for var in _HF_ENV_VARS:
            os.environ.pop(var, None)
    logger.info(
        "hf_auth_applied",
        source=("env" if _EXTERNAL_HF_TOKEN else ("settings" if effective else "none")),
        active=bool(effective),
    )
    return effective
