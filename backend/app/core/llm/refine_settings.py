"""The ONE producer of the Server screen's LLM endpoint values (RULE-21).

The ``llm_refine`` settings module holds ``{base_url, provider, model}``. Every
read of it goes through this module, for one reason found the hard way
(LANE-49):

    ``settings.get("model", DEFAULT)`` does **not** defend against a
    present-but-empty key.

``backend/settings.json`` really did contain ``"model": ""`` — the Server screen
persists whatever is in its picker, and an empty picker persists an empty
string. ``dict.get`` returned that ``""`` because the key was PRESENT, so the
default never applied, and caption refine POSTed ``{"model": ""}`` to Ollama,
which answers ``400 {"error":{"message":"model is required"}}`` (the traceback in
``server.log`` at 2026-08-31T17:37:17Z, ``caption_refine_batch.py:138``).

An unset optional setting and an empty one mean the same thing here — "the user
has not chosen" — so **empty is absent** at every read, and the default applies
to both. The accessors below are the only place that rule is written.

``base_url`` is returned in the ``SERVER_ROOT`` convention
(:mod:`app.core.llm.base_url_conventions`); a consumer that needs the OpenAI
API-base spelling transforms it there, never here.
"""

from __future__ import annotations

from app.core.llm.base_url_conventions import to_server_root

#: Settings module name. The user-visible surface is the Server screen's
#: "LLM Refine Endpoint" card; the module name never reaches the UI.
MODULE = "llm_refine"

#: Fallback endpoint. ``SERVER_ROOT`` convention — ``OllamaClient`` appends
#: ``/v1/chat/completions`` and ``/api/tags`` to it.
DEFAULT_BASE_URL = "http://localhost:11434"

#: Curated refine models, best-first. ``CURATED_MODELS[0]`` is the fallback the
#: refine paths use when the user has chosen nothing, and the Server screen
#: shows it by name so "no default" does not mean "unknown default".
CURATED_MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b-instruct-q4_K_M", "qwen2.5:3b-instruct"]

DEFAULT_MODEL = CURATED_MODELS[0]


def _manager():
    """Indirection seam — patched in tests."""
    from app.core.settings_manager import get_settings_manager

    return get_settings_manager()


def raw_settings() -> dict:
    """The stored ``llm_refine`` dict, or ``{}``."""
    return _manager().get_module_settings(MODULE) or {}


def stored_base_url_of(settings: dict) -> str:
    """The endpoint the user actually chose, or ``""`` -- the default is NOT
    applied.

    Used by the inheritance seam: inheriting a default nobody set would make a
    fresh install report the custom captioning provider as "configured" off an
    endpoint the user has never seen. Empty is still absent, so a stored ``""``
    reads as "chose nothing" rather than as a configured blank.
    """
    return str(settings.get("base_url") or "").strip()


def base_url_of(settings: dict) -> str:
    """Endpoint from an already-loaded settings dict, in ``SERVER_ROOT`` form.

    Empty/whitespace is treated as absent (see the module docstring), and a
    value a user typed in the OpenAI API-base spelling (``.../v1``) is folded
    back to the root so the client that appends ``/v1/chat/completions`` cannot
    double-suffix it.
    """
    stored = stored_base_url_of(settings)
    return to_server_root(stored) if stored else DEFAULT_BASE_URL


def model_of(settings: dict) -> str:
    """Default refine model from an already-loaded settings dict.

    Empty/whitespace is absent, so this never returns ``""`` and no caller can
    POST an empty model.
    """
    return str(settings.get("model") or "").strip() or DEFAULT_MODEL


def base_url() -> str:
    """Endpoint from storage, in ``SERVER_ROOT`` form."""
    return base_url_of(raw_settings())


def model() -> str:
    """Default refine model from storage. Never empty."""
    return model_of(raw_settings())
