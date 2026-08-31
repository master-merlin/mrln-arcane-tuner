"""The two base-URL conventions this app stores, and the transforms between them.

One LLM server has ONE address, but its consumers spell that address
differently, and until LANE-49 nothing in the code said so. The two shapes:

``SERVER_ROOT``
    The bare origin of the server — ``http://localhost:11434``. Its consumer is
    :class:`app.core.llm.ollama_client.OllamaClient`, which appends the full
    path itself: ``/v1/chat/completions`` (``ollama_client.py:71``),
    ``/api/tags`` (``:82``), ``/api/pull`` (``:99``). This is the convention of
    the ``llm_refine.base_url`` setting the Server screen writes.

``OPENAI_API_BASE``
    The OpenAI-compatible **API base**, i.e. the origin plus the version
    segment — ``http://localhost:11434/v1``. Its consumer is
    :mod:`app.core.llm.openai_compat`, which appends only the resource:
    ``/models`` (``openai_compat.py:200``) and ``/chat/completions``
    (``:122``). Every entry in ``PROVIDER_BASE_URLS``
    (``openai_compat.py:27-33``) is in this convention, which is what makes it
    the convention of the ``api_captioning.providers.*.base_url`` store.

**Why this module exists.** LANE-46 made the custom captioning provider inherit
``llm_refine.base_url``. That was a copy, and a copy is only correct when both
stores hold the same KIND of string — these two never did. Measured against a
live Ollama: ``GET http://localhost:11434/models`` → 404,
``GET http://localhost:11434/v1/models`` → 200. So inheriting is a *transform*,
and it lives here, once.

Both transforms are idempotent: a user who has already typed a ``/v1``-suffixed
URL into either field must not get a second one appended.
"""

from __future__ import annotations

#: Path suffixes that mean "this string is already an OpenAI API base".
#:
#: Evidence: every value in ``openai_compat.PROVIDER_BASE_URLS``
#: (``openai_compat.py:27-33``) ends in one of these — ``.../v1`` for
#: openai/anthropic/openrouter, ``.../v1beta/openai`` for gemini — and LM Studio's
#: own documented endpoint (the Server screen's ``PROVIDER_DEFAULTS.lmstudio``,
#: ``llm-endpoint-settings.ts:14``) is ``http://localhost:1234/v1``.
#:
#: This list is deliberately short and literal. A reverse-proxy root such as
#: ``https://host/ollama`` matches nothing here and therefore gets ``/v1``
#: appended — which is correct, because it IS a server root.
OPENAI_API_BASE_SUFFIXES = ("/v1", "/openai")


def to_openai_api_base(url: str) -> str:
    """Return *url* in the ``OPENAI_API_BASE`` convention.

    A server root gains ``/v1``; a string already in this convention is
    returned unchanged (idempotent — see the module docstring). ``""`` stays
    ``""``: "configured nowhere" must not become ``"/v1"``.
    """
    trimmed = url.strip().rstrip("/")
    if not trimmed:
        return ""
    if trimmed.endswith(OPENAI_API_BASE_SUFFIXES):
        return trimmed
    return f"{trimmed}/v1"


def to_server_root(url: str) -> str:
    """Return *url* in the ``SERVER_ROOT`` convention.

    Strips a trailing ``/v1`` so that a value stored in the OpenAI API-base
    convention can still be handed to a client that appends its own full path.
    Without this, LM Studio's documented ``http://localhost:1234/v1`` typed into
    the Server screen would make ``OllamaClient.chat`` request
    ``/v1/v1/chat/completions``.

    ``/openai`` is NOT stripped: that suffix belongs to a hosted provider
    (gemini) which has no server-root form and is never an ``OllamaClient``
    destination.
    """
    trimmed = url.strip().rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


def is_usable_endpoint(url: str) -> bool:
    """True when *url* could form a request, not merely when it is non-empty.

    ``configured`` used to be ``bool(base_url)``, which is a test of whether a
    string exists rather than of whether it addresses anything: a green badge
    followed a 502 (LANE-49). A value that cannot be used is not configuration,
    so the badge asks this instead. It is a SHAPE check -- absolute http(s) URL
    with a host -- and deliberately not a reachability check: a status route
    that dials the network turns a page load into a timeout.
    """
    from urllib.parse import urlsplit

    trimmed = url.strip()
    if not trimmed:
        return False
    try:
        parts = urlsplit(trimmed)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)
