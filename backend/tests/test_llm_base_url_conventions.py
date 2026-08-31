"""LANE-49 (b): one address, two conventions, and the transform between them.

The regression these pin: LANE-46 made the custom captioning provider inherit
``llm_refine.base_url``. Both sides then held the same string and every existing
test agreed with both -- because every existing test asserted that the badge and
the request used the *same string*, never that the string was right for the
consumer that would spell a URL out of it.

So the assertions here are on **the URL a consumer actually REQUESTS**, captured
off an ``httpx`` transport. A test that only checks the two sides agree cannot
see a value that is wrong for one of them.

Measured against a live Ollama on 2026-08-31 (the fact the transform encodes):
``GET http://localhost:11434/models`` answers 404,
``GET http://localhost:11434/v1/models`` answers 200 with 3 models.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.core.llm import provider_settings
from app.core.llm.base_url_conventions import (
    is_usable_endpoint,
    to_openai_api_base,
    to_server_root,
)
from app.core.llm.ollama_client import OllamaClient
from app.core.llm.openai_compat import PROVIDER_BASE_URLS, list_models

SERVER_ROOT = "http://localhost:11434"
OPENAI_BASE = "http://localhost:11434/v1"


# --------------------------------------------------------------------------
# The transforms themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (SERVER_ROOT, OPENAI_BASE),  # the actual regression input
        ("http://localhost:11434/", OPENAI_BASE),  # trailing slash
        ("  http://localhost:11434  ", OPENAI_BASE),  # user whitespace
        (OPENAI_BASE, OPENAI_BASE),  # already converted: NO double suffix
        ("http://localhost:11434/v1/", OPENAI_BASE),
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),  # LM Studio
        # gemini's builtin is an OpenAI base with a non-/v1 tail; appending /v1
        # to it would 404 every model list.
        ("https://generativelanguage.googleapis.com/v1beta/openai",
         "https://generativelanguage.googleapis.com/v1beta/openai"),
        ("", ""),  # "configured nowhere" must not become "/v1"
        ("   ", ""),
    ],
)
def test_to_openai_api_base(stored: str, expected: str) -> None:
    assert to_openai_api_base(stored) == expected


def test_to_openai_api_base_is_idempotent() -> None:
    """Applying it twice must not add a second ``/v1``.

    This is the property the fix rests on: the transform runs on every read, so
    a user who typed the API-base spelling into either field is transformed too.
    """
    for raw in (SERVER_ROOT, OPENAI_BASE, "https://api.openai.com/v1", "https://h/ollama"):
        once = to_openai_api_base(raw)
        assert to_openai_api_base(once) == once, raw


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (SERVER_ROOT, SERVER_ROOT),
        (OPENAI_BASE, SERVER_ROOT),
        ("http://localhost:1234/v1", "http://localhost:1234"),  # LM Studio
        ("http://localhost:11434/", SERVER_ROOT),
        # /openai is a hosted provider's tail, never an OllamaClient destination
        ("https://generativelanguage.googleapis.com/v1beta/openai",
         "https://generativelanguage.googleapis.com/v1beta/openai"),
    ],
)
def test_to_server_root(stored: str, expected: str) -> None:
    assert to_server_root(stored) == expected


def test_every_builtin_provider_base_is_already_an_openai_api_base() -> None:
    """The evidence the ``OPENAI_API_BASE_SUFFIXES`` row cites, asserted.

    If a provider is added whose base is a bare server root, the transform will
    silently rewrite it -- which is correct -- but the claim in the comment
    ("every entry ends in one of these") would have become false, and a comment
    that describes a check the code does not make is worse than none.
    """
    for provider, base in PROVIDER_BASE_URLS.items():
        if base is None:  # custom has no builtin
            continue
        assert to_openai_api_base(base) == base, f"{provider} -> {base}"


# --------------------------------------------------------------------------
# What the CONSUMER requests -- the assertion the old tests could not make
# --------------------------------------------------------------------------


def _server_screen_holds(monkeypatch, base_url: str) -> None:
    """Point the Server screen's ``llm_refine`` store at *base_url*.

    Patched at ``provider_settings._manager`` — the same seam the module's own
    settings reads use — so this exercises the real inheritance branch rather
    than a stub standing in for it.
    """

    class _Fake:
        def get_module_settings(self, module: str) -> dict:
            return {"base_url": base_url} if module == provider_settings.SERVER_SETTINGS_MODULE else {}

    monkeypatch.setattr(provider_settings, "_manager", _Fake)


def _capture_get(status: int = 200, payload: dict | None = None) -> tuple[list[str], httpx.MockTransport]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(status, json=payload if payload is not None else {"data": []})

    return seen, httpx.MockTransport(handler)


def test_openai_compat_requests_v1_models_for_an_inherited_server_root(monkeypatch) -> None:
    """The regression, end to end, asserted on the wire.

    Server screen holds a SERVER_ROOT; the custom provider's own store is empty;
    the captioning consumer must still GET ``/v1/models``. Before the fix this
    requested ``http://localhost:11434/models`` -- 404 against a live Ollama.
    """
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": ""})
    _server_screen_holds(monkeypatch, SERVER_ROOT)

    effective = provider_settings.effective_base_url("custom")
    assert effective.source == "server_settings"

    seen, transport = _capture_get(payload={"data": [{"id": "gemma3:12b"}]})
    models = list_models(base_url=effective.base_url, api_key=None, transport=transport)

    assert seen == ["http://localhost:11434/v1/models"]
    assert models == ["gemma3:12b"]


def test_openai_compat_does_not_double_suffix_a_v1_the_user_typed(monkeypatch) -> None:
    """A user who already typed the API-base spelling gets exactly one ``/v1``."""
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": ""})
    _server_screen_holds(monkeypatch, OPENAI_BASE)

    seen, transport = _capture_get()
    list_models(base_url=provider_settings.effective_base_url("custom").base_url,
                api_key=None, transport=transport)
    assert seen == ["http://localhost:11434/v1/models"]


def test_openai_compat_does_not_double_suffix_the_providers_own_field(monkeypatch) -> None:
    """The other field the user can type into (``api_captioning.custom``)."""
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": OPENAI_BASE})
    seen, transport = _capture_get()
    list_models(base_url=provider_settings.effective_base_url("custom").base_url,
                api_key=None, transport=transport)
    assert seen == ["http://localhost:11434/v1/models"]


def test_openai_compat_upgrades_a_server_root_typed_into_the_providers_own_field(monkeypatch) -> None:
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": SERVER_ROOT})
    seen, transport = _capture_get()
    list_models(base_url=provider_settings.effective_base_url("custom").base_url,
                api_key=None, transport=transport)
    assert seen == ["http://localhost:11434/v1/models"]


@pytest.mark.parametrize("stored", [SERVER_ROOT, OPENAI_BASE, "http://localhost:11434/"])
def test_ollama_client_requests_one_v1_whatever_convention_is_stored(stored: str) -> None:
    """The other consumer of the same address, asserted on the wire.

    ``http://localhost:1234/v1`` is LM Studio's own documented endpoint and the
    Server screen offers it as that provider's default, so it reaches this
    client; before the fix it produced ``/v1/v1/chat/completions``.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await OllamaClient(base_url=stored, client=http).chat("m", "s", "u")

    assert asyncio.run(run()) == "ok"
    assert seen == ["http://localhost:11434/v1/chat/completions"]


def test_ollama_client_does_not_double_suffix_lm_studios_default() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"models": []})

    async def run() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await OllamaClient(base_url="http://localhost:1234/v1", client=http).list_models()

    asyncio.run(run())
    assert seen == ["http://localhost:1234/api/tags"]


# --------------------------------------------------------------------------
# ``configured`` -- a value that cannot be used is not configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "usable"),
    [
        (SERVER_ROOT, True),
        (OPENAI_BASE, True),
        ("https://api.openai.com/v1", True),
        ("", False),
        ("   ", False),
        ("localhost:11434", False),  # no scheme -- a plausible thing to type
        ("localhost:11434/v1", False),
        ("ftp://localhost:11434/v1", False),
        ("http:///v1", False),  # scheme, no host
        ("not a url", False),
    ],
)
def test_is_usable_endpoint(url: str, usable: bool) -> None:
    assert is_usable_endpoint(url) is usable


def test_a_scheme_less_endpoint_is_not_reported_as_configured(monkeypatch, client) -> None:
    """``configured`` must test usability, not non-emptiness.

    ``localhost:11434`` is a plausible thing to type - and it is non-empty, so
    ``bool(base_url)`` called it configured while ``openai_compat`` refused it
    ("Provider base URL must start with http:// or https://") three layers down,
    as a 502. That is the badge/request disagreement in its second form: not a
    wrong convention this time, but a string that addresses nothing.
    """
    _server_screen_holds(monkeypatch, "localhost:11434")
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": ""})

    status = next(s for s in client.get("/api/captions/api-providers").json()
                  if s["provider"] == "custom")
    assert status["configured"] is False

    # ...and the request side agrees, with a reason naming the value.
    with pytest.raises(ValueError, match="not a usable endpoint"):
        provider_settings.resolve_provider("custom")


def test_a_usable_endpoint_is_still_reported_as_configured(monkeypatch, client) -> None:
    """Positive control: the predicate must be able to answer True.

    Without this, tightening ``configured`` to something that is always False
    would pass the test above.
    """
    _server_screen_holds(monkeypatch, SERVER_ROOT)
    monkeypatch.setattr(provider_settings, "get_provider_raw",
                        lambda p: {"api_key": "", "base_url": ""})

    status = next(s for s in client.get("/api/captions/api-providers").json()
                  if s["provider"] == "custom")
    assert status["configured"] is True
    assert status["base_url"] == OPENAI_BASE
    assert provider_settings.resolve_provider("custom").base_url == OPENAI_BASE
