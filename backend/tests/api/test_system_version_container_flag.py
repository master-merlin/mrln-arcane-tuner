"""``GET /api/system/version`` tells the SPA whether it runs in a container.

WHY THE FIELD EXISTS. The Server screen's Backend Port input is authoritative
on a local install and **ignored** in a container, where ``resolve_port``
treats the settings file as not a port source at all (argv -> ``PORT`` -> 8000)
and the host side of ``docker run -p`` lives in the daemon, unreadable from
inside. Until this field there was no way for the SPA to know, so the screen
offered an operator a control that silently did nothing.

WHY IT IS TESTED THROUGH THE HTTP RESPONSE rather than by calling the handler.
FastAPI filters a handler's return value through ``response_model`` and drops
undeclared keys **silently**. A test that called ``get_version()`` directly
would pass while the wire carried no ``container`` at all — the field would be
declared, returned, and invisible to the only consumer that matters.

WHY IT PATCHES THE FUNCTION RATHER THAN THE ENVIRONMENT (both are here, and
the pair is the point). Patching ``container_config.is_container`` and seeing
the response follow proves the route *uses* the shared resolver; a route that
re-derived ``os.environ.get("MRLN_CONTAINER") == "1"`` for itself would be
unaffected by that patch and would fail this test. That is RULE-21 asserted
behaviourally instead of by scanning the source for a forbidden string — a
source scan flags the comment explaining the rule, which is how two sibling
guards in this repo cried wolf on a clean tree.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


def _client() -> AsyncClient:
    from app.main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def _version_payload() -> dict:
    async with _client() as client:
        response = await client.get("/api/system/version")
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_the_field_survives_the_response_model():
    """Presence, not truthiness: ``False`` and "dropped" both read as falsey."""
    payload = await _version_payload()
    assert "container" in payload, (
        "GET /api/system/version carries no `container` key. If the handler "
        "returns one, VersionResponse has not declared it and FastAPI dropped "
        "it silently."
    )
    assert isinstance(payload["container"], bool)
    assert payload["version"], "the version field regressed while adding container"


@pytest.mark.asyncio
@pytest.mark.parametrize("in_container", [True, False])
async def test_the_flag_follows_the_shared_resolver(monkeypatch, in_container):
    """Both states, and the patch target is the contract.

    Driving both values is what makes this an assertion rather than a
    coincidence: on a developer machine the honest answer is always ``False``,
    so a one-sided test passes against a handler that hardcodes it.
    """
    from app.core import container_config

    monkeypatch.setattr(container_config, "is_container", lambda: in_container)
    payload = await _version_payload()
    assert payload["container"] is in_container


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("1", True), ("0", False), ("true", False), (None, False)],
)
async def test_the_real_environment_path_works_end_to_end(
    monkeypatch, env_value, expected
):
    """The production path, unpatched, including the exact-``"1"`` contract.

    ``"true"`` -> ``False`` is deliberate and matches ``container_config``:
    the entrypoint sets exactly ``"1"``, and widening the accepted set here
    would let a stray truthy value put a local install into container mode,
    where its port field would stop being the authority for no reason.
    """
    if env_value is None:
        monkeypatch.delenv("MRLN_CONTAINER", raising=False)
    else:
        monkeypatch.setenv("MRLN_CONTAINER", env_value)

    payload = await _version_payload()
    assert payload["container"] is expected
