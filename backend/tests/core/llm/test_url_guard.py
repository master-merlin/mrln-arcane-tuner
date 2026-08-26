"""Outbound provider URLs are contained when hosted, unrestricted when local.

The local column is not a lesser case to be tolerated — pointing at
``localhost:11434`` for Ollama is the documented, correct use of the "custom"
provider, and a guard that breaks it has broken the feature. So the accept side
is asserted as firmly as the reject side.

The reject side exists because the same field, in a container on rented
infrastructure, is a way to make the server fetch ``169.254.169.254`` and hand
back instance credentials.
"""

from __future__ import annotations

import pytest

from app.core import url_guard
from app.core.url_guard import (
    ALLOW_PRIVATE_ENV,
    OutboundUrlRejected,
    assert_url_allowed,
    validate_base_url,
)


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """The opt-out must never leak in from the developer's own environment."""
    monkeypatch.delenv(ALLOW_PRIVATE_ENV, raising=False)


@pytest.fixture()
def resolves(monkeypatch):
    """Pin DNS so the tests assert on the guard, not on the network.

    Patching `_resolved_addresses` — not `socket` — keeps the test honest about
    which seam is under test: the address *policy*. Resolution failure has its
    own test below.
    """

    def _install(mapping: dict[str, list[str]]):
        def fake(host, port):  # noqa: ARG001
            if host not in mapping:
                raise OutboundUrlRejected(f"Provider host {host!r} could not be resolved.")
            return mapping[host]

        monkeypatch.setattr(url_guard, "_resolved_addresses", fake)

    return _install


# ── Hosted: the addresses that must never be requested ──────────────────

BLOCKED = [
    ("cloud metadata", "169.254.169.254"),
    ("link-local", "169.254.10.1"),
    ("loopback v4", "127.0.0.1"),
    ("loopback v6", "::1"),
    ("private 10/8", "10.0.0.5"),
    ("private 172.16/12", "172.16.4.9"),
    ("private 192.168/16", "192.168.1.50"),
    ("unique-local v6", "fd00::1"),
    ("unspecified", "0.0.0.0"),
]


@pytest.mark.parametrize("label,addr", BLOCKED, ids=[b[0] for b in BLOCKED])
def test_hosted_rejects_internal_addresses(resolves, label, addr):
    resolves({"provider.example": [addr]})
    with pytest.raises(OutboundUrlRejected) as exc:
        assert_url_allowed("http://provider.example/v1", hosted=True)
    # The message must name the address, or an operator cannot tell which of
    # several DNS answers tripped it.
    assert addr in str(exc.value)


def test_hosted_rejects_when_any_resolved_address_is_internal(resolves):
    """A name resolving to one public AND one internal address must be refused.

    Checking only the first answer is the classic hole: a hostile name server
    returns a public address first and an internal one second, and a guard that
    stops at index 0 waves it through.
    """
    resolves({"split.example": ["93.184.216.34", "169.254.169.254"]})
    with pytest.raises(OutboundUrlRejected) as exc:
        assert_url_allowed("https://split.example/v1", hosted=True)
    assert "169.254.169.254" in str(exc.value)


def test_hosted_allows_a_public_address(resolves):
    resolves({"api.openai.com": ["93.184.216.34"]})
    assert_url_allowed("https://api.openai.com/v1", hosted=True)


def test_hosted_rejects_unresolvable_host(resolves):
    resolves({})
    with pytest.raises(OutboundUrlRejected):
        assert_url_allowed("https://nope.invalid/v1", hosted=True)


# ── Local: today's behaviour, which the guard must not break ────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://192.168.1.50:8000/v1",
        "http://[::1]:11434/v1",
    ],
)
def test_local_still_reaches_local_providers(url):
    """Ollama / LM Studio / vLLM on the user's own machine keep working.

    No DNS pinning here on purpose: in local mode the guard must not resolve at
    all, so this also pins that it does no network I/O on the local path.
    """
    assert_url_allowed(url, hosted=False)


def test_local_mode_does_not_resolve(monkeypatch):
    """Prove the negative: the local path performs no name resolution."""

    def explode(*a, **k):  # noqa: ARG001
        raise AssertionError("local mode must not resolve the provider host")

    monkeypatch.setattr(url_guard, "_resolved_addresses", explode)
    assert_url_allowed("http://localhost:11434/v1", hosted=False)


# ── The opt-in, for a provider genuinely beside the container ───────────


def test_explicit_override_allows_private_when_hosted(resolves, monkeypatch):
    resolves({"ollama.internal": ["10.0.0.5"]})
    monkeypatch.setenv(ALLOW_PRIVATE_ENV, "1")
    assert_url_allowed("http://ollama.internal:11434/v1", hosted=True)


def test_override_is_off_unless_explicitly_truthy(resolves, monkeypatch):
    """An empty or accidental value must not widen the guard."""
    resolves({"ollama.internal": ["10.0.0.5"]})
    for value in ("", "0", "false", "no", "maybe"):
        monkeypatch.setenv(ALLOW_PRIVATE_ENV, value)
        with pytest.raises(OutboundUrlRejected):
            assert_url_allowed("http://ollama.internal:11434/v1", hosted=True)


# ── Scheme and shape, in both columns ───────────────────────────────────


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "notaurl"],
)
@pytest.mark.parametrize("hosted", [True, False])
def test_non_http_schemes_rejected_in_both_modes(url, hosted):
    with pytest.raises(OutboundUrlRejected):
        assert_url_allowed(url, hosted=hosted)


def test_validate_base_url_strips_trailing_slash():
    """The normalisation existing callers relied on must survive the rewrite."""
    assert validate_base_url("http://localhost:11434/v1/", hosted=False) == (
        "http://localhost:11434/v1"
    )


def test_rejection_is_a_valueerror():
    """Existing callers already treat a bad provider URL as ValueError.

    If this subclassing is ever dropped, those call sites stop catching and the
    failure changes shape from a configuration error into a 500.
    """
    assert issubclass(OutboundUrlRejected, ValueError)


# ── Redirects: the hop is where the guard is usually lost ───────────────


def test_redirect_target_is_checked_by_the_same_function(resolves):
    """A caller following redirects re-checks each hop with this same call.

    Pinned as behaviour rather than left to a comment: the first response can
    otherwise point the request anywhere it likes.
    """
    resolves({"safe.example": ["93.184.216.34"], "evil.example": ["169.254.169.254"]})
    assert_url_allowed("https://safe.example/v1", hosted=True)
    with pytest.raises(OutboundUrlRejected):
        assert_url_allowed("https://evil.example/v1", hosted=True)
