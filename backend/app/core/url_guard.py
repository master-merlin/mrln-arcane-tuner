"""Outbound URL containment for user-supplied provider endpoints.

Layer L0 substrate, deliberately here and **not** beside ``_path_guard``:
that file lives under ``app/api/`` (layer L3) and is already recorded as
substrate filed in the wrong place. Both the layer L2 engine and the layer L3
API need this check, so putting it in ``app/core/`` keeps them from importing
upward.

The problem
-----------
A user can point captioning at any OpenAI-compatible endpoint. That is a
deliberate feature -- Ollama, LM Studio and vLLM all run locally, and
``core/llm/openai_compat.py`` documents the host as intentionally unrestricted.
On a laptop that is exactly right: the "attacker" and the operator are the same
person, and reaching ``localhost:11434`` is the point.

In a container on rented infrastructure it is a different question. The same
field becomes a way to make the server issue requests to addresses the caller
cannot reach: other tenants on the private network, internal services, and
above all the cloud metadata endpoint at ``169.254.169.254``, which hands out
credentials to anything that asks from inside the instance.

So the guard is **mode-dependent**, which is the only honest resolution: the
local column keeps today's behaviour, the hosted column is contained, and a
user who genuinely runs a provider next to their container can opt back in.

What this does and does not protect against
-------------------------------------------
It resolves the hostname and rejects the request if **any** returned address is
not global (loopback, link-local, private, reserved, multicast, unspecified,
CGNAT shared address space -- the decision is ``not is_global``, the names are
for the log line). Checking every address matters: a name that resolves to one
public and one internal address would otherwise pass on the public one.

It does **not** close DNS rebinding. Between this check and the socket
connecting, a hostile name server can return a different address, and the only
complete fix is to connect to the address that was validated -- which means
owning the transport, not just the URL. That is a larger change than this
guard, and pretending otherwise would be worse than saying so: treat this as
raising the cost, not as a boundary you may trust with secrets.

Redirects are the same story in a different shape, and are the reason
:func:`assert_url_allowed` is exported separately -- a caller that follows
redirects MUST re-check each hop, or the first response simply points the
request wherever it likes.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from app.core.container_config import is_container
from app.core.logger import get_logger

logger = get_logger(__name__)

#: Schemes a provider URL may use. A non-HTTP scheme is always a
#: misconfiguration; rejecting it early gives a clear message instead of an
#: httpx UnsupportedProtocol raised deep inside a batch worker.
ALLOWED_SCHEMES = ("http", "https")

#: Opt back in to private/loopback destinations while running hosted. For the
#: user who really does run Ollama beside their container. Named so it reads as
#: a deliberate widening in a process list, not as a tuning knob.
ALLOW_PRIVATE_ENV = "MRLN_ALLOW_PRIVATE_PROVIDER_URLS"


class OutboundUrlRejected(ValueError):
    """A provider URL was refused. Subclasses ValueError so existing callers,
    which already surface ValueError as a configuration error, keep working."""


def _allow_private_override() -> bool:
    return os.environ.get(ALLOW_PRIVATE_ENV, "").strip().lower() in {"1", "true", "yes"}


def hosted_mode() -> bool:
    """True when outbound URLs should be contained.

    Keyed on the same container signal the rest of layer L0 uses, so there is
    one producer of "am I hosted?" rather than a second opinion here.
    """
    return is_container()


def _describe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return why *ip* is disallowed, or None if it is fine to reach.

    The deciding predicate is ``not ip.is_global`` (the last check): an address
    is reachable only if the stdlib calls it global. The named checks in front
    of it exist for the sentence they produce, not for the decision -- a list of
    "bad" classes is the wrong shape for a guard, because a class nobody listed
    passes. CGNAT ``100.64.0.0/10`` (RFC 6598, where Alibaba and Tencent serve
    instance metadata at ``100.100.100.200``) is exactly such a class: on Python
    3.12 it answers False to every predicate below AND to ``is_global``
    (release audit B6, LANE-69).

    ``is_multicast`` is the one named check that is also load-bearing:
    ``224.0.0.1`` answers ``is_global=True`` on the same interpreter, so the
    catch-all alone would let it through.
    """
    if ip.is_loopback:
        return "loopback"
    # Checked before is_private: 169.254.169.254 is link-local AND private, and
    # naming it exactly is what makes the log line useful.
    if ip.is_link_local:
        return "link-local (cloud metadata range)"
    if ip.is_private:
        return "private"
    if ip.is_reserved:
        return "reserved"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if not ip.is_global:
        return "non-global (e.g. CGNAT shared address space)"
    return None


def _resolved_addresses(host: str, port: int | None) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise OutboundUrlRejected(
            f"Provider host {host!r} could not be resolved: {e}."
        ) from e
    # De-duplicated but order-preserving, so the error names the first offender
    # deterministically rather than whichever the resolver happened to order.
    seen: dict[str, None] = {}
    for info in infos:
        seen.setdefault(info[4][0], None)
    return list(seen)


def assert_url_allowed(url: str, *, hosted: bool | None = None) -> None:
    """Raise :class:`OutboundUrlRejected` if *url* must not be requested.

    Call this on the ORIGINAL url and again on every redirect hop. Silent on
    success so it reads as an assertion at the call site.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise OutboundUrlRejected(
            f"Provider base URL must start with http:// or https:// (got {url!r})."
        )
    if not parsed.hostname:
        raise OutboundUrlRejected(f"Provider base URL has no host: {url!r}")

    contained = hosted_mode() if hosted is None else hosted
    if not contained:
        # Local install: the user's machine is the trust boundary and pointing
        # at a local provider is the documented, correct use. Unchanged.
        return

    if _allow_private_override():
        logger.warning(
            "outbound_url_private_allowed",
            host=parsed.hostname,
            reason=f"{ALLOW_PRIVATE_ENV} is set",
        )
        return

    for addr in _resolved_addresses(parsed.hostname, parsed.port):
        why = _describe(ipaddress.ip_address(addr))
        if why is not None:
            logger.warning(
                "outbound_url_blocked",
                host=parsed.hostname,
                address=addr,
                reason=why,
            )
            raise OutboundUrlRejected(
                f"Provider host {parsed.hostname!r} resolves to a {why} address "
                f"({addr}), which this server will not request while running "
                f"hosted. Set {ALLOW_PRIVATE_ENV}=1 if you intend to reach a "
                f"provider on the local network."
            )


def validate_base_url(url: str, *, hosted: bool | None = None) -> str:
    """Validate a provider base URL and return it normalised.

    Normalisation is only the trailing-slash strip the LLM clients already
    relied on; the check itself is :func:`assert_url_allowed`.
    """
    assert_url_allowed(url, hosted=hosted)
    return url.rstrip("/")
