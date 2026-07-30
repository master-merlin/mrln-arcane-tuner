"""Test support: make text-encoder RESIDENCY visible to sampler unit tests.

Why this exists
---------------
Family sampler tests stub ``driver.encode_text`` with a function that fabricates
tensors on the target device. That is fast and keeps the tests GPU-free, but it
also makes an entire class of bug invisible: the stub answers happily whether or
not the real text encoder is loaded, on the right device, or loaded at all.

That is not hypothetical. A "lazy uncond encode" optimisation moved a LIVE text
encoder forward outside the phase that brackets the TE onto the sampling device.
At the default guidance scale the sampler then ran ``encode_text`` against a
CPU-resident module with CUDA inputs and raised a device mismatch — which the
training loop's broad catch swallowed, so that family silently produced ZERO
preview images for a whole run. Both tests written for the change stubbed
``encode_text``, so neither could see it.

How to use it
-------------
Wrap the stub instead of replacing the driver's method outright::

    drv.text_encoder = FakeTextEncoder()          # somewhere to observe
    drv.encode_text = residency_checked(
        _stub_encode_text, driver=drv, device=device
    )

Now the stub still fabricates tensors (no GPU, no weights), but calling it while
the encoder sits off-device raises :class:`TextEncoderNotResident` instead of
quietly succeeding. ``torch.device("meta")`` stands in for "somewhere that is
not the sampling device", so this works on a CPU-only machine.
"""

from __future__ import annotations

import torch


class TextEncoderNotResident(AssertionError):
    """Raised when ``encode_text`` runs while its encoder is off-device."""


class FakeTextEncoder(torch.nn.Module):
    """A stand-in whose reported placement is observable and movable.

    Real text encoders are 5-40 GB; the placement contract only needs to know
    where the module currently claims to be. The reported device is modelled
    explicitly rather than by putting the parameter on a real ``meta`` device:
    the production bracket moves the encoder with ``.to(device)``, and torch
    refuses to move a meta tensor back ("Cannot copy out of meta tensor"), so a
    real meta parameter fails the bracket for the wrong reason and the test
    would pass whether or not the bracket existed.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.probe_device = torch.device(device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        target = kwargs.get("device")
        if target is None and args and isinstance(args[0], (str, torch.device)):
            target = args[0]
        if target is not None:
            self.probe_device = torch.device(target)
        return self

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError("FakeTextEncoder is a placement probe only")


def text_encoder_devices(driver) -> list[torch.device]:
    """Devices of every text encoder the driver exposes.

    Uses ``get_text_encoders()`` when present (the house contract) and falls
    back to the conventional attribute names, so this works for drivers that
    have not been fully constructed in a unit test.
    """
    encoders = {}
    getter = getattr(driver, "get_text_encoders", None)
    if callable(getter):
        try:
            encoders = getter() or {}
        except Exception:  # noqa: BLE001 - partially-built driver in a unit test
            encoders = {}
    if not encoders:
        encoders = {
            name: getattr(driver, name)
            for name in ("text_encoder", "text_encoder_2", "tokenizer_encoder")
            if isinstance(getattr(driver, name, None), torch.nn.Module)
        }

    devices: list[torch.device] = []
    for module in encoders.values():
        probe = getattr(module, "probe_device", None)
        if probe is not None:
            devices.append(torch.device(probe))
            continue
        param = next(module.parameters(), None)
        if param is not None:
            devices.append(param.device)
    return devices


def assert_te_resident(driver, device: torch.device) -> None:
    """Raise unless every text encoder sits on *device*."""
    off = [d for d in text_encoder_devices(driver) if d.type != torch.device(device).type]
    if off:
        raise TextEncoderNotResident(
            f"encode_text was called with text encoder(s) on {off} while the "
            f"sampler is on {device}. A live TE forward must happen INSIDE the "
            "phase that brackets the encoder onto the sampling device."
        )


def residency_checked(encode_text, *, driver, device: torch.device):
    """Wrap a stubbed ``encode_text`` so it fails when the TE is off-device.

    Meaningful only when the sampler device DIFFERS from where an offloaded
    encoder sits — i.e. on a real GPU run. On a CPU-only box the production
    bracket (``_ensure_on_gpu``) compares ``param.device.type`` against its own
    ``cpu`` target, finds them equal and correctly does nothing, so there is no
    move to observe. For CPU unit tests use :func:`record_bracket_order`, which
    pins the ordering that the residency actually depends on.
    """

    def _wrapped(*args, **kwargs):
        assert_te_resident(driver, device)
        return encode_text(*args, **kwargs)

    return _wrapped


def record_bracket_order(sampler, driver, *, encode_text):
    """Log the interleaving of GPU-bracket calls and ``encode_text`` calls.

    Returns the shared event list. Entries are ``("bracket", names)`` and
    ``("encode", captions)``, so a test can assert that every live encode is
    preceded by a bracket covering the text encoder — the property whose
    absence produced a whole run of empty previews, and the one thing about it
    that is observable without a GPU.
    """
    events: list[tuple] = []
    original_bracket = sampler._ensure_on_gpu

    def _bracket(names):
        events.append(("bracket", tuple(names)))
        return original_bracket(names)

    def _encode(captions, *args, **kwargs):
        events.append(("encode", tuple(captions)))
        return encode_text(captions, *args, **kwargs)

    sampler._ensure_on_gpu = _bracket
    driver.encode_text = _encode
    return events


def assert_encode_is_bracketed(events, component: str = "text_encoder") -> None:
    """Every ``encode`` event must follow a bracket naming *component*."""
    bracketed = False
    for kind, payload in events:
        if kind == "bracket" and component in payload:
            bracketed = True
        elif kind == "encode":
            if not bracketed:
                raise TextEncoderNotResident(
                    f"encode_text{payload} ran with no preceding "
                    f"_ensure_on_gpu([... {component!r} ...]). A live text-encoder "
                    "forward outside that bracket runs against an off-device "
                    "module — the failure is swallowed by the training loop's "
                    "catch and the run produces no previews at all."
                )
