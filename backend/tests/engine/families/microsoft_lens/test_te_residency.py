"""The lazy uncond encode must happen with the text encoder RESIDENT.

W5.T10 made ``denoise()`` encode the CFG unconditional ("") embedding lazily —
a LIVE ``driver.encode_text`` call from inside the denoise loop, which is
outside the Phase-1 bracket that moves the text encoder onto the sampling
device. At the default guidance scale that ran the encoder against CPU-resident
weights with CUDA inputs and raised a device mismatch; ``pipeline_train``'s
broad catch swallowed it, so this family silently produced ZERO preview images
for an entire run. The fix was ``self._ensure_on_gpu(["text_encoder"])``
immediately before the encode.

Neither test written for that change could have caught it: both replaced
``encode_text`` with a stub that fabricates tensors on the target device and
never consults the encoder at all. This pins the placement contract instead of
the return value — see ``tests/support/te_placement.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.support.te_placement import (
    FakeTextEncoder,
    TextEncoderNotResident,
    assert_encode_is_bracketed,
    assert_te_resident,
    record_bracket_order,
    residency_checked,
)

from .test_sampler import _defn, _fake_prompt_embedding, _FakeVAE, _tiny_dit


def _sampler_with_observable_te(device=torch.device("cpu")):
    """The standard sampler fixture, plus a text encoder whose device moves.

    ``encode_text`` still fabricates tensors (no GPT-OSS weights, no GPU), but
    it now refuses to answer while the encoder is off-device.
    """
    from app.engine.core.text_encoding import TextEncoderOutput
    from app.engine.models.families.microsoft_lens.driver import MicrosoftLensDriver
    from app.engine.models.families.microsoft_lens.sampler import MicrosoftLensSampler

    dit = _tiny_dit().eval()
    vae = _FakeVAE()
    drv = MicrosoftLensDriver(_defn(), device)
    drv.transformer = dit
    drv.vae = vae

    # Starts OFF the sampling device — "meta" is just a name for "not here"
    # that needs no GPU. The production bracket has to move it before encoding.
    te = FakeTextEncoder(device="meta")
    drv.text_encoder = te

    calls: list[str] = []

    def _stub_encode_text(captions, dtype, s_txt=5):
        calls.append(captions[0] if captions else "")
        return TextEncoderOutput(
            embeddings=torch.randn(
                len(captions), 4, s_txt, 2880, dtype=dtype, device=device
            ),
            attention_mask=torch.ones(
                len(captions), s_txt, dtype=torch.bool, device=device
            ),
        )

    pipeline = SimpleNamespace(
        config={},
        device=device,
        driver=drv,
        transformer=dit,
        vae=vae,
        components={"text_encoder": te},
        _block_swap_managers=None,
    )
    sampler = MicrosoftLensSampler(pipeline)
    events = record_bracket_order(sampler, drv, encode_text=_stub_encode_text)
    return sampler, drv, te, calls, events


class TestSupportHelperItself:
    """The probe has to fail on the bug before it is worth trusting."""

    def test_offdevice_encoder_is_rejected(self):
        drv = SimpleNamespace(text_encoder=FakeTextEncoder(device="meta"))
        with pytest.raises(TextEncoderNotResident):
            assert_te_resident(drv, torch.device("cpu"))

    def test_resident_encoder_is_accepted(self):
        drv = SimpleNamespace(text_encoder=FakeTextEncoder())
        assert_te_resident(drv, torch.device("cpu"))

    def test_wrapper_passes_through_when_resident(self):
        drv = SimpleNamespace(text_encoder=FakeTextEncoder())
        wrapped = residency_checked(
            lambda *a, **kw: "encoded", driver=drv, device=torch.device("cpu")
        )
        assert wrapped(["x"], torch.float32) == "encoded"

    def test_driver_with_no_encoder_is_not_a_false_positive(self):
        """A partially-built driver must not trip the probe."""
        assert_te_resident(SimpleNamespace(), torch.device("cpu"))


class TestLazyUncondEncodeIsBracketed:
    def test_cfg_encode_is_preceded_by_the_te_bracket(self):
        """With guidance > 1 the sampler encodes "" mid-denoise — a LIVE text
        encoder forward from inside the denoise loop. It must bracket the
        encoder onto the sampling device first."""
        sampler, drv, te, calls, events = _sampler_with_observable_te()

        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = sampler._create_initial_noise(64, 64, gen)
        emb = _fake_prompt_embedding(sampler.device, torch.float32)

        sampler.denoise(noise, emb, num_steps=2, guidance_scale=3.5, seed=0)

        assert "" in calls, "the unconditional prompt was never encoded"
        # Deleting the _ensure_on_gpu(["text_encoder"]) line fails this.
        assert_encode_is_bracketed(events)

    def test_no_encode_at_all_when_cfg_is_off(self):
        """guidance <= 1 needs no uncond pass, so nothing should touch the
        encoder — no encode means no bracket is required either."""
        sampler, drv, te, calls, events = _sampler_with_observable_te()

        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = sampler._create_initial_noise(64, 64, gen)
        emb = _fake_prompt_embedding(sampler.device, torch.float32)
        sampler.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)

        assert calls == []
        assert_encode_is_bracketed(events)

    def test_the_ordering_probe_rejects_an_unbracketed_encode(self):
        """The probe has to fail on the bug it exists for."""
        with pytest.raises(TextEncoderNotResident):
            assert_encode_is_bracketed([("encode", ("",))])

    def test_the_ordering_probe_ignores_a_bracket_for_another_component(self):
        with pytest.raises(TextEncoderNotResident):
            assert_encode_is_bracketed(
                [("bracket", ("vae",)), ("encode", ("",))]
            )
