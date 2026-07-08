"""hv15 driver forward + dual-TE encode wiring (fakes, real driver code).

Pins:
- T2V forward: 65-ch input with zero cond/mask, zero (B,729,1152)
  ``image_embeds``, RAW [0,1000] timestep passed through unchanged.
- I2V forward: first-frame cond/mask from the batch stash, Siglip embed
  passthrough, zeros fallback when the Siglip cache is missing, hard error
  without the first-frame stash.
- attach_conditioning stashes the CLEAN frame-0 latent (i2v only).
- encode_text: Qwen chat-template path (hidden_states[-3], crop 108) + ByT5
  glyph path (zero tensor + zero mask for quote-less captions).
"""

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import (
    Hv15Driver,
    encode_byt5_prompt,
    encode_qwen_prompt,
)


class _Defn:
    def __init__(self, mode: str = "t2v"):
        self.architecture_params = {"mode": mode, "transformer.num_layers": 1}
        self.lora_targetable_modules: list[str] = []


class _CaptureTransformer(torch.nn.Module):
    """Records forward kwargs; returns a velocity over the first 32 channels."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        return (hidden[:, :32].clone(),)


def _driver(mode: str) -> tuple[Hv15Driver, _CaptureTransformer]:
    driver = Hv15Driver(_Defn(mode), torch.device("cpu"))
    model = _CaptureTransformer()
    driver.assign_components({"unet": model})
    return driver, model


def _text4(b=1, l1=6, d1=16, l2=4, d2=8):
    return (
        torch.randn(b, l1, d1),
        torch.ones(b, l1, dtype=torch.int64),
        torch.zeros(b, l2, d2),
        torch.zeros(b, l2, dtype=torch.int64),
    )


# ── T2V forward ────────────────────────────────────────────────────────────


def test_t2v_forward_builds_65ch_zeros_and_zero_image_embeds():
    driver, model = _driver("t2v")
    noisy = torch.randn(2, 32, 3, 4, 4)
    t = torch.tensor([250.0, 900.0])

    out = driver.forward_pass(noisy, t, _text4(2), {})

    call = model.calls[0]
    x = call["hidden_states"]
    assert x.shape == (2, 65, 3, 4, 4)
    assert torch.equal(x[:, :32], noisy)
    assert torch.all(x[:, 32:] == 0)  # zero cond latents + zero mask
    img = call["image_embeds"]
    assert img.shape == (2, 729, 1152)
    assert torch.all(img == 0)
    # RAW [0, 1000] timestep — never divided in the forward.
    assert torch.equal(call["timestep"], t)
    assert call["return_dict"] is False
    assert out.shape == noisy.shape


def test_forward_passes_both_text_streams():
    driver, model = _driver("t2v")
    emb, mask, emb2, mask2 = _text4(1)
    driver.forward_pass(torch.randn(1, 32, 1, 4, 4), torch.tensor([500.0]),
                        (emb, mask, emb2, mask2), {})
    call = model.calls[0]
    assert torch.equal(call["encoder_hidden_states"], emb)
    assert torch.equal(call["encoder_attention_mask"], mask)
    assert torch.equal(call["encoder_hidden_states_2"], emb2)
    assert torch.equal(call["encoder_attention_mask_2"], mask2)


def test_forward_rejects_non_tuple_text():
    driver, _ = _driver("t2v")
    with pytest.raises(ValueError, match="4-tuple"):
        driver.forward_pass(
            torch.randn(1, 32, 1, 4, 4), torch.tensor([500.0]),
            torch.randn(1, 6, 16), {},
        )


# ── I2V forward ────────────────────────────────────────────────────────────


def test_i2v_forward_uses_stashed_first_frame_and_siglip_embed():
    driver, model = _driver("i2v")
    noisy = torch.randn(1, 32, 3, 4, 4)
    first = torch.randn(1, 32, 1, 4, 4)
    siglip = torch.randn(1, 729, 1152)
    batch = {
        Hv15Driver.BATCH_FIRST_FRAME_LATENT: first,
        Hv15Driver.BATCH_IMAGE_EMBED: siglip,
    }

    driver.forward_pass(noisy, torch.tensor([500.0]), _text4(1), batch)

    call = model.calls[0]
    x = call["hidden_states"]
    assert x.shape == (1, 65, 3, 4, 4)
    assert torch.equal(x[:, :32], noisy)
    # cond channels: frame 0 = the stash, frames 1: zero.
    assert torch.equal(x[:, 32:64, 0], first[:, :, 0])
    assert torch.all(x[:, 32:64, 1:] == 0)
    # mask channel: 1 at frame 0, 0 after.
    assert torch.all(x[:, 64, 0] == 1.0)
    assert torch.all(x[:, 64, 1:] == 0.0)
    assert torch.allclose(call["image_embeds"], siglip)


def test_i2v_forward_missing_image_embed_falls_back_to_zeros():
    driver, model = _driver("i2v")
    batch = {Hv15Driver.BATCH_FIRST_FRAME_LATENT: torch.randn(1, 32, 1, 4, 4)}
    driver.forward_pass(
        torch.randn(1, 32, 3, 4, 4), torch.tensor([500.0]), _text4(1), batch
    )
    img = model.calls[0]["image_embeds"]
    assert img.shape == (1, 729, 1152)
    assert torch.all(img == 0)


def test_i2v_forward_requires_first_frame_stash():
    driver, _ = _driver("i2v")
    with pytest.raises(ValueError, match="first-frame latent"):
        driver.forward_pass(
            torch.randn(1, 32, 3, 4, 4), torch.tensor([500.0]), _text4(1), {}
        )


# ── attach_conditioning ────────────────────────────────────────────────────


def test_attach_conditioning_stashes_clean_frame0_for_i2v():
    driver, _ = _driver("i2v")
    latents = torch.randn(2, 32, 5, 4, 4)
    batch: dict = {}
    driver.attach_conditioning(batch, latents)
    stash = batch[Hv15Driver.BATCH_FIRST_FRAME_LATENT]
    assert stash.shape == (2, 32, 1, 4, 4)
    assert torch.equal(stash[:, :, 0], latents[:, :, 0])
    # Detached clone — mutating the source must not leak into the stash.
    latents[:, :, 0] += 1.0
    assert not torch.equal(stash[:, :, 0], latents[:, :, 0])


def test_attach_conditioning_noop_for_t2v():
    driver, _ = _driver("t2v")
    batch: dict = {}
    driver.attach_conditioning(batch, torch.randn(1, 32, 5, 4, 4))
    assert batch == {}


def test_prepare_latents_lifts_still_to_one_frame():
    driver, _ = _driver("t2v")
    still = torch.randn(2, 32, 8, 8)
    assert driver.prepare_latents(still).shape == (2, 32, 1, 8, 8)
    clip = torch.randn(2, 32, 3, 8, 8)
    assert driver.prepare_latents(clip).shape == (2, 32, 3, 8, 8)


# ── Dual-TE encode (fakes) ────────────────────────────────────────────────


class _FakeQwenTokenizer:
    """apply_chat_template fake — pads to max_length, records the call."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append({"conversations": conversations, **kwargs})
        b = len(conversations)
        length = kwargs["max_length"]
        return SimpleNamespace(
            input_ids=torch.arange(b * length).reshape(b, length),
            attention_mask=torch.ones(b, length, dtype=torch.int64),
        )


class _FakeQwenTE(torch.nn.Module):
    """hidden_states[i] is a constant-``i`` tensor → pins the [-3] selection."""

    def __init__(self, num_layers=5, d=16):
        super().__init__()
        self.num_layers = num_layers
        self.d = d

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=None):
        b, length = input_ids.shape
        hidden_states = [
            torch.full((b, length, self.d), float(i)) for i in range(self.num_layers)
        ]
        return SimpleNamespace(hidden_states=hidden_states)


def test_encode_qwen_uses_layer_minus3_and_crops_108():
    tok = _FakeQwenTokenizer()
    te = _FakeQwenTE(num_layers=5, d=16)
    emb, mask = encode_qwen_prompt(
        te, tok, ["a caption", ""], torch.device("cpu"), torch.float32
    )
    # Padded to 1000 + 108, cropped back to 1000.
    assert emb.shape == (2, 1000, 16)
    assert mask.shape == (2, 1000)
    # hidden_states[-(2+1)] == index 2 of 5 layers.
    assert torch.all(emb == 2.0)
    call = tok.calls[0]
    assert call["max_length"] == 1108
    assert call["add_generation_prompt"] is True
    assert call["padding"] == "max_length"
    # Chat template: system message first, empty user prompt becomes " ".
    convo = call["conversations"][1]
    assert convo[0]["role"] == "system"
    assert convo[1] == {"role": "user", "content": " "}


class _FakeByT5Batch(dict):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class _FakeByT5Tokenizer:
    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        return _FakeByT5Batch(
            torch.ones(1, length, dtype=torch.int64),
            torch.ones(1, length, dtype=torch.int64),
        )


class _FakeByT5TE(torch.nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.config = SimpleNamespace(d_model=d_model)

    def forward(self, input_ids=None, attention_mask=None):
        b, length = input_ids.shape
        return (torch.full((b, length, self.config.d_model), 7.0),)


def test_encode_byt5_zero_path_without_quotes():
    emb2, mask2 = encode_byt5_prompt(
        _FakeByT5TE(), _FakeByT5Tokenizer(), ["no quoted text"],
        torch.device("cpu"), torch.float32,
    )
    assert emb2.shape == (1, 256, 8)
    assert mask2.shape == (1, 256)
    assert torch.all(emb2 == 0) and torch.all(mask2 == 0)
    assert mask2.dtype == torch.int64


def test_encode_byt5_real_path_with_quotes():
    emb2, mask2 = encode_byt5_prompt(
        _FakeByT5TE(), _FakeByT5Tokenizer(), ['a sign "OPEN"'],
        torch.device("cpu"), torch.float32,
    )
    assert torch.all(emb2 == 7.0)
    assert torch.all(mask2 == 1)


def test_driver_encode_text_returns_4_tuple():
    driver = Hv15Driver(_Defn("t2v"), torch.device("cpu"))
    driver.assign_components(
        {
            "text_encoder": _FakeQwenTE(num_layers=5, d=16),
            "tokenizer": _FakeQwenTokenizer(),
            "text_encoder_2": _FakeByT5TE(d_model=8),
            "tokenizer_2": _FakeByT5Tokenizer(),
        }
    )
    out = driver.encode_text(["cap without quotes"], torch.float32)
    assert isinstance(out, tuple) and len(out) == 4
    emb, mask, emb2, mask2 = out
    assert emb.shape == (1, 1000, 16)
    assert mask.shape == (1, 1000)
    assert emb2.shape == (1, 256, 8)
    assert mask2.shape == (1, 256)
