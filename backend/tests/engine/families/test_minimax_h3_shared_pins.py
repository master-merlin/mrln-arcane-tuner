"""The model-generic MiniMax-H3 code lives in ``minimax_h3_shared``, and moving
it there changed nothing the video family produces.

Two kinds of test sit here:

* **Pins** (written and green on the code before the move): the video family's
  text-cache key and cache scope, its latent-cache fingerprint, and the LoRA
  artifact (key set, shapes and the exact float32 bytes) for a fixed tiny PEFT
  model. The strings and digests are the values the video code produced before
  the shared package existed. A change to any of them means every cached
  embedding or latent of an existing user is rebuilt, or a LoRA file changes,
  so a failure here is never fixed by editing the pinned value. (The LoRA
  digests were recomputed once, when the fill stopped using ``torch.randn``,
  whose bytes differ between torch builds: the new values were computed on the
  video code before the move and on the code after it, and the two agreed.)
* **The shared package itself**: the video family uses its pieces rather than
  keeping copies, and each piece takes what differs per family (the cache-key
  prefix, the native module map) as an argument.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

_FAMILIES = Path(__file__).resolve().parents[3] / "app" / "engine" / "models" / "families"

# ── The pinned values (computed on the video family's code before the move) ──

_PROMPT = "a cat sits"
_PINNED_TE_KEYS = {
    "minimax-h3-t2va": "minimax_h3|minimax-h3-t2va|tap=50|tok=75755e838770207b|a cat sits",
    "minimax-h3-fl2va": "minimax_h3|minimax-h3-fl2va|tap=50|tok=75755e838770207b|a cat sits",
    "minimax-h3-ref2va": "minimax_h3|minimax-h3-ref2va|tap=50|tok=75755e838770207b|a cat sits",
}
_PINNED_TE_SCOPE = "tap50-75755e838770207b"
_PINNED_LATENT_FINGERPRINT = "fc26cc06"

# key -> (shape, sha256[:16] of the float32 bytes)
_PINNED_ARTIFACT: dict[str, tuple[tuple[int, ...], str]] = {
    "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": ((2, 32), "244eed92f5d37426"),
    "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": ((16, 2), "7b88c87a0a90e4cf"),
    "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": ((6, 16), "c6697023247c5385"),
    "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": ((96, 6), "fb026cc6b43dabc5"),
    "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": ((2, 16), "4a5750a304704e26"),
    "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": ((64, 2), "2cb6c6a93a24c5bc"),
    "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": ((2, 32), "2e43336306c3f457"),
    "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": ((16, 2), "684b7ffa2fc50d5c"),
    "diffusion_model.blocks.1.attn.out_proj.lora_A.weight": ((2, 32), "cac0e8f84eb26b35"),
    "diffusion_model.blocks.1.attn.out_proj.lora_B.weight": ((16, 2), "746cb7dba3868415"),
    "diffusion_model.blocks.1.attn.qkv_proj.lora_A.weight": ((6, 16), "f760344981417ecf"),
    "diffusion_model.blocks.1.attn.qkv_proj.lora_B.weight": ((96, 6), "39fc97a98e0e3c7c"),
    "diffusion_model.blocks.1.mlp.fc1.lora_A.weight": ((2, 16), "5d5e582477547779"),
    "diffusion_model.blocks.1.mlp.fc1.lora_B.weight": ((64, 2), "a5591baf9e9660c6"),
    "diffusion_model.blocks.1.mlp.fc2.lora_A.weight": ((2, 32), "14f4a65b49966119"),
    "diffusion_model.blocks.1.mlp.fc2.lora_B.weight": ((16, 2), "74c1f74853d14113"),
    "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_A.weight": ((2, 32), "82c05a4829ae95bc"),
    "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight": ((16, 2), "60dd51421b652e52"),
    "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight": ((6, 16), "d130bbc67fb64cb6"),
    "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight": ((96, 6), "ae669e4e38dcaabb"),
    "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight": ((2, 16), "11dc78b64aa4db22"),
    "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight": ((64, 2), "8d35650cb0d75251"),
    "diffusion_model.token_refiner.blocks.0.mlp.fc2.lora_A.weight": ((2, 32), "cc97ab94d1adc86a"),
    "diffusion_model.token_refiner.blocks.0.mlp.fc2.lora_B.weight": ((16, 2), "5a0a70d555293491"),
}

# The tiny video transformer, written out here rather than imported from a
# conftest, so the pin cannot move when a shared fixture does.
_TINY_VIDEO_KWARGS = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 16,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 32,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 16,
    "freq_dim": 16,
    "time_embed_hidden_dim": 16,
    "time_embed_dim": 8,
    "rope_freq_dim": 2,
    "rope_theta": 10000.0,
}


class _Tokenizer:
    """A whitespace tokenizer with a fixed vocabulary (the fingerprint input)."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        assert add_special_tokens is False
        return {"input_ids": [len(w) + 1 for w in text.split()]}

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "a": 1, "cat": 2, "sits": 3}


class _Processor:
    tokenizer = _Tokenizer()


class _VisualVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = {"latent_channels": 32, "spatial_compression_ratio": 32}
        self.p = torch.nn.Linear(1, 1)


@pytest.fixture(scope="module")
def definitions():
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return dict(ModelRegistry._definitions)


def _video_driver(definition):
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver

    driver = MiniMaxH3Driver(definition, torch.device("cpu"))
    driver.assign_components({"tokenizer": _Processor()})
    return driver


# ── Pins: the video family's outputs are the ones it produced before the move ─


@pytest.mark.parametrize("def_id", sorted(_PINNED_TE_KEYS))
def test_video_te_cache_key_is_unchanged(definitions, def_id):
    driver = _video_driver(definitions[def_id])
    assert driver.te_cache_key(_PROMPT) == _PINNED_TE_KEYS[def_id], (
        "the video family's text-cache key changed: every cached embedding would be rebuilt"
    )
    assert driver.te_cache_scope() == _PINNED_TE_SCOPE


def test_video_latent_cache_fingerprint_is_unchanged(definitions):
    from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE

    driver = _video_driver(definitions["minimax-h3-t2va"])
    driver.assign_components({"vae": H3PixelAdaptedVAE(_VisualVAE())})
    assert driver.latent_cache_fingerprint() == _PINNED_LATENT_FINGERPRINT, (
        "the video family's latent-cache fingerprint changed: every cached latent would be rebuilt"
    )


def _lora_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """The LoRA weights of a PEFT model, in a fixed order (sorted by name)."""
    return [(n, p) for n, p in sorted(model.named_parameters()) if ".lora_A." in n or ".lora_B." in n]


def _video_peft_model(definitions):
    from diffusers import MiniMaxH3Transformer3DModel
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    targets = list(definitions["minimax-h3-t2va"].lora_targetable_modules)
    model = get_peft_model(
        MiniMaxH3Transformer3DModel(**_TINY_VIDEO_KWARGS),
        LoraConfig(r=2, lora_alpha=4.0, target_modules=targets),
    )
    _pin_fill(model)
    return model


def _pin_fill(model) -> None:
    """Every LoRA weight, in a fixed order, gets the next slice of one ramp of
    distinct multiples of 1/64 centred on zero: exact in float32 and the same
    bytes on every platform and torch build. Never a random-number kernel:
    ``torch.randn`` bytes differ between builds (the Linux CPU wheel and the
    Windows CUDA wheel disagree), and the pin would then pin that kernel
    instead of the artifact code. Distinct values keep the pin sensitive to
    the qkv slot order, the fc1 row swap and the scaling fold."""
    params = _lora_params(model)
    mid = sum(p.numel() for _, p in params) // 2
    offset = 0
    with torch.no_grad():
        for _, p in params:
            ramp = (torch.arange(p.numel(), dtype=torch.int64) + offset - mid).to(torch.float32) / 64
            p.copy_(ramp.reshape(p.shape))
            offset += p.numel()


def _video_artifact_of(model) -> dict[str, torch.Tensor]:
    from app.engine.models.families.minimax_h3.lora_keys import adapters_from_peft_model, build_artifact

    return build_artifact(adapters_from_peft_model(model))


def _video_artifact(definitions) -> dict[str, torch.Tensor]:
    return _video_artifact_of(_video_peft_model(definitions))


def test_video_lora_pin_inputs_are_exact_fractions(definitions):
    """The pinned digests hash exact values only, so they hold on every
    platform and torch build: each LoRA weight is its slice of one ramp of
    multiples of 1/64 (recomputed here independently), never the output of a
    random-number kernel, and the artifact keeps every value a multiple of
    1/64 (no rounding-sensitive op between the weights and the file)."""
    model = _video_peft_model(definitions)
    params = _lora_params(model)
    total = sum(p.numel() for _, p in params)
    ramp = (torch.arange(total, dtype=torch.float64) - total // 2) / 64
    start = 0
    for name, p in params:
        expected = ramp[start : start + p.numel()].reshape(p.shape).to(torch.float32)
        assert torch.equal(p.detach().cpu(), expected), f"{name}: not the deterministic ramp"
        start += p.numel()
    for key, tensor in _video_artifact_of(model).items():
        scaled = tensor.double() * 64
        assert torch.equal(scaled, scaled.round()), f"{key}: a value is not a multiple of 1/64"


def test_video_lora_artifact_is_unchanged(definitions):
    artifact = _video_artifact(definitions)
    assert set(artifact) == set(_PINNED_ARTIFACT), (
        f"key set changed: missing={sorted(set(_PINNED_ARTIFACT) - set(artifact))[:4]} "
        f"extra={sorted(set(artifact) - set(_PINNED_ARTIFACT))[:4]}"
    )
    for key, (shape, digest) in _PINNED_ARTIFACT.items():
        tensor = artifact[key]
        assert tuple(tensor.shape) == shape, f"{key}: shape {tuple(tensor.shape)} != {shape}"
        got = hashlib.sha256(tensor.float().contiguous().numpy().tobytes()).hexdigest()[:16]
        assert got == digest, f"{key}: the tensor's bytes changed"


# ── The shared package ───────────────────────────────────────────────────────


def test_shared_package_ships_no_definitions_and_no_family():
    shared = _FAMILIES / "minimax_h3_shared"
    assert (shared / "__init__.py").is_file()
    assert not (shared / "family.py").exists(), "a support package is never registered as a family"
    assert not list((shared / "definitions").glob("*.yaml"))


def test_shared_cache_key_takes_the_family_prefix(definitions):
    from app.engine.models.families.minimax_h3_shared import text

    driver = _video_driver(definitions["minimax-h3-t2va"])
    fingerprint = text.tokenizer_fingerprint(_Processor())
    assert text.te_cache_scope(50, fingerprint) == _PINNED_TE_SCOPE
    key = text.te_cache_key("minimax_h3|", "minimax-h3-t2va", 50, fingerprint, _PROMPT)
    assert key == _PINNED_TE_KEYS["minimax-h3-t2va"] == driver.te_cache_key(_PROMPT)
    other = text.te_cache_key("minimax_h3_image|", "minimax-h3-t2va", 50, fingerprint, _PROMPT)
    assert other != key and other.startswith("minimax_h3_image|"), "the prefix must separate two families' caches"
    source = inspect.getsource(text)
    assert '"minimax_h3|' not in source and "'minimax_h3|" not in source, "the prefix is passed in, never hardcoded"


def test_shared_text_tap_encodes_raw_tokens_at_the_tap():
    from app.engine.models.families.minimax_h3_shared import text
    from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL

    te = StubQwen3VL(num_hidden_layers=4)
    out = text.encode_captions(
        StubProcessor(), te, ["a cat sits", "", "w " * 9], torch.bfloat16,
        tap=3, max_tokens=5, device=torch.device("cpu"),
    )
    assert out.embeddings.dtype is torch.bfloat16
    assert out.attention_mask.sum(dim=1).tolist() == [3, 1, 5]
    assert torch.all(out.embeddings[out.attention_mask.bool()].float() == 3.0), "not the tap's hidden state"
    with pytest.raises(ValueError, match="tap_index=9"):
        text.encode_captions(
            StubProcessor(), te, ["a"], torch.float32, tap=9, max_tokens=5, device=torch.device("cpu"),
        )


def test_shared_flow_is_the_inverted_convention():
    from app.engine.models.families.minimax_h3_shared import flow

    x0 = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    t = torch.tensor([1.0, 0.25])
    xt = flow.add_noise(x0, noise, t)
    assert torch.equal(xt[0], x0[0]), "t = 1 is clean"
    assert torch.allclose(xt[1], 0.25 * x0[1] + 0.75 * noise[1])
    assert torch.equal(flow.velocity_target(x0, noise), x0 - noise)
    sigma = torch.tensor([0.0, 0.3, 1.0])
    assert torch.equal(flow.sigma_to_t(sigma), 1.0 - sigma)
    assert torch.equal(flow.t_to_sigma(flow.sigma_to_t(sigma)), sigma)


def test_shared_pixel_adapter_handles_images_and_video():
    from app.engine.models.families.minimax_h3_shared.pixel_adapter import IMAGENET_MEAN, H3PixelAdaptedVAE

    class _Echo(torch.nn.Module):
        def encode(self, x):
            return x

    vae = H3PixelAdaptedVAE(_Echo())
    for shape in ((1, 3, 4, 4), (1, 3, 2, 4, 4)):
        out = vae.encode(-torch.ones(shape))  # black pixels
        assert torch.allclose(out[0, 0].flatten()[0], torch.tensor(-IMAGENET_MEAN[0] / 0.229))


def test_shared_lora_core_takes_the_module_map():
    from app.engine.models.families.minimax_h3_shared.lora_artifact import ARTIFACT_PREFIX, Adapter, build_artifact

    g = torch.Generator().manual_seed(0)

    def _adapter(out_features: int, scaling: float = 2.0) -> Adapter:
        return Adapter(
            lora_A=torch.randn(2, 8, generator=g), lora_B=torch.randn(out_features, 2, generator=g), scaling=scaling,
        )

    adapters = {
        "blocks.0.attn.to_q": _adapter(8),
        "blocks.0.attn.to_k": _adapter(8),
        "blocks.0.attn.to_v": _adapter(8),
        "blocks.0.fc1": _adapter(6),
    }
    native = {
        "blocks.0.attn.to_q": ("blocks.0.attn.qkv_proj", "q"),
        "blocks.0.attn.to_k": ("blocks.0.attn.qkv_proj", "k"),
        "blocks.0.attn.to_v": ("blocks.0.attn.qkv_proj", "v"),
        "blocks.0.fc1": ("blocks.0.mlp.fc1", "rename"),
    }
    out = build_artifact(adapters, native.__getitem__)
    assert ARTIFACT_PREFIX == "diffusion_model."
    a = out["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
    b = out["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    delta = b @ a
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        ad = adapters[f"blocks.0.attn.{proj}"]
        assert torch.allclose(delta[i * 8 : (i + 1) * 8], ad.scaling * ad.lora_B @ ad.lora_A, atol=1e-6)
    fc1 = adapters["blocks.0.fc1"]
    assert torch.equal(out["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"], fc1.lora_B * 2.0), (
        "a plain rename keeps the row order; the scale is folded into B"
    )


def test_video_family_uses_the_shared_pieces_and_keeps_no_copies():
    from app.engine.models.families.minimax_h3 import lora_keys, pixel_adapter
    from app.engine.models.families.minimax_h3_shared import lora_artifact
    from app.engine.models.families.minimax_h3_shared import pixel_adapter as shared_pixel

    assert pixel_adapter.H3PixelAdaptedVAE is shared_pixel.H3PixelAdaptedVAE
    assert pixel_adapter.IMAGENET_MEAN == shared_pixel.IMAGENET_MEAN
    assert pixel_adapter.IMAGENET_STD == shared_pixel.IMAGENET_STD
    assert lora_keys.Adapter is lora_artifact.Adapter
    assert lora_keys.ARTIFACT_PREFIX == lora_artifact.ARTIFACT_PREFIX
    video = _FAMILIES / "minimax_h3"
    copies = {
        "pixel_adapter.py": ("class H3PixelAdaptedVAE", "IMAGENET_MEAN: tuple"),
        "lora_keys.py": ("get_peft_model_state_dict", "class Adapter", "b_fused = torch.zeros"),
        "driver.py": ("hashlib.sha256()", "output_hidden_states=True", "return latents - noise"),
        "trainer.py": ("import psutil", "cuda_allocated_after_bytes"),
    }
    for name, needles in copies.items():
        src = (video / name).read_text(encoding="utf-8")
        left = [n for n in needles if n in src]
        assert not left, f"minimax_h3/{name} still carries its own copy: {left}"
