"""qwen_image2 trainer, driver, sampler and saver on tiny real models (LANE-132 Task 3d).

Pinned, each against upstream diffusers ``6256aa7666`` (``pipeline_qwenimage21.py``):

* the transformer returns the WHOLE text-plus-image sequence and the caller
  keeps the last image-token positions (upstream line 783,
  ``noise_pred[:, -latents.size(1):]``); the driver's prediction is that
  slice, unpacked, and the model sees the timestep divided by 1000 ONCE
  (LESSONS: flow-match timestep scale) with upstream's ``img_mask`` layout;
* a tiny training run on CPU through the real PEFT path lowers the loss over
  ``3`` steps;
* the saved LoRA loads through diffusers' ``QwenImageLoraLoaderMixin`` -- the
  mixin ``QwenImage21Pipeline`` inherits (spec scope 7) -- with ``0``
  unexpected and ``0`` missing keys, one adapted module per A/B pair, and the
  saved weights applied;
* the saved file's ``modelspec.license`` is the definition's licence (spec
  scenario 4), and a definition without one writes no key;
* sampling keeps a float32 trajectory and runs the DiT with autocast OFF even
  under an outer autocast (LESSONS: autocast sampler collapse);
* ``torch.compile`` is declared off (upstream #14821).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import structlog
import torch
from safetensors import safe_open

from app.engine.models.registry import registry
from app.engine.strategies.noise_interpolation import NoiseInterpolation

_TINY = pathlib.Path(__file__).resolve().parent / "fixtures" / "qwen_image2" / "tiny.py"
_spec = importlib.util.spec_from_file_location("qwen_image2_tiny", _TINY)
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

C = 8  # tiny latent channels
LICENCE = "qwen-research (non-commercial)"


def _definition():
    registry.initialize()
    defn = registry.get_definition("qwen-image-2.1")
    assert defn is not None
    return defn


def _trainer(transformer, *, vae=None, config=None):
    from app.engine.models.families.qwen_image2.driver import QwenImage2Driver
    from app.engine.models.families.qwen_image2.trainer import QwenImage2Trainer

    defn = _definition()
    t = object.__new__(QwenImage2Trainer)
    t.definition = defn
    t.device = torch.device("cpu")
    t.config = {"network_rank": 4, "network_alpha": 4, **(config or {})}
    t.logger = structlog.get_logger("test_qwen_image2_training")
    t.text_cache = {}
    t.driver = QwenImage2Driver(defn, t.device)
    t.components = {"unet": transformer, "vae": vae, "text_encoder": None, "tokenizer": None}
    t._assign_components()
    return t


def _text(batch: int = 2, length: int = 5):
    torch.manual_seed(1)
    emb = torch.randn(batch, length, tiny.TEXT_HIDDEN)
    mask = torch.ones(batch, length, dtype=torch.long)
    mask[-1, -2:] = 0  # a shorter caption, right-padded
    emb[-1, -2:] = 0
    return emb, mask


def _upstream_prediction(transformer, x, t, emb, mask):
    """Upstream's calling convention, written out: pack, append one image slot
    per 2x2 latent group to the text mask, ``timestep / 1000``, keep the LAST
    image tokens (``pipeline_qwenimage21.py`` 753-783), unpack."""
    b, c, h, w = x.shape
    packed = x.reshape(b, c, h * w).transpose(1, 2)
    img_mask = torch.cat(
        [torch.zeros(b, emb.shape[1], dtype=torch.bool), torch.ones(b, h * w // 4, dtype=torch.bool)], dim=1,
    )
    out = transformer(
        hidden_states=packed, timestep=t / 1000, encoder_hidden_states=emb,
        encoder_hidden_states_mask=mask, img_shapes=[[(1, h, w)]] * b, img_mask=img_mask,
        return_dict=False,
    )[0]
    return out, out[:, -h * w :].transpose(1, 2).reshape(b, c, h, w)


# ── Forward pass ────────────────────────────────────────────────────────────


def test_forward_pass_predicts_from_the_last_image_tokens_at_one_timestep_scale():
    tr = tiny.tiny_transformer(in_channels=C).eval()
    t = _trainer(tr)
    torch.manual_seed(0)
    x = torch.randn(2, C, 4, 4)
    ts = torch.tensor([500.0, 250.0])
    emb, mask = _text()

    with torch.no_grad():
        pred = t.driver.forward_pass(x, ts, (emb, mask), {})
        full, expected = _upstream_prediction(tr, x, ts, emb, mask)

    assert pred.shape == x.shape
    assert full.shape[1] > 16, "the transformer returned no text positions to slice off"
    assert torch.equal(pred, expected)


# ── Training ────────────────────────────────────────────────────────────────


def test_tiny_training_through_the_real_peft_path_lowers_the_loss():
    t = _trainer(tiny.tiny_transformer(in_channels=C))
    t._apply_peft()
    model = t.driver.get_primary_model()
    assert model is t.transformer and model is t.components["unet"]
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "PEFT attached no trainable parameters"
    opt = torch.optim.Adam(params, lr=1e-2)
    interp = NoiseInterpolation("linear")

    torch.manual_seed(0)
    latents, noise = torch.randn(1, C, 4, 4), torch.randn(1, C, 4, 4)
    ts = torch.tensor([600.0])
    emb, mask = _text(batch=1)
    losses = []
    for _ in range(3):
        noisy = interp.add_noise(latents, noise, ts)
        pred = t.driver.forward_pass(noisy, ts, (emb, mask), {})
        loss = torch.nn.functional.mse_loss(pred, t.driver.compute_target(latents, noise, ts))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[2] < losses[0], f"loss did not fall over 3 steps: {losses}"


def test_compile_is_declared_off():
    tr = tiny.tiny_transformer(in_channels=C)
    tr._fp8_training_mode = True  # the flag that makes the base compile
    t = _trainer(tr)
    t._compile_if_quantized()
    assert t.driver.get_primary_model() is tr


# ── Saver ───────────────────────────────────────────────────────────────────


def _trained_lora_file(tmp_path, *, licence=LICENCE) -> pathlib.Path:
    from app.engine.models.families.qwen_image2.saver import QwenImage2Saver

    t = _trainer(tiny.tiny_transformer(in_channels=C))
    t._apply_peft()
    model = t.driver.get_primary_model()
    torch.manual_seed(3)
    with torch.no_grad():  # non-zero B, so "applied" is observable
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.1)
    path = tmp_path / "lora.safetensors"
    saver = t.driver.get_saver() if licence == LICENCE else QwenImage2Saver(license_text=licence)
    saver.save({"unet": model, "config": {"save_precision": "fp32"}}, path)
    assert path.exists(), "the saver wrote no file"
    return path


def test_saved_lora_loads_through_the_qwen_lora_loader_mixin(tmp_path):
    from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin
    from peft.utils import get_peft_model_state_dict

    path = _trained_lora_file(tmp_path)
    with safe_open(str(path), "pt") as f:
        saved = {k: f.get_tensor(k) for k in f.keys()}
    pairs = sum(1 for k in saved if ".lora_A." in k)
    assert pairs > 0 and pairs == sum(1 for k in saved if ".lora_B." in k)

    class _Consumer(QwenImageLoraLoaderMixin):
        """The mixin as ``QwenImage21Pipeline`` carries it, on a bare transformer."""

        hf_device_map = None  # a DiffusionPipeline attribute the loader reads

        def __init__(self, transformer):
            self.transformer = transformer
            self.components = {"transformer": transformer}  # DiffusionPipeline.components

    fresh = tiny.tiny_transformer(in_channels=C)
    _Consumer(fresh).load_lora_weights(str(path), adapter_name="uat")

    adapted = [n for n, m in fresh.named_modules() if hasattr(m, "lora_A") and "uat" in m.lora_A]
    assert len(adapted) == pairs, f"{len(adapted)} modules adapted for {pairs} A/B pairs"

    # What the consumer reads out of the file (after its own key conversion),
    # set against what landed in the model: 0 unexpected, 0 missing.
    consumed = QwenImageLoraLoaderMixin.lora_state_dict(str(path))
    consumed = consumed[0] if isinstance(consumed, tuple) else consumed
    consumed = {k.removeprefix("transformer."): v for k, v in consumed.items()}
    loaded = get_peft_model_state_dict(fresh, adapter_name="uat")
    unexpected = sorted(set(consumed) - set(loaded))
    missing = sorted(set(loaded) - set(consumed))
    assert unexpected == [] and missing == [], f"unexpected {unexpected[:4]}, missing {missing[:4]}"
    for k, v in loaded.items():
        assert torch.equal(v, consumed[k]), f"{k} was not applied as saved"
    # The file is in the diffusers layout the mixin applies UNCONVERTED. The
    # mixin also converts a ComfyUI ``diffusion_model.`` file (measured: that
    # mutant loads with 0 unexpected / 0 missing), so the layout is pinned on
    # its own -- spec scope 7 names the diffusers PEFT format.
    off_layout = [k for k in saved if not k.startswith("transformer.")]
    assert off_layout == [], f"keys outside the diffusers layout: {off_layout[:2]}"


def test_saved_file_records_the_definition_licence(tmp_path):
    path = _trained_lora_file(tmp_path)
    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
    assert _definition().license == LICENCE
    assert meta.get("modelspec.license") == LICENCE


def test_a_definition_without_a_licence_writes_no_licence_key(tmp_path):
    path = _trained_lora_file(tmp_path, licence=None)
    with safe_open(str(path), "pt") as f:
        assert "modelspec.license" not in f.metadata()


# ── Sampler ─────────────────────────────────────────────────────────────────


def test_sampler_keeps_a_float32_trajectory_with_autocast_off():
    from app.engine.models.families.qwen_image2.sampler import QwenImage2Sampler

    tr = tiny.tiny_transformer(in_channels=64).to(torch.bfloat16)
    t = _trainer(tr, vae=tiny.tiny_vae())
    sampler = QwenImage2Sampler(t)

    autocast_seen: list[bool] = []
    tr.register_forward_pre_hook(lambda m, a: autocast_seen.append(torch.is_autocast_enabled("cpu")))
    emb, mask = _text(batch=1)
    gen = torch.Generator().manual_seed(42)
    noise = sampler._create_initial_noise(64, 64, gen)
    assert noise.dtype == torch.float32

    with torch.autocast("cpu", dtype=torch.bfloat16):  # an outer autocast must not leak in
        latents = sampler.denoise(noise, {"embeds": emb, "mask": mask}, 3, 1.0, 42)

    assert autocast_seen and not any(autocast_seen)
    assert latents["latents"].dtype == torch.float32
    assert torch.isfinite(latents["latents"]).all()
    image = sampler.decode_latents(latents)
    assert image.size == (64, 64) and image.mode == "RGB"


@pytest.mark.parametrize("h,w", [(64, 64), (96, 64)])
def test_sampler_noise_is_the_upstream_token_grid(h, w):
    from app.engine.models.families.qwen_image2 import utils
    from app.engine.models.families.qwen_image2.sampler import QwenImage2Sampler

    t = _trainer(tiny.tiny_transformer(in_channels=64), vae=tiny.tiny_vae())
    noise = QwenImage2Sampler(t)._create_initial_noise(w, h, torch.Generator().manual_seed(0))
    lh, lw = utils.latent_grid(h, w)
    assert tuple(noise.shape) == (1, lh * lw, 64)
