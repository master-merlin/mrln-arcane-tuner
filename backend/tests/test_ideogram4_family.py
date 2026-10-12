"""Smoke tests for the Ideogram 4 model family (no weights downloaded)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry


def test_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("ideogram4")
    assert family_cls.family_name == "ideogram4"


def test_family_returns_trainer_class():
    from app.engine.models.families.ideogram4.family import IdeogramV4Family
    from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer

    definition = ModelDefinition(id="x", family="ideogram4", name="X")
    family = IdeogramV4Family(definition, {})
    assert family.get_trainer_class() is IdeogramV4Trainer


def test_training_uses_ideogram_flow_convention():
    """Ideogram 4 DiT convention: t=0 -> noise, t=1 -> data, velocity = data - noise.

    The pretrained DiT (proven by the upstream sampler) integrates z=randn at
    t~0 up to data at t~1 with velocity = data - noise. The generic trainer's
    default is the OPPOSITE (add_noise t=1->noise, target = noise - latents).
    Training in the default convention trains the LoRA against a flipped-time,
    sign-negated target vs the frozen base -> ~2-3 loss and white samples.
    This pins the corrected convention.
    """
    from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer

    tr = IdeogramV4Trainer(ModelDefinition(id="x", family="ideogram4", name="X"), {})
    data = torch.full((1, 4, 8), 5.0)
    noise = torch.full((1, 4, 8), -3.0)

    # Target is the velocity data - noise (NOT the base default noise - latents).
    tgt = tr.compute_target(data, noise, torch.tensor([500.0]))
    assert torch.allclose(tgt, data - noise)

    # add_noise: t=0 -> all noise; t=1000 -> all data; t=500 -> 50/50 blend.
    assert torch.allclose(tr.add_noise(data, noise, torch.tensor([0.0])), noise)
    assert torch.allclose(tr.add_noise(data, noise, torch.tensor([1000.0])), data)
    assert torch.allclose(
        tr.add_noise(data, noise, torch.tensor([500.0])), 0.5 * noise + 0.5 * data
    )


def test_registry_dispatches_to_trainer():
    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.registry import ModelRegistry
    from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer

    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("ideogram4")
    family = family_cls(ModelDefinition(id="x", family="ideogram4", name="X"), {})
    assert family.get_trainer_class() is IdeogramV4Trainer


def test_dequantize_fp8_state_dict_applies_scale():
    from app.engine.models.families.ideogram4.utils import dequantize_fp8_state_dict

    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([2.0, 0.5], dtype=torch.float32)
    sd = {"blk.weight": w, "blk.weight_scale": scale}

    out = dequantize_fp8_state_dict(sd)

    assert "blk.weight_scale" not in out          # scale consumed
    expected = w.to(torch.float32) * scale[:, None]
    assert torch.allclose(out["blk.weight"].float(), expected)


def test_dequantize_passes_through_unscaled():
    from app.engine.models.families.ideogram4.utils import dequantize_fp8_state_dict

    sd = {"norm.weight": torch.ones(4)}
    out = dequantize_fp8_state_dict(sd)
    assert torch.allclose(out["norm.weight"], torch.ones(4))


def test_patchify_roundtrip():
    from app.engine.models.families.ideogram4.utils import (
        patchify_to_seq, unpatchify_from_seq,
    )
    x = torch.randn(2, 32, 16, 16)
    seq = patchify_to_seq(x)               # [2, (8*8), 128]
    assert seq.shape == (2, 64, 128)
    back = unpatchify_from_seq(seq, 8, 8)
    assert torch.allclose(back, x)


def test_patchify_matches_upstream_ordering():
    """Pin the token feature layout to upstream's ``(p_h, p_w, c)`` order.

    Upstream ``pipeline_ideogram4.py::_decode`` splits the 128-dim token as
    ``view(B, grid_h, grid_w, patch, patch, ae_channels)`` -> the feature dim
    is ordered ``(p_h, p_w, c)``. We build a latent whose value encodes its
    (channel, row, col) coordinate and assert the first token's flat vector is
    laid out patch-row-outer, patch-col-middle, channel-inner. This FAILS for
    the old ``(c, p_h, p_w)`` ordering.
    """
    from app.engine.models.families.ideogram4.utils import patchify_to_seq

    p = 2
    c = 3  # tiny channel count (real model uses 32); token dim = c*p*p = 12
    # value(channel, row, col) = channel*100 + row*10 + col, unique per cell.
    x = torch.zeros(1, c, p, p)
    for ch in range(c):
        for row in range(p):
            for col in range(p):
                x[0, ch, row, col] = ch * 100 + row * 10 + col

    seq = patchify_to_seq(x)  # [1, 1, c*p*p]
    assert seq.shape == (1, 1, c * p * p)
    token = seq[0, 0]

    # Upstream order: index = ((p_h * p) + p_w) * c + ch  -> (p_h, p_w, c)
    expected = torch.empty(c * p * p)
    idx = 0
    for ph in range(p):
        for pw in range(p):
            for ch in range(c):
                expected[idx] = ch * 100 + ph * 10 + pw
                idx += 1
    assert torch.equal(token, expected), (
        f"token layout {token.tolist()} != upstream (p_h,p_w,c) {expected.tolist()}"
    )

    # Guard: the WRONG (c, p_h, p_w) ordering must differ, so this test has teeth.
    wrong = torch.empty(c * p * p)
    idx = 0
    for ch in range(c):
        for ph in range(p):
            for pw in range(p):
                wrong[idx] = ch * 100 + ph * 10 + pw
                idx += 1
    assert not torch.equal(expected, wrong)


HID = 8  # tiny stand-in for Qwen3-VL hidden size


class _FakeTok:
    """Minimal Qwen3-VL-style tokenizer stub.

    Records the chat-template content shape and the ``add_special_tokens`` kwarg
    so parity tests can assert the driver follows the ai-toolkit path.
    """

    def __init__(self):
        self.last_messages = None
        self.last_add_special_tokens = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        assert add_generation_prompt is True
        assert tokenize is False
        # Render the typed-list content into a flat string (one token per word).
        content = messages[0]["content"]
        if isinstance(content, list):
            return " ".join(part["text"] for part in content)
        return content

    def __call__(self, texts, **kw):
        self.last_add_special_tokens = kw.get("add_special_tokens")
        n = max(len(t.split()) for t in texts) or 1
        return {
            "input_ids": torch.ones(len(texts), n, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), n, dtype=torch.long),
        }


class _FakeRotary(torch.nn.Module):
    def forward(self, hidden_states, position_ids):
        # Return a (cos, sin) pair shaped like real mRoPE embeddings; the fake
        # decoder layer ignores them, but the contract (a 2-tuple) is preserved.
        b = hidden_states.shape[0]
        length = hidden_states.shape[1]
        cos = torch.zeros(b, length, HID)
        return (cos, cos.clone())


class _FakeDecoderLayer(torch.nn.Module):
    """Identity-plus-bias decoder layer: output(k) = input + (k+1).

    The deterministic per-layer offset lets a test assert the driver captures
    the OUTPUT of layer ``k`` (not the input / not ``k+1``).
    """

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, **kw):
        return hidden_states + float(self.layer_idx + 1)


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()

        class _Cfg:
            _attn_implementation = "eager"

        self.config = _Cfg()
        self.embed_tokens = torch.nn.Embedding(8, HID)
        self.layers = torch.nn.ModuleList(
            [_FakeDecoderLayer(i) for i in range(n_layers)]
        )
        self.rotary_emb = _FakeRotary()

    def forward(self, *a, **kw):  # not used; manual loop drives the layers
        raise NotImplementedError


class _FakeQwen3VL(torch.nn.Module):
    """AutoModel-style wrapper exposing ``.language_model`` (like Qwen3VLModel)."""

    def __init__(self, n_layers):
        super().__init__()
        self.language_model = _FakeLanguageModel(n_layers)


def _make_text_driver(monkeypatch, n_layers):
    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.families.ideogram4 import driver as driver_mod
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    # The fake decoder layers don't consume the causal mask; stub it out so the
    # manual forward runs without a real Qwen3-VL config/attn backend.
    # The stub BINDS the installed signature first: a stale keyword fails here
    # (a bare ``lambda **kw: None`` accepted any keyword and hid a rename).
    import inspect

    sig = inspect.signature(driver_mod.create_causal_mask)

    def _bound_mask(**kw):
        sig.bind(**kw)
        return None

    monkeypatch.setattr(driver_mod, "create_causal_mask", _bound_mask)

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    drv = IdeogramV4Driver(defn, torch.device("cpu"))
    tok = _FakeTok()
    drv.text_encoder = _FakeQwen3VL(n_layers)
    drv.tokenizer = tok
    return drv, tok


def test_encode_text_concats_selected_layers(monkeypatch):
    from app.engine.models.families.ideogram4.utils import QWEN3VL_SELECTED_LAYERS

    n_layers = max(QWEN3VL_SELECTED_LAYERS) + 1  # 36 real layers
    drv, _ = _make_text_driver(monkeypatch, n_layers)

    out = drv.encode_text(["a cat sitting"], torch.float32)
    # Concat width is exactly len(selected) * hidden -- the manual layer path.
    assert out.embeddings.shape[-1] == len(QWEN3VL_SELECTED_LAYERS) * HID
    assert out.embeddings.shape[0] == 1
    assert out.attention_mask.dtype == torch.bool


def test_encode_text_captures_layer_outputs_directly(monkeypatch):
    """Index ``k`` must map to the OUTPUT of decoder layer ``k`` (no +1 offset).

    The fake layers add ``(layer_idx + 1)`` so the running activation after
    layer ``k`` equals ``embed + sum_{j<=k}(j+1)``. The reshaped feature for a
    selected layer ``k`` (innermost interleaved axis) must equal that cumulative
    sum -- proving direct layer-output capture, not the old HF ``[k+1]`` tap.
    """
    drv, _ = _make_text_driver(monkeypatch, n_layers=6)
    drv.selected_layers = (0, 2, 5)

    # Force embed_tokens to a known constant (0) so we can predict the sums.
    torch.nn.init.zeros_(drv.text_encoder.language_model.embed_tokens.weight)

    out = drv.encode_text(["a b c"], torch.float32)  # 3 tokens
    emb = out.embeddings[0]  # (L, HID * 3)
    L = emb.shape[0]
    # Interleaved layout: feature[:, j*? ] -> reshape was (L, HID, n) -> (L, HID*n)
    # so the last axis groups by (hidden_unit, layer): index = h*n + layer_pos.
    n = len(drv.selected_layers)
    reshaped = emb.reshape(L, HID, n)
    # cumulative output offset after layer k = sum_{j=0..k}(j+1)
    expected = {0: 1.0, 2: 1.0 + 2.0 + 3.0, 5: float(sum(j + 1 for j in range(6)))}
    for pos, k in enumerate(drv.selected_layers):
        assert torch.allclose(
            reshaped[:, :, pos], torch.full((L, HID), float(expected[k]))
        ), f"layer {k} feature should equal its OUTPUT activation {expected[k]}"


def test_encode_text_tokenization_matches_ai_toolkit(monkeypatch):
    """Parity guard: typed-list chat content + add_special_tokens=False.

    These two together are the ai-toolkit tokenization path; the old path used a
    bare-string content and add_special_tokens=True.
    """
    from app.engine.models.families.ideogram4.utils import QWEN3VL_SELECTED_LAYERS

    n_layers = max(QWEN3VL_SELECTED_LAYERS) + 1
    drv, tok = _make_text_driver(monkeypatch, n_layers)

    drv.encode_text(["a cat"], torch.float32)

    # add_special_tokens=False (Qwen3-VL has no BOS; the template emits specials).
    assert tok.last_add_special_tokens is False
    # content is the typed list [{"type": "text", "text": ...}], not a bare str.
    content = tok.last_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "a cat"


def test_latent_norm_roundtrip():
    from app.engine.models.families.ideogram4.utils import (
        denormalize_latents, normalize_latents,
    )
    x = torch.randn(2, 7, 128)
    back = denormalize_latents(normalize_latents(x))
    assert torch.allclose(back, x, atol=1e-5)


def test_latent_norm_constants_match_upstream():
    """Constants must equal upstream get_latent_norm() exactly (no approximation)."""
    from app.engine.models.families.ideogram4 import utils

    assert len(utils.LATENT_SHIFT) == 128
    assert len(utils.LATENT_SCALE) == 128
    # spot-check the documented upstream endpoints
    assert utils.LATENT_SHIFT[0] == 0.01984364
    assert utils.LATENT_SHIFT[-1] == -0.01760592
    assert utils.LATENT_SCALE[0] == 1.63933691
    assert utils.LATENT_SCALE[-1] == 1.68533454


def _make_driver_with_stub_dit(feat_dim=8, latent_h=4, latent_w=4):
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    captured = {}

    class _DiT(torch.nn.Module):
        # The release Ideogram4Transformer2DModel signature (diffusers 0.41.0,
        # transformer_ideogram4.py:374); ``dtype`` is what the driver casts to.
        dtype = torch.float32

        def forward(
            self, *, hidden_states, timestep, encoder_hidden_states, position_ids,
            segment_ids, indicator, return_dict=True,
        ):
            captured["t"] = timestep.detach().clone()
            captured["L"] = hidden_states.shape[1]
            captured["position_ids"] = position_ids.detach().clone()
            captured["segment_ids"] = segment_ids.detach().clone()
            captured["indicator"] = indicator.detach().clone()
            captured["llm_features"] = encoder_hidden_states.detach().clone()
            captured["x"] = hidden_states.detach().clone()
            out = torch.zeros(hidden_states.shape)
            return (out,) if not return_dict else {"sample": out}

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    drv = IdeogramV4Driver(defn, torch.device("cpu"))
    drv.transformer = _DiT()
    drv._latent_h, drv._latent_w = latent_h, latent_w
    return drv, captured


def test_forward_pass_timestep_scale_divides_by_1000():
    """Trainer passes [0,1000]; DiT wants [0,1]. Guards the prior ×1000 noise bug."""
    drv, captured = _make_driver_with_stub_dit(feat_dim=8, latent_h=4, latent_w=4)

    image_seq = torch.randn(1, 16, 128)            # S_img = 4*4 = 16
    feats = torch.randn(1, 5, 8)
    text = (feats, torch.ones(1, 5, dtype=torch.bool))
    ts = torch.tensor([500.0])                     # [0,1000] convention in

    out = drv.forward_pass(image_seq, ts, text, {"latent_h": 4, "latent_w": 4})

    assert torch.allclose(captured["t"], ts / 1000.0), \
        "driver must divide [0,1000] timestep by 1000"
    # image-position outputs sliced back to [B, S_img, 128]
    assert out.shape == (1, 16, 128)
    assert captured["L"] == 5 + 16


def test_sampler_denoise_feeds_raw_flow_time_not_inverted():
    """Sampler must feed the schedule's RAW flow time to the DiT (matches upstream).

    Upstream ``pipeline_ideogram4.py::__call__`` walks the Euler loop feeding the
    raw ``t_val = schedule(step_intervals[i+1])`` STRAIGHT to the transformer's
    ``t=`` argument, and ``driver.forward_pass`` (proven byte-exact vs upstream)
    divides the ``[0,1000]`` timestep by 1000 to the DiT's ``[0,1]``. So the
    sampler must pass ``ts = t_val * 1000`` — NOT ``(1 - t_val) * 1000``. The
    inverted form told the DiT "t=1 (data)" while ``z`` was still noise at the
    first step: a band-aid that contradicts the proven forward convention
    (t=0 noise, t=1 data). This pins the raw, upstream-matching convention.
    """
    from app.engine.models.families.ideogram4.sampler import (
        NUM_TRAIN_TIMESTEPS,
        IdeogramV4Sampler,
        _Ideogram4FlowSchedule,
    )

    captured_ts: list[float] = []
    precision_violations: list[str] = []

    class _Driver:
        def forward_pass(self, latents, ts, text, batch):
            captured_ts.append(float(ts.reshape(-1)[0]))
            # ── Precision contract (GPU ablation 2026-06-10) ──
            # autocast around this family's DiT collapses sampling to the
            # conditional mean; trajectory/timesteps/features must be fp32.
            if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
                precision_violations.append("autocast enabled in denoise forward")
            if latents.dtype != torch.float32:
                precision_violations.append(f"latents dtype {latents.dtype}")
            if ts.dtype != torch.float32:
                precision_violations.append(f"ts dtype {ts.dtype}")
            if text[0].dtype != torch.float32:
                precision_violations.append(f"text feats dtype {text[0].dtype}")
            return torch.zeros_like(latents)

    class _Pipe:
        config: dict = {}
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        use_amp = False

        def __init__(self):
            self.driver = _Driver()
            self.transformer = torch.nn.Linear(1, 1)  # has params -> dtype probe

    sampler = IdeogramV4Sampler(_Pipe())
    sampler._height, sampler._width = 512, 512
    sampler._latent_h, sampler._latent_w = 2, 2  # S_img = 2*2 = 4

    num_steps = 2
    noise = torch.randn(1, 4, 128)  # [1, S, 128]
    prompt_embedding = {
        "cond": (torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)),
    }

    sampler.denoise(
        noise, prompt_embedding, num_steps=num_steps, guidance_scale=1.0, seed=0,
    )

    schedule = _Ideogram4FlowSchedule(num_steps=num_steps, height=512, width=512)
    times = schedule.flow_times()
    expected = [
        times[i + 1] * NUM_TRAIN_TIMESTEPS for i in range(num_steps - 1, -1, -1)
    ]
    assert captured_ts == pytest.approx(expected, abs=1e-3), (
        f"sampler must feed raw flow time*1000 {expected}, got {captured_ts}; "
        "feeding (1 - t_val)*1000 inverts the DiT timestep convention"
    )
    # Sharpen: the FIRST step starts at the noise end (t~0), not data (t~1000).
    assert captured_ts[0] < 1.0, (
        "first denoise step must start near t=0 (noise); a value near 1000 means "
        "the inverted (1 - t_val) band-aid is back"
    )
    # Precision contract: no autocast, fp32 trajectory/timesteps/features.
    # GPU-validated 2026-06-10: autocast(bf16) around this DiT collapses the
    # 20-step loop to the conditional mean (cos(z_final, f32 ref) = 0.32 ->
    # flat image); fp32-no-autocast matches the upstream pipeline (cos ~1.0).
    assert not precision_violations, f"precision contract violated: {precision_violations}"


def test_driver_precision_spec_disables_amp():
    """ideogram4 trains WITHOUT autocast (AMP-off A/B, 2026-06-10).

    GPU ablation proved autocast(bf16) around this DiT corrupts the forward
    (the vendored model keeps deliberate f32 islands — 1e4-scaled t-sinusoids,
    RoPE over 65536-offset positions, adaln — that autocast force-downcasts):
    sampling collapses outright (cos 0.32 vs f32 ref) and the training forward
    carries ~10%% error (cos 0.86-0.94). bf16 INPUTS without autocast are
    harmless (cos 0.97-1.0), so autocast_dtype stays bf16 for the pipeline's
    input/cache casting while ``use_amp`` is force-disabled. Matches upstream
    + ai-toolkit, which never autocast this model.
    """
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    drv = IdeogramV4Driver(
        ModelDefinition(id="x", family="ideogram4", name="X"), torch.device("cpu"),
    )
    for mixed_precision in ("bf16", "fp16", "fp32"):
        spec = drv.get_precision_spec(mixed_precision)
        assert spec.use_amp is False, f"AMP must be off for {mixed_precision}"
        assert spec.grad_scaler_enabled is False
        assert spec.autocast_dtype == torch.bfloat16


def test_build_packed_inputs_shapes_and_role_counts():
    """_build_packed_inputs: pos last-dim 3, indicator/segment len L, role counts."""
    drv, _ = _make_driver_with_stub_dit(latent_h=2, latent_w=3)

    s_text, s_img = 4, 6  # 2*3
    text_feats = torch.randn(1, s_text, 8)
    # one padded text token at the end
    text_mask = torch.tensor([[True, True, True, False]])
    image_seq = torch.randn(1, s_img, 128)

    packed = drv._build_packed_inputs(text_feats, text_mask, image_seq, 2, 3)
    L = s_text + s_img

    assert packed["position_ids"].shape == (1, L, 3)
    assert packed["indicator"].shape == (1, L)
    assert packed["segment_ids"].shape == (1, L)
    assert packed["x"].shape == (1, L, 128)
    assert packed["llm_features"].shape == (1, L, 8)

    ind = packed["indicator"][0]
    # 3 real text tokens -> LLM_TOKEN_INDICATOR(3); 6 image -> OUTPUT_IMAGE(2);
    # 1 padded text -> 0.
    assert int((ind == 3).sum()) == 3
    assert int((ind == 2).sum()) == 6
    assert int((ind == 0).sum()) == 1

    # image positions are offset; text positions are < offset
    pos = packed["position_ids"][0]
    assert (pos[:s_text] < 65536).all()
    assert (pos[s_text:] >= 65536).all()
    # padded text position has zeroed llm_features
    assert torch.allclose(packed["llm_features"][0, 3], torch.zeros(8))
    # image latents land at image positions
    assert torch.allclose(packed["x"][0, s_text:], image_seq[0])
    assert torch.allclose(packed["x"][0, :s_text], torch.zeros(s_text, 128))


def test_trainer_setup_family_wires_strategies():
    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver
    from app.engine.models.families.ideogram4.loader import IdeogramV4Loader
    from app.engine.models.families.ideogram4.saver import IdeogramV4Saver

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    trainer = IdeogramV4Trainer(defn, {})
    trainer._setup_family()

    assert isinstance(trainer.driver, IdeogramV4Driver)
    assert isinstance(trainer.loader, IdeogramV4Loader)
    assert isinstance(trainer.saver, IdeogramV4Saver)


def test_saver_architecture_name():
    from app.engine.core.pipeline.saver_base import GenericLoRASaver
    from app.engine.models.families.ideogram4.saver import IdeogramV4Saver

    saver = IdeogramV4Saver()
    assert isinstance(saver, GenericLoRASaver)
    assert saver.architecture_name == "ideogram4"


def test_driver_get_saver_returns_saver():
    import torch
    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver
    from app.engine.models.families.ideogram4.saver import IdeogramV4Saver

    drv = IdeogramV4Driver(ModelDefinition(id="x", family="ideogram4", name="X"), torch.device("cpu"))
    assert isinstance(drv.get_saver(), IdeogramV4Saver)


def test_definition_yaml_loads():
    from pathlib import Path
    import yaml
    from app.engine.core.definitions import ModelDefinition

    yaml_path = (
        Path(__file__).parent.parent
        / "app/engine/models/families/ideogram4/definitions/ideogram4_fp8.yaml"
    )
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if "components" in data:
        for k, v in data["components"].items():
            if isinstance(v, str):
                data["components"][k] = {"path": v}

    definition = ModelDefinition(**data)
    assert definition.family == "ideogram4"
    assert "qkv" in definition.lora_targetable_modules
    assert "w1" in definition.lora_targetable_modules
    assert "o" in definition.lora_targetable_modules


def test_asymmetric_cfg_helpers():
    import torch
    from app.engine.models.families.ideogram4.sampler import (
        combine_asymmetric_cfg, zeroed_like_text,
    )
    cond = torch.full((1, 16, 128), 2.0)
    uncond = torch.full((1, 16, 128), 1.0)
    out = combine_asymmetric_cfg(cond, uncond, guidance_scale=3.0)
    assert torch.allclose(out, torch.full((1, 16, 128), 4.0))  # 1 + 3*(2-1)

    feats = torch.randn(1, 5, 104)
    z = zeroed_like_text(feats)
    assert z.shape == feats.shape
    assert torch.count_nonzero(z) == 0


def test_run_qwen3vl_layers_builds_the_mask_with_the_installed_transformers():
    """The driver's mask call reaches the REAL ``create_causal_mask`` (no stub)."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    cfg = Qwen3VLTextConfig(
        num_hidden_layers=2, hidden_size=HID, intermediate_size=64,
        num_attention_heads=2, num_key_value_heads=2, head_dim=HID // 2,
        vocab_size=8,
    )
    cfg._attn_implementation = "eager"
    lm = _FakeLanguageModel(2)
    lm.config = cfg

    drv = IdeogramV4Driver(
        ModelDefinition(id="x", family="ideogram4", name="X"), torch.device("cpu"),
    )
    drv.selected_layers = (0, 1)
    token_ids = torch.ones(1, 5, dtype=torch.long)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    pos_2d = torch.arange(5)[None, :]
    out = drv._run_qwen3vl_layers(lm, token_ids, attention_mask, pos_2d, torch.float32)
    assert out.shape == (1, 5, HID * 2)
    assert torch.isfinite(out).all()


# ── LANE-150: diffusers 0.41.0's Ideogram4Transformer2DModel replaces vendor/ ──
#
# The fixtures are what the VENDORED fused model produced (task 13's probe,
# .agent/workdir/lane-150/probes/ideogram4_parity_probe.py); nothing below imports vendor/.

_IG4_FIXTURES = Path(__file__).resolve().parents[1] / "app" / "engine" / "tests" / "fixtures" / "ideogram4"
_IG4_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
# The fixture's tiny config in the release spelling (hidden 16 = 8 x 2 heads).
_IG4_TINY = dict(
    in_channels=8, num_layers=2, attention_head_dim=8, num_attention_heads=2, intermediate_size=32,
    adaln_dim=8, llm_features_dim=12, rope_theta=10000, mrope_section=(1, 1, 1), norm_eps=1e-5,
)
_IG4_H = 16


def _ig4_npz() -> dict:
    import numpy as np

    with np.load(_IG4_FIXTURES / "vendor_parity.npz") as z:
        return {k: z[k] for k in z.files}


def _ig4_release_tiny():
    """A fresh release tiny model carrying the fixture's (vendored, fused) weights."""
    from diffusers import Ideogram4Transformer2DModel

    from app.engine.models.families.ideogram4 import utils

    sd = {k[3:]: torch.from_numpy(v) for k, v in _ig4_npz().items() if k.startswith("sd.")}
    model = Ideogram4Transformer2DModel(**_IG4_TINY)
    model.load_state_dict(utils.convert_transformer_state_dict(sd), strict=True)
    return model.eval()


def _ig4_inputs() -> dict:
    s = _ig4_npz()
    return dict(
        hidden_states=torch.from_numpy(s["in.x"]), timestep=torch.from_numpy(s["in.t"]),
        encoder_hidden_states=torch.from_numpy(s["in.llm_features"]),
        position_ids=torch.from_numpy(s["in.position_ids"]),
        segment_ids=torch.from_numpy(s["in.segment_ids"]), indicator=torch.from_numpy(s["in.indicator"]),
    )


def _ig4_forward(model) -> torch.Tensor:
    with torch.no_grad():
        return model(**_ig4_inputs(), return_dict=False)[0].float()


def _ig4_peft(rank: int, targets=None):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        _ig4_release_tiny(), LoraConfig(r=rank, lora_alpha=rank, target_modules=targets or _IG4_TARGETS),
    )


def _ig4_lora_layers(peft_model) -> dict:
    """``module path -> PEFT LoRA layer`` (paths without the ``base_model.model.`` prefix)."""
    return {
        n.removeprefix("base_model.model."): m
        for n, m in peft_model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
    }


def _ig4_seed_adapter(peft_model, seed: int = 0) -> None:
    """Nonzero ``lora_A`` AND ``lora_B`` (PEFT's zero ``lora_B`` init would pin nothing)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in _ig4_lora_layers(peft_model).values():
            for w in (layer.lora_A["default"].weight, layer.lora_B["default"].weight):
                w.copy_(torch.randn(w.shape, generator=g) * 0.1)


def _ig4_split_deltas(peft_model) -> dict:
    return {
        path: layer.lora_B["default"].weight.detach() @ layer.lora_A["default"].weight.detach()
        for path, layer in _ig4_lora_layers(peft_model).items()
    }


def _ig4_save(peft_model, path) -> dict:
    """The CURRENT saver on the PEFT model; returns the file's tensors (fp32 so 1e-6 holds)."""
    from safetensors.torch import load_file

    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    defn = ModelDefinition(id="x", family="ideogram4", name="X", lora_targetable_modules=["qkv", "o", "w1", "w2", "w3"])
    IdeogramV4Driver(defn, torch.device("cpu")).get_saver().save(
        components={"unet": peft_model, "config": {"save_precision": "fp32"}}, path=path,
    )
    return load_file(str(path))


def _ig4_convert(file_sd: dict) -> dict:
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_ideogram4_lora_to_diffusers

    return _convert_non_diffusers_ideogram4_lora_to_diffusers(dict(file_sd))


def _ig4_apply_converted(converted: dict):
    """diffusers' own loader path: ranks read from the tensor shapes, no alphas -> scale 1."""
    model = _ig4_release_tiny()
    model.load_lora_adapter(converted, prefix="transformer", adapter_name="default")
    return model


def _ig4_check(name: str, out: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    assert tuple(out.shape) == tuple(ref.shape), f"LANE-150 parity ideogram4/{name}: shape {tuple(out.shape)}"
    diff = float((out - ref).abs().max())
    assert diff <= atol, f"LANE-150 parity ideogram4/{name}: max|release - fixture| = {diff:.3e} > {atol:g}"


def test_get_lora_targets_expands_the_fused_names_at_the_peft_boundary():
    """The released definition list [qkv, o, w1, w2, w3] names the release's split modules."""
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    defn = ModelDefinition(id="x", family="ideogram4", name="X", lora_targetable_modules=["qkv", "o", "w1", "w2", "w3"])
    assert IdeogramV4Driver(defn, torch.device("cpu")).get_lora_targets() == _IG4_TARGETS
    bare = ModelDefinition(id="x", family="ideogram4", name="X")
    assert IdeogramV4Driver(bare, torch.device("cpu")).get_lora_targets() == _IG4_TARGETS


def test_expanded_targets_inject_into_the_release_model():
    """PEFT finds every expanded target on the release class (the fused names find nothing)."""
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    defn = ModelDefinition(id="x", family="ideogram4", name="X", lora_targetable_modules=["qkv", "o", "w1", "w2", "w3"])
    model = _ig4_peft(4, IdeogramV4Driver(defn, torch.device("cpu")).get_lora_targets())
    paths = set(_ig4_lora_layers(model))
    for i in range(2):
        for name in ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                     "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            assert f"layers.{i}.{name}" in paths


def test_ideogram4_saver_emits_the_released_fused_keys(tmp_path):
    """Q3 (interview): the file keeps the released ``attention.qkv`` / ``attention.o`` key names."""
    import json

    from safetensors import safe_open

    model = _ig4_peft(16)
    _ig4_seed_adapter(model)
    path = tmp_path / "ig4.safetensors"
    sd = _ig4_save(model, path)
    keys = sorted(sd)
    pinned = json.loads((_IG4_FIXTURES / "lora_keys.json").read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys ideogram4: saved keys differ from fixtures/ideogram4/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )
    # Rank 3r on the fused qkv (spec Decisions 1.01, the recorded behaviour change).
    a = sd["diffusion_model.layers.0.attention.qkv.lora_A.weight"]
    b = sd["diffusion_model.layers.0.attention.qkv.lora_B.weight"]
    assert tuple(a.shape) == (48, _IG4_H) and tuple(b.shape) == (3 * _IG4_H, 48)
    deltas = _ig4_split_deltas(model)
    for i in range(2):
        p = f"diffusion_model.layers.{i}.attention."
        fused = sd[p + "qkv.lora_B.weight"] @ sd[p + "qkv.lora_A.weight"]
        split = torch.cat([deltas[f"layers.{i}.attention.{n}"] for n in ("to_q", "to_k", "to_v")], dim=0)
        assert torch.allclose(fused, split, atol=1e-6, rtol=0)
        o = sd[p + "o.lora_B.weight"] @ sd[p + "o.lora_A.weight"]
        assert torch.allclose(o, deltas[f"layers.{i}.attention.to_out.0"], atol=1e-6, rtol=0)
    with safe_open(str(path), "pt") as f:
        assert f.metadata()["ss_network_dim"] == "16"


def test_ideogram4_refused_export_is_the_trained_function(tmp_path):
    """Adapter parity (ii): the file the saver writes is the vendored fused model's function.

    The fixture's split adapter (rank 4, alpha 4: scale 1) goes into the release PEFT model, through
    the CURRENT ``IdeogramV4Saver.save``, back out of the file through diffusers' 0.41 converter onto
    a FRESH release model; its forward must equal the vendored fused model carrying the same
    adapter as a merged delta (``adapter_forward``). A wrong block order fails here numerically.
    """
    s = _ig4_npz()
    model = _ig4_peft(4)
    layers = _ig4_lora_layers(model)
    installed = 0
    with torch.no_grad():
        for key, value in s.items():
            if not key.startswith("adapter."):
                continue
            path, which = key[len("adapter."):].rsplit(".", 1)
            getattr(layers[path], which)["default"].weight.copy_(torch.from_numpy(value))
            installed += 1
    assert installed == 2 * len(layers) == 28
    converted = _ig4_convert(_ig4_save(model, tmp_path / "ig4.safetensors"))
    out = _ig4_forward(_ig4_apply_converted(converted))
    _ig4_check("adapter_forward", out, torch.from_numpy(s["adapter_forward"]), 1e-5)


def test_ideogram4_lora_file_loads_through_the_0_41_converter(tmp_path):
    """Round trip (iii): converter output == installed split deltas; applied, == the PEFT forward."""
    model = _ig4_peft(16)
    _ig4_seed_adapter(model, seed=1)
    converted = _ig4_convert(_ig4_save(model, tmp_path / "ig4.safetensors"))
    deltas = _ig4_split_deltas(model)
    seen = 0
    for path, delta in deltas.items():
        a = converted[f"transformer.{path}.lora_A.weight"]
        b = converted[f"transformer.{path}.lora_B.weight"]
        assert torch.allclose(b @ a, delta, atol=1e-6, rtol=0), f"LANE-150 converter delta {path}"
        seen += 1
    assert seen == 14 and len(converted) == 2 * seen
    _ig4_check("converter_round_trip", _ig4_forward(_ig4_apply_converted(converted)), _ig4_forward(model), 1e-6)


def test_ideogram4_saver_zero_fills_an_untrained_projection(tmp_path):
    """A custom target list training only to_q still writes the fused layout, as the trained function."""
    model = _ig4_peft(4, ["to_q", "w1"])
    _ig4_seed_adapter(model, seed=2)
    sd = _ig4_save(model, tmp_path / "ig4.safetensors")
    deltas = _ig4_split_deltas(model)
    p = "diffusion_model.layers.0.attention.qkv."
    fused = sd[p + "lora_B.weight"] @ sd[p + "lora_A.weight"]
    assert tuple(fused.shape) == (3 * _IG4_H, _IG4_H)
    assert torch.allclose(fused[:_IG4_H], deltas["layers.0.attention.to_q"], atol=1e-6, rtol=0)
    assert torch.count_nonzero(fused[_IG4_H:]) == 0
    converted = _ig4_convert(sd)
    _ig4_check("partial_targets", _ig4_forward(_ig4_apply_converted(converted)), _ig4_forward(model), 1e-6)


def test_ideogram4_pre_lane_rank_r_file_loads_through_the_0_41_converter():
    """(iv): a file saved BEFORE this lane (shared rank-4 ``lora_A``) still loads on the release."""
    from safetensors.torch import load_file

    converted = _ig4_convert(load_file(str(_IG4_FIXTURES / "legacy_lora_rank4.safetensors")))
    for i in range(2):
        for proj in ("to_q", "to_k", "to_v"):
            p = f"transformer.layers.{i}.attention.{proj}"
            assert tuple(converted[p + ".lora_A.weight"].shape) == (4, _IG4_H)
            assert tuple(converted[p + ".lora_B.weight"].shape) == (_IG4_H, 4)
    model = _ig4_apply_converted({k: v.float() for k, v in converted.items()})
    assert torch.isfinite(_ig4_forward(model)).all()


def test_translate_transformer_config_accepts_both_spellings():
    from app.engine.models.families.ideogram4 import utils

    legacy = utils.translate_transformer_config(
        {"emb_dim": 4608, "num_heads": 18, "adanln_dim": 512, "num_layers": 34, "_class_name": "X"},
    )
    assert legacy["attention_head_dim"] == 256 and legacy["num_attention_heads"] == 18
    assert legacy["adaln_dim"] == 512 and legacy["num_layers"] == 34
    assert not {"emb_dim", "num_heads", "adanln_dim"} & set(legacy)
    release = {"attention_head_dim": 256, "num_attention_heads": 18, "adaln_dim": 512, "num_layers": 34}
    assert utils.translate_transformer_config(dict(release)) == release


def test_translate_transformer_config_refuses_a_silent_default():
    """Neither spelling: the class defaults (18 x 256, 512) must never fill the gap."""
    from app.engine.models.families.ideogram4 import utils

    with pytest.raises(ValueError, match="attention_head_dim"):
        utils.translate_transformer_config({})
    with pytest.raises(ValueError, match="num_attention_heads"):
        utils.translate_transformer_config({"emb_dim": 4608, "adaln_dim": 512})


def test_convert_transformer_state_dict_splits_qkv_and_renames_o():
    from app.engine.models.families.ideogram4 import utils

    qkv = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    out = utils.convert_transformer_state_dict({
        "layers.0.attention.qkv.weight": qkv, "layers.0.attention.o.weight": torch.ones(2, 2),
        "layers.0.feed_forward.w1.weight": torch.zeros(3, 2),
    })
    assert set(out) == {
        "layers.0.attention.to_q.weight", "layers.0.attention.to_k.weight", "layers.0.attention.to_v.weight",
        "layers.0.attention.to_out.0.weight", "layers.0.feed_forward.w1.weight",
    }
    assert torch.equal(torch.cat([out[f"layers.0.attention.{n}.weight"] for n in ("to_q", "to_k", "to_v")]), qkv)
    # An already-split dict passes through unchanged.
    assert set(utils.convert_transformer_state_dict(dict(out))) == set(out)


def test_forward_pass_bf16_release_model_fp32_latents_returns_float32():
    """The driver casts the inputs to the model dtype and the output back to float32."""
    from diffusers import Ideogram4Transformer2DModel
    from peft import LoraConfig, get_peft_model

    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    torch.manual_seed(0)
    dit = Ideogram4Transformer2DModel(
        in_channels=128, num_layers=1, attention_head_dim=8, num_attention_heads=2, intermediate_size=32,
        adaln_dim=8, llm_features_dim=8, rope_theta=10000, mrope_section=(1, 1, 1),
    ).to(torch.bfloat16)
    drv = IdeogramV4Driver(ModelDefinition(id="x", family="ideogram4", name="X"), torch.device("cpu"))
    drv.transformer = get_peft_model(dit, LoraConfig(r=4, lora_alpha=4, target_modules=drv.get_lora_targets()))
    text = (torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool))
    out = drv.forward_pass(torch.randn(2, 16, 128), torch.tensor([500.0, 250.0]), text, {"latent_h": 4, "latent_w": 4})
    assert out.dtype == torch.float32 and tuple(out.shape) == (2, 16, 128)
    assert torch.isfinite(out).all()


def test_no_vendor_import_remains():
    """ideogram4 keeps no vendored diffusers copy: no vendor/ directory, no import of it under backend/."""
    import re

    backend = Path(__file__).resolve().parents[1]
    family = backend / "app" / "engine" / "models" / "families" / "ideogram4"
    assert not (family / "vendor").exists(), f"ideogram4/vendor must be gone: {family / 'vendor'}"
    pat = re.compile("families" + r"\.ideogram4\.vendor|" + "modeling_" + "ideogram4|" + "autoencoder_" + "ideogram4")
    offenders = [
        str(f) for root in (backend / "app", backend / "tests") for f in root.rglob("*.py")
        if pat.search(f.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not offenders, f"ideogram4 still names the retired vendor copy: {offenders}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
