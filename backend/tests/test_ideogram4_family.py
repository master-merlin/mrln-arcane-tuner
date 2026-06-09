"""Smoke tests for the Ideogram 4 model family (no weights downloaded)."""
from __future__ import annotations

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
    monkeypatch.setattr(driver_mod, "create_causal_mask", lambda **kw: None)

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
        def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
            captured["t"] = t.detach().clone()
            captured["L"] = x.shape[1]
            captured["position_ids"] = position_ids.detach().clone()
            captured["segment_ids"] = segment_ids.detach().clone()
            captured["indicator"] = indicator.detach().clone()
            captured["llm_features"] = llm_features.detach().clone()
            captured["x"] = x.detach().clone()
            return torch.zeros(x.shape[0], x.shape[1], x.shape[2])

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
