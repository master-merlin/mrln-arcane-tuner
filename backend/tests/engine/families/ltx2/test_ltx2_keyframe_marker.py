"""LTX-2.5 ``keyframe_marker``: the transformer's keyframe embedding, wired deliberately.

diffusers 0.41 ``LTX2VideoTransformer3DModel.forward`` adds the learned
``keyframes_abs_pos_embedding`` to every video token the optional
``video_keyframes_mask`` marks, right after ``proj_in`` (transformer_ltx2.py
:1517-1520). Upstream's DFR pipelines always mark the causal first latent frame
(pipeline_ltx2_dfr.py:1048-1050). LANE-153 lets an LTX-2.5 run opt into the same
marker on every training and preview forward.

What this file proves, and why each observable is the one it is:

* the embedding proof reads the input of ``transformer_blocks[0]`` (a forward
  pre-hook), never the model output: self-attention mixes the offset into EVERY
  token after the first block, so "unmarked tokens move by exactly 0" holds only
  before it. The parameter is zero-initialised, so it is set non-zero first.
* the wiring tests spy on the three transformer calls (the video-only and the
  joint training branch, the sampler's preview forward) and check the mask came
  from ONE producer, ``Ltx2Driver._keyframes_mask`` (RULE-21).
* the transfer test runs the real ``Ltx2Trainer._setup_family`` so a run config's
  ``keyframe_marker`` reaches the driver the way ``train_audio`` does.
* the registry, contract and metadata tests pin the option to ``ltx2-5-base``,
  refuse it elsewhere (D10: never a silent no-op), and record it in the LoRA.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ltx2 import trainer as trainer_mod
from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.sampler import Ltx2Sampler
from app.engine.models.families.ltx2.saver import Ltx2Saver
from app.engine.models.families.ltx2.trainer import Ltx2Trainer
from app.engine.models.registry import ModelRegistry

diffusers_ltx2 = pytest.importorskip("diffusers.models.transformers.transformer_ltx2")
LTX2VideoTransformer3DModel = diffusers_ltx2.LTX2VideoTransformer3DModel

KEYFRAME_DEFINITION = "ltx2-5-base"


# ---------------------------------------------------------------------------
# 1. The embedding proof on a tiny real transformer (CPU, fp32, no weights)
# ---------------------------------------------------------------------------

# The smallest kwargs that build the real 0.41 class on CPU (transformer_ltx2.py
# :1140-1187 is the signature): inner_dim = 2 heads x 8 = 16 on both streams.
_TINY_KWARGS = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=2,
    attention_head_dim=8,
    cross_attention_dim=16,
    caption_channels=16,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=2,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=16,
    num_layers=1,
)

# Packed video F=2, H=2, W=2, patch 1: 8 tokens, the first H*W = 4 are frame 0.
_F, _H, _W = 2, 2, 2
_TPF = _H * _W
_TOKENS = _F * _TPF


def _tiny_model(use_keyframes: bool = True) -> nn.Module:
    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(
        **_TINY_KWARGS, use_keyframes_abs_pos_embedding=use_keyframes,
    )
    return model.to(torch.float32).eval()


def _tiny_inputs() -> dict:
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, _TOKENS, 8, generator=g),
        audio_hidden_states=torch.randn(1, 1, 8, generator=g),
        encoder_hidden_states=torch.randn(1, 4, 16, generator=g),
        audio_encoder_hidden_states=torch.randn(1, 4, 16, generator=g),
        timestep=torch.tensor([500.0]),
        sigma=torch.tensor([500.0]),
        num_frames=_F,
        height=_H,
        width=_W,
        fps=24.0,
        audio_num_frames=1,
        isolate_modalities=True,
        return_dict=False,
    )


def _block0_input(model: nn.Module, **extra) -> torch.Tensor:
    """The video hidden states as they ENTER the first transformer block."""
    captured: list[torch.Tensor] = []

    def _hook(_module, _args, kwargs):
        captured.append(kwargs["hidden_states"].detach().clone())

    handle = model.transformer_blocks[0].register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**_tiny_inputs(), **extra)
    finally:
        handle.remove()
    assert len(captured) == 1, "the pre-hook must fire exactly once per forward"
    return captured[0]


def test_tiny_transformer_builds_on_cpu():
    model = _tiny_model()
    assert model.config.use_keyframes_abs_pos_embedding is True
    assert tuple(model.keyframes_abs_pos_embedding.shape) == (1, 16)
    out = _block0_input(model)
    assert out.shape == (1, _TOKENS, 16)


def test_marked_tokens_move_by_exactly_the_embedding_and_unmarked_do_not():
    model = _tiny_model()
    torch.manual_seed(0)
    nn.init.normal_(model.keyframes_abs_pos_embedding)
    embedding = model.keyframes_abs_pos_embedding.detach()
    assert embedding.abs().sum() > 0, "the zero-initialised embedding must be set non-zero"

    mask = Ltx2Driver._keyframes_mask(
        num_tokens=_TOKENS, tpf=_TPF, batch_size=1,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    none_run = _block0_input(model, video_keyframes_mask=None)
    masked_run = _block0_input(model, video_keyframes_mask=mask)

    delta = masked_run - none_run
    # Unmarked first: this is the assertion an output-level observable cannot
    # pass (attention spreads the offset), so it is what refuses the wrong probe.
    assert torch.equal(delta[:, _TPF:], torch.zeros_like(delta[:, _TPF:])), (
        "tokens after the first latent frame must move by exactly 0"
    )
    torch.testing.assert_close(
        delta[:, :_TPF], embedding.expand(1, _TPF, -1), atol=1e-6, rtol=0,
    )


def test_an_all_zero_mask_leaves_every_token_unchanged():
    model = _tiny_model()
    torch.manual_seed(0)
    nn.init.normal_(model.keyframes_abs_pos_embedding)
    none_run = _block0_input(model, video_keyframes_mask=None)
    zero_run = _block0_input(
        model, video_keyframes_mask=torch.zeros(1, _TOKENS, 1),
    )
    assert torch.equal(zero_run, none_run)


def test_a_none_mask_equals_a_model_built_without_the_flag():
    flagged = _tiny_model(use_keyframes=True)
    torch.manual_seed(0)
    nn.init.normal_(flagged.keyframes_abs_pos_embedding)
    plain = _tiny_model(use_keyframes=False)
    missing, unexpected = plain.load_state_dict(flagged.state_dict(), strict=False)
    assert missing == [], f"the flag-less model lacks weights: {missing}"
    assert unexpected == ["keyframes_abs_pos_embedding"], unexpected

    assert torch.equal(
        _block0_input(plain), _block0_input(flagged, video_keyframes_mask=None),
    )


# ---------------------------------------------------------------------------
# 2. The three forwards: off -> None, on -> the one producer's (B, tokens, 1)
# ---------------------------------------------------------------------------

_B, _SF, _SH, _SW = 2, 3, 4, 4  # 48 tokens, tpf 16
_S_TPF = _SH * _SW
_S_TOKENS = _SF * _S_TPF


class _SpyTransformer:
    """Records every call's kwargs; returns zero (video, audio) of the input shapes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, **kw):
        self.calls.append(kw)
        return (
            torch.zeros_like(kw["hidden_states"]),
            torch.zeros_like(kw["audio_hidden_states"]),
        )


def _spy_driver(keyframe_marker: bool) -> Ltx2Driver:
    """A weightless driver (as test_ltx2_i2v.py:21-35) with a spy transformer."""
    d = object.__new__(Ltx2Driver)
    d.transformer = _SpyTransformer()
    d.patch_size = 1
    d.patch_size_t = 1
    d._latent_shape = (_SF, _SH, _SW)
    d._i2v_active = False
    d.train_audio = False
    d.keyframe_marker = keyframe_marker
    d.audio_in_channels = 128
    d.caption_channels = 3840
    d.use_prompt_embeddings = True
    d.audio_cross_attention_dim = 2048
    d.frame_rate = 24.0
    d.audio_sampling_rate = 16000
    d.audio_vae = None
    d.vocoder = None
    return d


def _te() -> TextEncoderOutput:
    return TextEncoderOutput(
        embeddings=torch.zeros(_B, 11, 3840),
        attention_mask=None,
        pooled=torch.zeros(_B, 11, 3840),
    )


def _video_only_call(d: Ltx2Driver) -> dict:
    d.train_audio = False
    d.forward_pass(
        torch.zeros(_B, _S_TOKENS, 128), torch.full((_B,), 500.0), _te(), {},
    )
    return d.transformer.calls[-1]


def _joint_call(d: Ltx2Driver) -> dict:
    d.train_audio = True
    batch = {"audio_clean": torch.zeros(_B, 5, 128)}
    d.forward_pass(
        torch.zeros(_B, _S_TOKENS, 128), torch.full((_B,), 500.0), _te(), batch,
    )
    assert d.transformer.calls[-1]["isolate_modalities"] is False, "not the joint branch"
    return d.transformer.calls[-1]


def _sampler_call(d: Ltx2Driver) -> dict:
    d.train_audio = False
    pipe = type("P", (), {})()
    pipe.driver = d
    pipe.transformer = d.transformer
    pipe.vae = object()
    pipe.config = {"sample_num_frames": 25}
    pipe.device = torch.device("cpu")
    pipe.components = {}
    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = pipe.config
    s.device = pipe.device
    s.denoise(
        torch.zeros(_B, _S_TOKENS, 128), _te(), num_steps=1, guidance_scale=1.0, seed=0,
    )
    return d.transformer.calls[-1]


_FORWARDS = {
    "video_only": _video_only_call,
    "joint": _joint_call,
    "sampler": _sampler_call,
}


@pytest.mark.parametrize("path", sorted(_FORWARDS))
def test_marker_off_passes_an_explicit_none(path):
    call = _FORWARDS[path](_spy_driver(keyframe_marker=False))
    assert "video_keyframes_mask" in call, (
        f"{path}: the forward must pass video_keyframes_mask explicitly (None when off)"
    )
    assert call["video_keyframes_mask"] is None


@pytest.mark.parametrize("path", sorted(_FORWARDS))
def test_marker_on_passes_the_one_producers_mask(path, monkeypatch):
    produced: list[torch.Tensor] = []
    real = Ltx2Driver._keyframes_mask

    def _recording(*args, **kwargs):
        out = real(*args, **kwargs)
        produced.append(out)
        return out

    monkeypatch.setattr(Ltx2Driver, "_keyframes_mask", staticmethod(_recording))

    call = _FORWARDS[path](_spy_driver(keyframe_marker=True))
    mask = call.get("video_keyframes_mask")
    assert mask is not None, "video_keyframes_mask is None"
    assert any(mask is p for p in produced), (
        f"{path}: the mask must come from Ltx2Driver._keyframes_mask (RULE-21)"
    )
    assert tuple(mask.shape) == (_B, _S_TOKENS, 1)
    row = torch.tensor([1.0] * _S_TPF + [0.0] * (_S_TOKENS - _S_TPF))
    for b in range(_B):
        assert torch.equal(mask[b, :, 0].float().cpu(), row)
    assert torch.equal(
        mask,
        real(
            num_tokens=_S_TOKENS, tpf=_S_TPF, batch_size=_B,
            device=mask.device, dtype=mask.dtype,
        ),
    )


# ---------------------------------------------------------------------------
# 3. The run config reaches the driver (trainer._setup_family)
# ---------------------------------------------------------------------------

class _StubLoader:
    def __init__(self, device, train_audio=False) -> None:
        self.device = device
        self.train_audio = train_audio


class _StubDriver:
    def __init__(self, definition, device) -> None:
        self.definition = definition
        self.device = device


def _setup_family_with(config: dict, monkeypatch) -> Ltx2Trainer:
    monkeypatch.setattr(trainer_mod, "Ltx2Loader", _StubLoader)
    monkeypatch.setattr(trainer_mod, "Ltx2Driver", _StubDriver)
    t = object.__new__(Ltx2Trainer)
    t.definition = type("D", (), {"architecture_params": {"has_audio": True}})()
    t.device = torch.device("cpu")
    t.config = config
    t._setup_family()
    return t


def test_setup_family_surfaces_keyframe_marker_on_the_driver(monkeypatch):
    t = _setup_family_with({"keyframe_marker": True}, monkeypatch)
    assert t.driver.keyframe_marker is True


def test_setup_family_without_the_key_leaves_the_marker_off(monkeypatch):
    t = _setup_family_with({}, monkeypatch)
    assert t.driver.keyframe_marker is False


def test_a_driver_built_without_a_trainer_defaults_the_marker_off():
    assert Ltx2Driver.keyframe_marker is False
    assert object.__new__(Ltx2Driver).keyframe_marker is False


# ---------------------------------------------------------------------------
# 4. Registry visibility + the config contract
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    saved = (
        ModelRegistry._families, ModelRegistry._definitions,
        ModelRegistry._paths, ModelRegistry._discovered,
        getattr(ModelRegistry, "_definitions_loaded", False),
    )
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    reg = ModelRegistry()
    reg.initialize()
    yield reg
    (
        ModelRegistry._families, ModelRegistry._definitions,
        ModelRegistry._paths, ModelRegistry._discovered,
        ModelRegistry._definitions_loaded,
    ) = saved


def test_keyframe_marker_is_visible_for_exactly_ltx2_5_base(registry):
    from app.engine.core.archetypes import resolve_capabilities

    ids = registry.list_models()
    assert KEYFRAME_DEFINITION in ids and "ltx2-3-base" in ids
    visible = sorted(
        def_id for def_id in ids
        if resolve_capabilities(registry.get_definition(def_id))[
            "field_visibility"
        ]["keyframe_marker"]["supported"]
    )
    assert visible == [KEYFRAME_DEFINITION]


def test_keyframe_marker_on_a_flagless_definition_is_a_config_error(registry):
    from app.engine.core.video_contract import validate_video_config

    report = validate_video_config(
        registry.get_definition("ltx2-3-base"), {"keyframe_marker": True},
    )
    named = [e for e in report.errors if "keyframe_marker" in e]
    assert len(report.errors) == 1 and len(named) == 1, report.errors


def test_keyframe_marker_on_ltx2_5_base_is_accepted(registry):
    from app.engine.core.video_contract import validate_video_config

    report = validate_video_config(
        registry.get_definition(KEYFRAME_DEFINITION), {"keyframe_marker": True},
    )
    assert report.errors == []


def test_keyframe_marker_is_a_training_config_field_off_by_default():
    from app.engine.models.base import BaseTrainingConfig

    field = BaseTrainingConfig.model_fields.get("keyframe_marker")
    assert field is not None, "keyframe_marker is not a BaseTrainingConfig field"
    assert field.default is False


# ---------------------------------------------------------------------------
# 5. The LoRA header says whether the marker was trained in
# ---------------------------------------------------------------------------

def test_metadata_records_the_marker_when_on():
    assert Ltx2Saver._config_metadata({"keyframe_marker": True})["ss_keyframe_marker"] == "True"


def test_metadata_omits_the_marker_when_off():
    # The map writes every non-None value (saver.py:149-152), and str(False) is
    # non-empty: the implementation must skip this key explicitly when falsy.
    assert "ss_keyframe_marker" not in Ltx2Saver._config_metadata({"keyframe_marker": False})
    assert "ss_keyframe_marker" not in Ltx2Saver._config_metadata({"seed": 1})
