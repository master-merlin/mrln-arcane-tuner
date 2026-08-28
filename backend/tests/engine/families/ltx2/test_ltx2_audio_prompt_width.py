"""The audio prompt is read at two different widths, and the driver must pick.

Found by the LTX-2.5 GPU UAT, at the FIRST sampling step, several frames inside
a transformer block:

    RuntimeError: The size of tensor a (3840) must match the size of tensor b
    (2048) at non-singleton dimension 2

The driver fed ``audio_encoder_hidden_states`` at ``caption_channels`` (3840)
unconditionally. That is right for LTX-2.3 and wrong for LTX-2.5, and the
difference is one config flag:

* ``use_prompt_embeddings=True`` -- the model builds an
  ``audio_caption_projection`` (``caption_channels`` -> ``audio_inner_dim``) and
  is handed the RAW caption width. **LTX-2.3's checkpoint does not contain the
  key at all**, so diffusers' own default decides it, which is why nothing in
  the definition hinted at the dependency.
* ``use_prompt_embeddings=False`` -- LTX-2.5. No projection is built; the
  connector already emits per-modality widths and the audio cross-attention
  reads ``audio_cross_attention_dim`` directly.

Why unit tests could not have caught it before: both definitions declare the
SAME two numbers (3840 and 2048), so no comparison between the YAMLs shows a
divergence -- only the flag does, and only one of the two checkpoints states it.
Every test here is therefore anchored on diffusers' real signature and real
source, not on the literals.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    yield
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


def _arch(definition_id: str) -> dict:
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition(definition_id)
    assert defn is not None, f"{definition_id} did not register"
    return getattr(defn, "architecture_params", {}) or {}


def _driver(*, use_prompt_embeddings: bool, caption: int, audio_cross: int) -> Ltx2Driver:
    drv = object.__new__(Ltx2Driver)
    drv.caption_channels = caption
    drv.audio_cross_attention_dim = audio_cross
    drv.use_prompt_embeddings = use_prompt_embeddings
    drv.audio_in_channels = 128
    return drv


def _transformer_source() -> str:
    import diffusers.models.transformers.transformer_ltx2 as mod

    return pathlib.Path(inspect.getfile(mod)).read_text(encoding="utf-8")


# --- The rule itself ----------------------------------------------------


def test_with_prompt_embeddings_the_model_projects_so_it_wants_the_caption_width():
    drv = _driver(use_prompt_embeddings=True, caption=3840, audio_cross=2048)
    assert drv._audio_prompt_width() == 3840


def test_without_prompt_embeddings_the_audio_cross_attention_width_is_used():
    drv = _driver(use_prompt_embeddings=False, caption=3840, audio_cross=2048)
    assert drv._audio_prompt_width() == 2048


def test_the_dummy_audio_stream_is_built_at_that_width_not_the_caption_width():
    """The exact tensor the crash came from: video-only sampling's dummy audio."""
    drv = _driver(use_prompt_embeddings=False, caption=3840, audio_cross=2048)
    hidden = torch.zeros(2, 7, 128)
    video_emb = torch.zeros(2, 11, 4096)  # 2.5's connector emits inner_dim here
    audio_h, audio_emb = drv._dummy_audio_inputs(hidden, video_emb)

    assert audio_emb.shape == (2, 1, 2048), (
        "the video-only dummy audio prompt must match audio_cross_attention_dim; "
        "3840 here is the exact shape that killed the LTX-2.5 UAT at step 0"
    )
    assert audio_h.shape == (2, 1, 128)
    assert torch.count_nonzero(audio_emb) == 0


def test_the_missing_cache_fallback_uses_the_audio_width_not_the_video_one():
    """``zeros_like(video_emb)`` was wrong whenever the two widths differ."""
    drv = _driver(use_prompt_embeddings=False, caption=3840, audio_cross=2048)
    video_emb = torch.zeros(2, 11, 4096)

    class _NoPooled:
        pooled = None

    fallback = drv._audio_embeddings(_NoPooled(), video_emb)
    assert fallback.shape[-1] == 2048 != video_emb.shape[-1]
    assert torch.count_nonzero(fallback) == 0


def test_a_real_pooled_embedding_is_passed_through_untouched():
    drv = _driver(use_prompt_embeddings=False, caption=3840, audio_cross=2048)
    pooled = torch.randn(2, 5, 2048)

    class _WithPooled:
        def __init__(self) -> None:
            self.pooled = pooled

    assert torch.equal(drv._audio_embeddings(_WithPooled(), torch.zeros(2, 11, 4096)), pooled)


# --- Anchored on diffusers, so a runtime change cannot pass silently ----


def test_diffusers_still_defaults_the_flag_to_true_which_is_what_ltx2_3_relies_on():
    """LTX-2.3's config omits the key entirely — the default IS its value."""
    from diffusers import LTX2VideoTransformer3DModel

    default = inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters[
        "use_prompt_embeddings"
    ].default
    assert default is True, (
        "LTX-2.3's checkpoint does not declare use_prompt_embeddings; if diffusers "
        "flips this default, 2.3's audio prompt width changes with it"
    )


def test_the_audio_caption_projection_exists_only_under_the_flag():
    """The whole rule in one structural fact, read from diffusers' own source."""
    source = _transformer_source()
    start = source.find("if use_prompt_embeddings:")
    assert start > 0, "diffusers no longer gates the caption projections on the flag"
    guarded = source[start:start + 800]
    assert "self.audio_caption_projection" in guarded, (
        "audio_caption_projection moved out of the use_prompt_embeddings branch; "
        "the driver's width rule is derived from it being inside"
    )
    assert "in_features=caption_channels" in guarded


# --- Both shipped definitions, against the rule -------------------------


@pytest.mark.parametrize("definition_id", ["ltx2-3-base", "ltx2-5-base"])
def test_every_shipped_ltx2_definition_declares_a_usable_audio_width(definition_id):
    arch = _arch(definition_id)
    caption = int(arch.get("transformer.caption_channels", 3840))
    audio_cross = int(arch.get("transformer.audio_cross_attention_dim", 2048))
    # Absent means "diffusers decides", and diffusers says True.
    flag = bool(arch.get("transformer.use_prompt_embeddings", True))
    width = _driver(
        use_prompt_embeddings=flag, caption=caption, audio_cross=audio_cross,
    )._audio_prompt_width()
    assert width == (caption if flag else audio_cross)
    assert width > 0


def test_the_two_definitions_disagree_on_the_flag_and_agree_on_the_numbers():
    """Why a YAML diff could never have found this."""
    a, b = _arch("ltx2-3-base"), _arch("ltx2-5-base")
    assert a.get("transformer.caption_channels") == b.get("transformer.caption_channels")
    assert (
        a.get("transformer.audio_cross_attention_dim")
        == b.get("transformer.audio_cross_attention_dim")
    )
    assert a.get("transformer.use_prompt_embeddings") is None
    assert b.get("transformer.use_prompt_embeddings") is False
