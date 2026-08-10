"""minimax_h3 video contract + token budget.

The token budget is the binding constraint for this family: H3's native sparse
attention was NOT included in the weights release, so attention is dense over
the full video token sequence. A 4s clip is ~14.4K tokens; a 15s clip is
~52.4K. These numbers decide whether a configuration trains at all, so they are
pinned rather than left to be rediscovered by an OOM 40 minutes into a run.
"""

from __future__ import annotations

import pytest

from app.engine.core.video_contract import frame_predicate, resolve_video_profile
from app.engine.models.registry import ModelRegistry


def _profile(def_id: str = "minimax-h3-t2va"):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return resolve_video_profile(ModelRegistry._definitions[def_id])


def _frame_rule(def_id: str = "minimax-h3-t2va") -> str:
    """The family's declared Nn+M frame rule, read from the YAML.

    Every frame-rule assertion in this file must obtain the rule THROUGH this
    helper (or ``resolve_video_profile``), never as a hardcoded ``"17n+5"``
    literal — a hardcoded literal would let the suite keep passing after
    someone edited ``video.frame_rule`` in the YAML, which is exactly the
    drift this file exists to catch.
    """
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id].architecture_params["video.frame_rule"]


def _tokens(frames: int, height: int, width: int, def_id: str = "minimax-h3-t2va") -> int:
    """Effective sequence length, DERIVED FROM THE DEFINITION — not hardcoded.

    Reading the spatial factors out of architecture_params is the whole
    point: this test must fail if someone edits the YAML's vae_spatial or
    patch_size, which a hardcoded 32 would sail straight past. (The raw
    ``video.vae_temporal`` factor is pinned separately, by
    ``test_profile_declares_raw_vae_factor_not_post_patchify`` — it is not
    read here, because H3's latent-frame count comes from the ``17n+5``
    frame rule, not the raw temporal-downsample factor.)
    """
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    arch = ModelRegistry._definitions[def_id].architecture_params

    # Effective spatial factor = raw VAE factor x the transformer's spatial patch.
    # patch_size is [t, h, w]; h and w are equal for H3.
    patch_t, patch_h, patch_w = arch["transformer.patch_size"]
    spatial_h = arch["video.vae_spatial"] * patch_h
    spatial_w = arch["video.vae_spatial"] * patch_w

    # H3 chunks 17 pixel frames -> 5 latent frames. The naive
    # (F-1)//video.vae_temporal + 1 derivation is WRONG here and yields
    # non-integers.
    step, offset = 17, 5
    latent_frames = ((frames - offset) // step * offset + 2) // patch_t
    return latent_frames * (height // spatial_h) * (width // spatial_w)


def _audio_rows(frames: int, def_id: str = "minimax-h3-t2va") -> int:
    """Audio rows are NOT negligible and are NOT optional.

    H3 packs [text | conditions | audio | video] into ONE sequence; the audio
    rows are always present, even for a silent clip (silence noised at the
    audio sigma, contributing zero loss). Any sequence-length budget that
    omits them under-counts.
    """
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    arch = ModelRegistry._definitions[def_id].architecture_params
    latents = round(frames / arch["video.frame_rate"] * arch["audio.latent_rate"])
    return latents * arch["audio.channels"]


def test_effective_factors_derive_to_32x_and_4x():
    """Guards the derivation itself, so the table below cannot pass for the
    wrong reason (e.g. vae_spatial 32 with patch 1 also yields 32); also pins
    the raw 4x temporal factor the test's name promises."""
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    arch = ModelRegistry._definitions["minimax-h3-t2va"].architecture_params
    assert arch["video.vae_spatial"] == 16
    assert arch["transformer.patch_size"] == [1, 2, 2]
    assert arch["video.vae_spatial"] * arch["transformer.patch_size"][1] == 32
    assert arch["video.vae_temporal"] == 4


def test_declared_frame_rule_is_17n_plus_5():
    """A typo/edit in the YAML's video.frame_rule (e.g. '17n+5' -> '17n+6')
    must fail LOUDLY here, not just silently change which frame counts the
    parametrized tests below happen to accept."""
    assert _frame_rule() == "17n+5"
    assert _profile().frame_rule == "17n+5"


@pytest.mark.parametrize("frames,valid", [(5, True), (107, True), (124, True), (345, True),
                                          (97, False), (106, False), (1, False)])
def test_frame_rule_is_17n_plus_5(frames, valid):
    assert frame_predicate(_frame_rule())(frames) is valid


def test_profile_declares_raw_vae_factor_not_post_patchify():
    # WAN declares 8 with a 2x2 patchify; LTX-2 declares 32 with patch_size 1.
    # Declaring 32 here would double-count H3's 2x patchify and silently halve
    # the usable resolution grid.
    profile = _profile()
    assert profile.vae_spatial == 16
    assert profile.vae_temporal == 4
    assert profile.divisibility == 32
    assert profile.frame_rule == "17n+5"


@pytest.mark.parametrize("frames,expected", [(107, 18_432), (124, 21_312), (345, 58_752)])
def test_video_token_budget_matches_the_spec_table(frames, expected):
    # 107 -> 32 latent frames x 576 = 18,432 independently matches the row
    # count ai-toolkit reports for the same geometry.
    assert _tokens(frames, 768, 768) == expected


@pytest.mark.parametrize("frames,expected", [(107, 356), (124, 414), (345, 1150)])
def test_audio_rows_are_counted_not_ignored(frames, expected):
    assert _audio_rows(frames) == expected


def test_default_clip_is_a_valid_17n_plus_5_bucket():
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    defaults = ModelRegistry._definitions["minimax-h3-t2va"].defaults
    assert defaults["num_frames"] == 107
    assert frame_predicate(_frame_rule())(defaults["num_frames"])
    # Guards the original mistake: 97 came from a 4n+1 derivation and is invalid.
    assert defaults["num_frames"] != 97
