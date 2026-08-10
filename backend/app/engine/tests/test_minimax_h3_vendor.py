"""Vendor smoke tests for the minimax_h3 family (Task 1 — vendor drop only).

No loader/driver/trainer/sampler yet. This pins two contracts so later tasks
build on stable ground:

  1. Import smoke: all four vendored modules import with diffusers 0.39.0
     installed, despite being written against 0.36.0.dev0 internals.
  2. Tiny instantiate + CPU forward: a divisibility-respecting tiny config
     produces finite output of the expected shape, proving the MRLN-PATCH
     shims did not break the forward path.

The tiny config is deliberately un-real. It preserves the relationships that
matter (patch_size vs spatial dims, in_channels vs VAE latent width) while
shrinking everything else, so this runs on CPU in seconds with no weights.

Reconciled against the REAL vendored ``MiniMaxH3Transformer3DModel``
(``__init__`` and ``.forward``), which differ from a naive first guess in two
load-bearing ways:

  * ``MiniMaxH3Transformer3DModel`` does NOT patchify video latents itself --
    ``proj_in`` is a plain ``nn.Linear(video_patch_dim, hidden_size)``. A
    driver assembling the packed sequence (not built until a later task) is
    expected to patchify before calling forward. ``_patchify_video`` below
    reproduces that math (matching ``patch_size`` exactly) so this smoke test
    can feed a correctly-shaped tensor without a real driver.
  * H3 packs text + video + audio rows into ONE sequence and self-attends
    over it with no cross-attention at all (see ``MiniMaxH3AttnProcessor``'s
    docstring). ``forward`` takes the packed rows directly
    (``hidden_states``, ``audio_hidden_states``, ``encoder_hidden_states``)
    plus the bookkeeping that describes the packing: ``token_tags`` (which
    modality each row is), ``timestep_indices`` (which of the *distinct*
    ``timestep`` values each row is at -- H3 genuinely runs multiple noise
    levels in one forward, e.g. clean conditioning rows next to a noisy
    target), ``position_ids`` (the (t, h, w) rope grid) and three index
    tensors locating each modality's rows in the packed sequence. The test
    below builds a real two-noise-level packed sequence (text + audio
    conditioning at one timestep, target video at another) rather than
    collapsing everything to a single timestep, to actually exercise the
    per-row ``timestep_indices`` -> AdaLN-table addressing this model is
    built around.
"""

from __future__ import annotations

import torch

from app.engine.models.families.minimax_h3.vendor.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)

# Tiny, divisibility-respecting. Real checkpoint values are in the definition
# YAML; these exist only to exercise code paths on CPU.
#
# Divisibility relationships preserved from the real 56/128/5376/... config:
#   - rope: rotary_dim = 2 * 3 * rope_freq_dim must be <= attention_head_dim
#     (MiniMaxH3RotaryPosEmbed rotates the leading `rotary_dim` channels of
#     every head and passes the rest through unchanged -- see
#     `_apply_rotary_emb`). 2 * 3 * 2 = 12 <= attention_head_dim=16.
#   - time embedding funnel: freq_dim (sinusoidal width) -> time_embed_hidden_dim
#     (TimestepEmbedding's internal width) -> time_embed_dim (the width every
#     AdaLN projection actually consumes), shrinking at each stage exactly as
#     the real config does (256 -> 5376 -> 2688, i.e. hidden > final).
TINY_TRANSFORMER_KWARGS: dict = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 16,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 32,
    "in_channels": 24,          # REAL — must match the visual VAE latent width
    "audio_in_channels": 32,    # REAL — must match the audio VAE latent width
    "patch_size": [1, 2, 2],    # REAL — drives the 2x spatial patchify
    "text_dim": 16,
    "freq_dim": 16,
    "time_embed_hidden_dim": 16,
    "time_embed_dim": 8,
    "rope_freq_dim": 2,
    "rope_theta": 10000.0,
}


def build_tiny_transformer() -> MiniMaxH3Transformer3DModel:
    """A tiny CPU transformer for structural tests. Shared with the
    definitions test so both derive targets from the SAME structure."""
    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER_KWARGS).eval()


def _patchify_video(latents: torch.Tensor, patch_size: list[int]) -> torch.Tensor:
    """Reproduce the patchify math the (not-yet-built) driver is expected to
    do before calling ``forward`` -- the vendored transformer's ``proj_in``
    is a plain ``nn.Linear(video_patch_dim, hidden_size)``, it does not
    patchify raw ``(batch, channels, frames, height, width)`` latents itself.

    Row order matches `in_channels` docstring's "rows ordered as they appear
    in the packed sequence": frames outermost, then height, then width.
    """
    batch, channels, frames, height, width = latents.shape
    pt, ph, pw = patch_size
    assert frames % pt == 0 and height % ph == 0 and width % pw == 0
    latents = latents.view(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    num_tokens = (frames // pt) * (height // ph) * (width // pw)
    patch_dim = channels * pt * ph * pw
    return latents.reshape(batch, num_tokens, patch_dim)


def test_all_vendored_modules_import():
    import importlib

    base = "app.engine.models.families.minimax_h3.vendor."
    for name in (
        "transformer_minimax_h3",
        "autoencoder_kl_minimax_h3",
        "autoencoder_kl_minimax_h3_audio",
        "scheduling_minimax_h3",
    ):
        assert importlib.import_module(base + name) is not None


def test_revision_file_pins_the_expected_sha():
    import pathlib

    revision = (
        pathlib.Path(__file__).resolve().parents[1]
        / "models" / "families" / "minimax_h3" / "vendor" / "REVISION"
    ).read_text(encoding="utf-8")
    assert "245d78fb48f1c87dfb560a94bea6e191c9f9f1c0" in revision


def test_tiny_transformer_instantiates_with_expected_block_counts():
    model = build_tiny_transformer()
    # Walk the REAL checkpoint attribute paths rather than a flat
    # ".".count() heuristic: `transformer_blocks.{i}.` sits one level deep,
    # `token_refiner.refiner_blocks.{i}.` is NESTED two levels deep under
    # `token_refiner` (confirmed from the HF safetensors index: 50 main
    # blocks + 2 refiner blocks under `token_refiner.refiner_blocks` in the
    # real checkpoint). num_layers=2 -> 2 main blocks, num_refiner_layers=1
    # -> 1 refiner block.
    main_blocks = {
        name
        for name, _ in model.named_modules()
        if name.startswith("transformer_blocks.") and name.count(".") == 1
    }
    refiner_blocks = {
        name
        for name, _ in model.named_modules()
        if name.startswith("token_refiner.refiner_blocks.") and name.count(".") == 2
    }
    assert len(main_blocks) == 2, f"expected 2 main blocks, walked {sorted(main_blocks)}"
    assert len(refiner_blocks) == 1, f"expected 1 refiner block, walked {sorted(refiner_blocks)}"
    # Cross-check against the module's own bookkeeping, so this test would
    # also fail loudly if `transformer_blocks`/`token_refiner.refiner_blocks`
    # ever stopped being where the blocks live.
    assert len(model.transformer_blocks) == 2
    assert len(model.token_refiner.refiner_blocks) == 1


def test_tiny_transformer_forward_is_finite():
    model = build_tiny_transformer()
    patch_size = TINY_TRANSFORMER_KWARGS["patch_size"]

    # 1 latent frame, 4x4 latent grid -> patchifies cleanly by [1, 2, 2] into
    # 4 video tokens of width in_channels * prod(patch_size) = 24*1*2*2 = 96.
    latents = torch.randn(1, TINY_TRANSFORMER_KWARGS["in_channels"], 1, 4, 4)
    video_hidden_states = _patchify_video(latents, patch_size)
    num_video_tokens = video_hidden_states.shape[1]
    assert num_video_tokens == 4

    num_text_tokens = 8
    num_audio_tokens = 2
    encoder_hidden_states = torch.randn(1, num_text_tokens, TINY_TRANSFORMER_KWARGS["text_dim"])
    audio_hidden_states = torch.randn(1, num_audio_tokens, TINY_TRANSFORMER_KWARGS["audio_in_channels"])

    # Pack text, video, audio rows into one sequence (H3 self-attends over a
    # single packed document -- see MiniMaxH3AttnProcessor's docstring: no
    # cross-attention exists to feed these modalities through separately).
    text_indices = torch.arange(0, num_text_tokens, dtype=torch.long)
    video_indices = torch.arange(num_text_tokens, num_text_tokens + num_video_tokens, dtype=torch.long)
    audio_indices = torch.arange(
        num_text_tokens + num_video_tokens,
        num_text_tokens + num_video_tokens + num_audio_tokens,
        dtype=torch.long,
    )
    seq_len = num_text_tokens + num_video_tokens + num_audio_tokens

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[video_indices] = 0
    token_tags[audio_indices] = 2

    # Two DISTINCT noise levels in one forward -- text + audio are the clean
    # conditioning rows (timestep index 0), the target video is the noisy
    # row being denoised (timestep index 1). This genuinely exercises the
    # per-row `timestep_indices` -> `adaln_indices` addressing
    # (`timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags`) rather than
    # degenerating to the single-timestep case.
    timestep = torch.tensor([0.2, 0.8])
    timestep_indices = torch.empty(seq_len, dtype=torch.long)
    timestep_indices[text_indices] = 0
    timestep_indices[video_indices] = 1
    timestep_indices[audio_indices] = 0

    # (t, h, w) rope grid. Real coordinates for the video rows, matching
    # _patchify_video's row order (frame outermost, then height, then
    # width); text/audio rows carry no spatial meaning in H3 so 0 is a
    # structurally-valid placeholder.
    position_ids = torch.zeros(seq_len, 3)
    video_grid = torch.tensor(
        [[f, h, w] for f in range(1) for h in range(2) for w in range(2)],
        dtype=torch.float32,
    )
    position_ids[video_indices] = video_grid

    with torch.no_grad():
        out = model(
            hidden_states=video_hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
        )

    sample = out.sample if hasattr(out, "sample") else out[0]
    audio_sample = out.audio_sample if hasattr(out, "audio_sample") else out[1]

    assert sample.shape == video_hidden_states.shape
    assert audio_sample.shape == audio_hidden_states.shape
    assert torch.isfinite(sample).all(), "vendored forward produced NaN/Inf (video)"
    assert torch.isfinite(audio_sample).all(), "vendored forward produced NaN/Inf (audio)"
