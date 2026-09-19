"""minimax_h3 — the packed-sequence layout (plan row 1.3; ECOSYSTEM §6 `H3PackedLayout`).

# Method re-implemented from ostris/ai-toolkit@561a0236 (MIT)
# `extensions_built_in/diffusion_models/minimax_h3/src/packing.py`; no line copied.

One H3 forward runs over ONE packed 1-D sequence per batch item:

    [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]

The transformer (diffusers 0.40.0 `MiniMaxH3Transformer3DModel`) does not
build this: it checks the structural shapes (`:585-591`) and embeds whatever
`timestep` arrives (`:613`). This module owns the geometry — where every row
sits, its `(t, h, w)` rotary coordinate, its modality tag and its timestep
SLOT — and the patchify / audio packing whose inverses the tests round-trip.

Contract facts the released checkpoint depends on (research §3.3):

* rotary coordinates are float64 built with numpy ``linspace(endpoint=False)``
  — ``start + arange(n)·(stop−start)/n`` — which is NOT bit-identical to
  ``torch.linspace``; the last ulp is part of the audio/video alignment;
* video and audio share one 40-units/second clock: a latent frame spans
  ``5/3 × frames_per_latent`` units with the ``(1, 4, 4, 4, 4)`` chunk pattern
  (17 pixel frames → 5 latents), audio advances one unit per latent (40/s);
* the media clock starts AFTER the text rows: ``media_origin = num_text +
  media_advance`` — prompt length shifts every media coordinate;
* patch ``(1, 2, 2)``, rows frame-major then row-major, features ``[c, pt, ph, pw]``;
  audio rows channel-major ``(B, 2, C, T) → (B, 2·T, C)``;
* token tags ``0 video / 1 text / 2 audio`` index the AdaLN table (a checkpoint
  contract), and each row takes ONE of ≤ 3 distinct timesteps by SLOT:
  ``0`` target video (text rows inherit it), ``1`` target audio, ``2`` keyframe
  conditioning rows (``t_c``). The reference mode adds slot ``3`` (reference
  soundtracks) — not built in PR1.

``H3PackedLayout.version`` is load-bearing: it is part of the latent-cache
key, so a row-order change invalidates every cached latent (D10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

LAYOUT_VERSION = 1

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

SLOT_VIDEO = 0
SLOT_AUDIO = 1
SLOT_CONDITION_VIDEO = 2
SLOT_CONDITION_AUDIO = 3

AUDIO_CHANNELS = 2

# The shared rotary clock: 40 units per second = 24 fps × 5/3 units per pixel
# frame; a latent frame covers (1, 4, 4, 4, 4) pixel frames within its chunk.
_UNITS_PER_PIXEL_FRAME = 5.0 / 3.0
_PIXEL_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_SPATIAL_SCALE = 32.0

_PATCH = (1, 2, 2)


@dataclass(frozen=True)
class H3Geometry:
    """The inputs the layout is DERIVED from (nothing else)."""

    num_text: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int
    patch_size: tuple[int, int, int] = _PATCH
    keyframe_anchors: tuple[str, ...] = ()  # ("first",) / ("last",) / both — fl2va
    media_advance: float = 0.0  # reference blocks advance the media clock (ref2va)


@dataclass(frozen=True)
class H3PackedLayout:
    """Structural description of one packed sequence (shared by the batch)."""

    seq_len: int
    position_ids: torch.Tensor  # (S, 3) float64 — (t, h, w)
    token_tags: torch.Tensor  # (S,) int64 — 0 video / 1 text / 2 audio
    timestep_slot: torch.Tensor  # (S,) int64 — index into the distinct timestep set
    video_indices: torch.Tensor  # (n_v,) int64 — condition rows first, then target rows
    audio_indices: torch.Tensor  # (n_a,) int64 — condition rows first, then target rows
    text_indices: torch.Tensor  # (n_t,) int64
    video_grid: tuple[int, int, int]  # (T, H, W) latent grid of the TARGET video
    patch_size: tuple[int, int, int]
    num_condition_video_rows: int
    num_condition_audio_rows: int
    media_origin: float
    version: int = LAYOUT_VERSION


# ── Patchify / audio packing (and their inverses) ────────────────────────


def patchify_video(latents: torch.Tensor, patch_size: tuple[int, int, int] = _PATCH) -> torch.Tensor:
    """``(B, C, T, H, W) → (B, N, C·pt·ph·pw)``: frame-major then row-major
    rows, feature order ``[c, pt, ph, pw]`` (the `proj_in` contract)."""
    pt, ph, pw = patch_size
    b, c, t, h, w = latents.shape
    if t % pt or h % ph or w % pw:
        raise ValueError(f"latent grid {(t, h, w)} is not divisible by patch {patch_size}")
    x = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, -1, c * pt * ph * pw).contiguous()


def unpatchify_video(rows: torch.Tensor, layout: H3PackedLayout) -> torch.Tensor:
    """Inverse of :func:`patchify_video` for the layout's TARGET video grid."""
    t, h, w = layout.video_grid
    pt, ph, pw = layout.patch_size
    b, n, d = rows.shape
    expected = (t // pt) * (h // ph) * (w // pw)
    if n != expected:
        raise ValueError(f"{n} video rows for a {(t, h, w)} grid (expected {expected})")
    c = d // (pt * ph * pw)
    x = rows.reshape(b, t // pt, h // ph, w // pw, c, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(b, c, t, h, w).contiguous()


def pack_audio(latents: torch.Tensor) -> torch.Tensor:
    """``(B, 2, C, T) → (B, 2·T, C)``: all T latents of channel 0, then channel 1."""
    b, ch, c, t = latents.shape
    if ch != AUDIO_CHANNELS:
        raise ValueError(f"H3 audio latents are stereo ({AUDIO_CHANNELS} channels), got {ch}")
    return latents.permute(0, 1, 3, 2).reshape(b, ch * t, c).contiguous()


def unpack_audio(rows: torch.Tensor, audio_latents: int) -> torch.Tensor:
    """Inverse of :func:`pack_audio`."""
    b, n, c = rows.shape
    if n != AUDIO_CHANNELS * audio_latents:
        raise ValueError(f"{n} audio rows for {audio_latents} latents × {AUDIO_CHANNELS} channels")
    return rows.reshape(b, AUDIO_CHANNELS, audio_latents, c).permute(0, 1, 3, 2).contiguous()


# ── Rotary grids ─────────────────────────────────────────────────────────


def spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One spatial axis, area-normalised and centred: the axis covers
    ``dim / sqrt_area`` of the unit square around 0.5, sampled once per patch
    with numpy ``linspace(endpoint=False)`` and scaled by 32. float64."""
    extent = dim / sqrt_area
    start = (1.0 - extent) / 2.0
    grid = np.linspace(start, start + extent, dim // patch, endpoint=False) * _SPATIAL_SCALE
    return torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float64))


def latent_frame_spans(num_latent_frames: int) -> np.ndarray:
    """Rotary units each latent frame spans: ``5/3 × (1, 4, 4, 4, 4)[i mod 5]``."""
    pattern = np.array(_PIXEL_FRAMES_PER_LATENT, dtype=np.float64)
    reps = pattern[np.arange(num_latent_frames) % len(pattern)]
    return reps * _UNITS_PER_PIXEL_FRAME


def temporal_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Start time of each latent frame: ``origin`` plus the cumulative spans."""
    spans = latent_frame_spans(num_latent_frames)
    starts = np.concatenate([[0.0], np.cumsum(spans[:-1])]) + origin
    return torch.from_numpy(np.ascontiguousarray(starts, dtype=np.float64))


def temporal_span(num_latent_frames: int) -> float:
    """Total rotary units the target video covers (numpy pairwise sum — the
    reference builds the "last" keyframe anchor this way and the summation
    order shows in the last ulp)."""
    return float(latent_frame_spans(num_latent_frames).sum())


# ── The layout ───────────────────────────────────────────────────────────


def build_layout(geometry: H3Geometry) -> H3PackedLayout:
    """Derive the packed layout from the geometry alone (D10: derivable,
    never persisted)."""
    pt, ph, pw = geometry.patch_size
    t, h, w = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    if t % pt or h % ph or w % pw:
        raise ValueError(f"latent grid {(t, h, w)} is not divisible by patch {geometry.patch_size}")
    for anchor in geometry.keyframe_anchors:
        if anchor not in ("first", "last"):
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")

    num_text = int(geometry.num_text)
    rows_per_frame = (h // ph) * (w // pw)
    num_cond_video = len(geometry.keyframe_anchors) * rows_per_frame
    num_cond_audio = 0  # reference soundtracks: ref2va, not built in PR1
    num_audio = geometry.audio_latents * AUDIO_CHANNELS
    num_video = (t // pt) * rows_per_frame

    cond_start = num_text
    audio_start = cond_start + num_cond_video + num_cond_audio
    video_start = audio_start + num_audio
    seq_len = video_start + num_video

    media_origin = float(num_text) + float(geometry.media_advance)

    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    # Text rows: their own index on the time axis, no spatial coordinate.
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)

    sqrt_area = math.sqrt(h * w)
    h_grid = spatial_grid(h, ph, sqrt_area)
    w_grid = spatial_grid(w, pw, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame_grid = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)  # (rows_per_frame, 2)

    # Keyframe conditioning rows: pinned at the first / last target frame's
    # rotary time, on the target's spatial grid.
    for i, anchor in enumerate(geometry.keyframe_anchors):
        if anchor == "first":
            anchor_time = media_origin
        else:
            anchor_time = media_origin + temporal_span(t) - _UNITS_PER_PIXEL_FRAME
        rows = slice(cond_start + i * rows_per_frame, cond_start + (i + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Target audio rows: channel-major, one unit per latent from the media
    # origin, no height coordinate, width pinned to the grid extremes per channel.
    audio_time = media_origin + torch.arange(geometry.audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((geometry.audio_latents,), float(w_grid[0]), dtype=torch.float64),
            torch.full((geometry.audio_latents,), float(w_grid[-1]), dtype=torch.float64),
        ]
    )

    # Target video rows: frame-major then row-major.
    video_pos = torch.empty(t // pt, rows_per_frame, 3, dtype=torch.float64)
    video_pos[:, :, 0] = temporal_grid(t // pt, media_origin)[:, None]
    video_pos[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_pos.reshape(-1, 3)

    text_indices = torch.arange(num_text, dtype=torch.long)
    cond_video_indices = torch.arange(cond_start, cond_start + num_cond_video, dtype=torch.long)
    video_indices = torch.cat([cond_video_indices, torch.arange(video_start, seq_len, dtype=torch.long)])
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = TEXT_TAG
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    timestep_slot = torch.full((seq_len,), SLOT_VIDEO, dtype=torch.long)  # text inherits t_v
    timestep_slot[audio_indices] = SLOT_AUDIO
    timestep_slot[cond_video_indices] = SLOT_CONDITION_VIDEO

    return H3PackedLayout(
        seq_len=seq_len,
        position_ids=position_ids,
        token_tags=token_tags,
        timestep_slot=timestep_slot,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        video_grid=(t, h, w),
        patch_size=geometry.patch_size,
        num_condition_video_rows=num_cond_video,
        num_condition_audio_rows=num_cond_audio,
        media_origin=media_origin,
    )


def num_distinct_timesteps(layout: H3PackedLayout) -> int:
    """How many distinct timestep values this layout addresses (``max slot + 1``)."""
    return int(layout.timestep_slot.max().item()) + 1


# ── The forward through the packed layout ────────────────────────────────


def packed_forward(
    transformer: Any,
    layout: H3PackedLayout,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    text_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    **extra: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the transformer on the packed layout.

    ``timesteps`` is the DISTINCT set, one value per slot ``(t_v, t_a[, t_c])``
    in ``[0, 1]`` unscaled; every row addresses it through
    ``layout.timestep_slot``. Returns ``(video_rows_out, audio_rows_out)`` in
    the row order of ``video_indices`` / ``audio_indices``.
    """
    needed = num_distinct_timesteps(layout)
    if timesteps.ndim != 1 or timesteps.numel() < needed:
        raise ValueError(
            f"the layout addresses {needed} distinct timesteps; got a tensor of shape "
            f"{tuple(timesteps.shape)} — pass the distinct set, never per-row values"
        )
    device = video_rows.device
    return transformer(
        hidden_states=video_rows,
        audio_hidden_states=audio_rows,
        encoder_hidden_states=text_embeds,
        timestep=timesteps.to(device),
        timestep_indices=layout.timestep_slot.to(device),
        token_tags=layout.token_tags.to(device),
        position_ids=layout.position_ids.to(device),
        video_indices=layout.video_indices.to(device),
        audio_indices=layout.audio_indices.to(device),
        text_indices=layout.text_indices.to(device),
        return_dict=False,
        **extra,
    )
