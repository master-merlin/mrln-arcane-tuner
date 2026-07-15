"""THE make-or-break parity gate for the Bernini-R vendored packed forward.

Bernini-R's only architectural novelty is data-side: clean condition-video
latents are **token-sequence concatenated** with the noisy target latents (NOT
channel-concat), disambiguated by a **source_id rotary phase multiply**, run
through **one full bidirectional attention window** spanning cond+target, with
the velocity prediction consumed for **target tokens only**.

The weights are 100%-stock Wan (``WanTransformer3DModel``); the vendored adapter
reuses the stock model's own submodules and only re-implements that assembly.
These tests pin the adapter against references built *independently* of the
adapter's own orchestration, so a bug in the concat order / target slice /
unpatchify / source_id composition surfaces as a numeric mismatch rather than
"trains but renders garbage" (the flowmatch-timestep / autocast-collapse class
of silent failure).

Contract: ``atol=1e-6`` in fp32 on CPU. Do NOT weaken it.

Reference construction (why it is NOT circular):

* ``test_source_id_rope_matches_upstream_complex_formula`` reconstructs the
  per-pair *complex* phase the adapter's real (cos,sin) rope encodes and asserts
  it equals upstream ``bernini/models/transformer_wan.py``'s
  ``freqs = freqs * freqs_visual_id`` computed from scratch here. The adapter
  never participates in building its own expected value.
* ``test_case_a_*`` builds the packed forward's expected output by hand from the
  model's OWN blocks (``model.blocks``), with the token concat, source_id rope
  (composed inline in this file — not via the adapter), full attention (the
  stock block already attends over the whole packed sequence), and target slice
  written out longhand. Only the model's trained submodules are shared; every
  orchestration step is duplicated independently.
* ``test_case_b_*`` (degenerate t2v) drives a single target stream at
  ``source_id=0`` and asserts the adapter output equals the *stock*
  ``WanTransformer3DModel.forward`` numerically — proving the no-condition path
  collapses onto unmodified diffusers.
"""

from __future__ import annotations

import torch
from diffusers import WanTransformer3DModel
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from app.engine.models.families.bernini_r.vendor.transformer_forward import (
    BERNINI_ROPE_THETA,
    bernini_packed_forward,
    source_id_rope,
)

ATOL = 1e-6


def _tiny_model(seed: int = 0) -> WanTransformer3DModel:
    """A tiny, deterministic, stock-key WanTransformer3DModel (fp32, CPU)."""
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=64,
        ffn_dim=64,
        num_layers=3,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )
    return model.to(torch.float32).eval()


def _rand_latent(seed: int, frames: int = 1, h: int = 8, w: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, frames, h, w, generator=g, dtype=torch.float32)


def _rand_text(seed: int, length: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, length, 16, generator=g, dtype=torch.float32)


# ── Independent reference helpers (do NOT call the adapter's orchestration) ──


def _reference_source_id_cos_sin(
    model: WanTransformer3DModel, latent: torch.Tensor, source_id: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the source_id rope in the real domain, written out longhand here.

    Starts from the trusted *stock* ``model.rope`` base tables and multiplies in
    the source_id phase inline — deliberately NOT calling the adapter's
    ``source_id_rope`` so the packed-forward reference is independent of it.
    """
    freqs_cos, freqs_sin = model.rope(
        latent
    )  # [1, seq, 1, head_dim], repeat-interleaved
    if float(source_id) == 0.0:
        return freqs_cos, freqs_sin
    head_dim = model.config.attention_head_dim
    id_freqs = get_1d_rotary_pos_embed(
        head_dim,
        torch.tensor([float(source_id)], dtype=torch.float64),
        BERNINI_ROPE_THETA,
        use_real=False,
        repeat_interleave_real=False,
        freqs_dtype=torch.float64,
    )  # complex [1, head_dim//2]
    id_cos = (
        id_freqs.real.to(freqs_cos.dtype)
        .repeat_interleave(2, dim=1)
        .view(1, 1, 1, head_dim)
    )
    id_sin = (
        id_freqs.imag.to(freqs_sin.dtype)
        .repeat_interleave(2, dim=1)
        .view(1, 1, 1, head_dim)
    )
    new_cos = freqs_cos * id_cos - freqs_sin * id_sin
    new_sin = freqs_sin * id_cos + freqs_cos * id_sin
    return new_cos, new_sin


def _reference_packed_forward(
    model: WanTransformer3DModel,
    streams: list[torch.Tensor],
    source_ids: list[float],
    timestep: torch.Tensor,
    text: torch.Tensor,
) -> torch.Tensor:
    """Hand-rolled packed forward using the model's own blocks.

    ``streams`` is ordered ``[cond..., target]`` (target LAST, matching upstream
    ``torch.cat(cond_lats + [noisy_latent], dim=1)``). Returns the velocity for
    the target (last) stream only, unpatchified to ``[B, out_ch, F, H, W]``.
    """
    p_t, p_h, p_w = model.config.patch_size

    tokens: list[torch.Tensor] = []
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for latent, sid in zip(streams, source_ids):
        cos, sin = _reference_source_id_cos_sin(model, latent, sid)
        emb = (
            model.patch_embedding(latent).flatten(2).transpose(1, 2)
        )  # [B, thw, inner]
        tokens.append(emb)
        cos_parts.append(cos)
        sin_parts.append(sin)

    hidden_states = torch.cat(tokens, dim=1)
    rotary_emb = (torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1))

    temb, timestep_proj, enc_hs, _ = model.condition_embedder(
        timestep, text, None, timestep_seq_len=None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    for block in model.blocks:
        hidden_states = block(hidden_states, enc_hs, timestep_proj, rotary_emb)

    shift, scale = (model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    hidden_states = (
        model.norm_out(hidden_states.float()) * (1 + scale) + shift
    ).type_as(hidden_states)
    hidden_states = model.proj_out(hidden_states)

    # Target = last stream. Its token count is the tail of the packed sequence.
    target = streams[-1]
    cond_total = sum(t.shape[1] for t in tokens[:-1])
    ppf = target.shape[2] // p_t
    pph = target.shape[3] // p_h
    ppw = target.shape[4] // p_w
    tgt = hidden_states[:, cond_total:, :]
    b = target.shape[0]
    tgt = tgt.reshape(b, ppf, pph, ppw, p_t, p_h, p_w, -1)
    tgt = tgt.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return tgt.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ── Test 1: source_id rope equals the upstream COMPLEX formula ───────────────


class TestSourceIdRope:
    def test_source_id_rope_matches_upstream_complex_formula(self):
        """Adapter's real (cos,sin) rope must encode the same per-pair phase as
        upstream ``freqs = freqs * freqs_visual_id`` (complex).
        """
        model = _tiny_model()
        latent = _rand_latent(seed=1)
        head_dim = model.config.attention_head_dim
        source_id = 1.0

        # Adapter (code under test): real (cos, sin), float32.
        a_cos, a_sin = source_id_rope(model, latent, source_id)
        # Per-pair phase the stock apply_rotary_emb actually consumes.
        adapter_phase = (
            a_cos[..., 0::2] + 1j * a_sin[..., 1::2]
        )  # [1, seq, 1, head_dim//2]

        # Upstream reference (built from scratch): base complex freqs * id complex.
        # Base = stock rope's underlying complex phase == cos + i*sin of the same
        # angles (recover from the model's own cos/sin tables to share the base).
        s_cos, s_sin = model.rope(latent)
        base_phase = s_cos[..., 0::2] + 1j * s_sin[..., 1::2]
        id_freqs = get_1d_rotary_pos_embed(
            head_dim,
            torch.tensor([source_id], dtype=torch.float64),
            BERNINI_ROPE_THETA,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        ).to(base_phase.dtype)  # complex [1, head_dim//2]
        upstream_phase = base_phase * id_freqs.view(1, 1, 1, -1)

        assert torch.allclose(adapter_phase, upstream_phase, atol=1e-5), (
            "source_id rope diverges from upstream complex `freqs * freqs_visual_id`; "
            f"max|diff|={(adapter_phase - upstream_phase).abs().max().item():.3e}"
        )

    def test_source_id_zero_is_identity(self):
        """source_id=0 must leave the stock rope byte-unchanged (phase == 1)."""
        model = _tiny_model()
        latent = _rand_latent(seed=2)
        s_cos, s_sin = model.rope(latent)
        a_cos, a_sin = source_id_rope(model, latent, 0.0)
        assert torch.equal(a_cos, s_cos)
        assert torch.equal(a_sin, s_sin)


# ── Test A: packed forward vs independent hand-rolled reference ──────────────


class TestPackedForwardParity:
    def test_case_a_v2v_matches_reference(self):
        """One condition stream (source_id=1) + target (source_id=0): the adapter
        packed forward must match the independently-assembled reference to 1e-6.
        """
        model = _tiny_model()
        cond = _rand_latent(seed=10)
        target = _rand_latent(seed=11)
        text = _rand_text(seed=12)
        timestep = torch.tensor([500.0], dtype=torch.float32)

        with torch.no_grad():
            got = bernini_packed_forward(
                model,
                cond_latents=[cond],
                cond_source_ids=[1.0],
                target_latent=target,
                timestep=timestep,
                encoder_hidden_states=text,
                return_dict=False,
            )
            got = got[0] if isinstance(got, tuple) else got
            expected = _reference_packed_forward(
                model, [cond, target], [1.0, 0.0], timestep, text
            )

        assert got.shape == target.shape, (
            f"target-slice shape {got.shape} != {target.shape}"
        )
        maxdiff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=ATOL), (
            f"packed forward diverges from reference; max|diff|={maxdiff:.3e}"
        )

    def test_case_a_multiframe_matches_reference(self):
        """Same, with F>1 so the temporal rope axis is exercised (4n+1 → F=5)."""
        model = _tiny_model()
        cond = _rand_latent(seed=20, frames=5)
        target = _rand_latent(seed=21, frames=5)
        text = _rand_text(seed=22)
        timestep = torch.tensor([250.0], dtype=torch.float32)

        with torch.no_grad():
            got = bernini_packed_forward(
                model, [cond], [1.0], target, timestep, text, return_dict=False
            )
            got = got[0] if isinstance(got, tuple) else got
            expected = _reference_packed_forward(
                model, [cond, target], [1.0, 0.0], timestep, text
            )

        maxdiff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=ATOL), (
            f"multiframe packed forward diverges; max|diff|={maxdiff:.3e}"
        )

    def test_full_bidirectional_attention_cond_influences_target(self):
        """Recon risk #1: attention is NOT per-stream isolated. Changing the
        condition latent MUST change the target prediction (one attention window
        spans cond+target).
        """
        model = _tiny_model()
        target = _rand_latent(seed=31)
        text = _rand_text(seed=32)
        timestep = torch.tensor([500.0], dtype=torch.float32)

        with torch.no_grad():
            out_a = bernini_packed_forward(
                model,
                [_rand_latent(seed=30)],
                [1.0],
                target,
                timestep,
                text,
                return_dict=False,
            )[0]
            out_b = bernini_packed_forward(
                model,
                [_rand_latent(seed=99)],
                [1.0],
                target,
                timestep,
                text,
                return_dict=False,
            )[0]

        assert not torch.allclose(out_a, out_b, atol=1e-4), (
            "target prediction is invariant to the condition stream — attention is "
            "wrongly per-stream isolated (should be one full window over cond+target)."
        )


# ── Test B: degenerate t2v collapses onto stock diffusers ────────────────────


class TestDegenerateT2V:
    def test_case_b_no_condition_equals_stock_forward(self):
        """source_id=0, no condition stream → adapter output == stock
        ``WanTransformer3DModel.forward`` numerically (atol=1e-6).
        """
        model = _tiny_model()
        target = _rand_latent(seed=40, frames=5)
        text = _rand_text(seed=41)
        timestep = torch.tensor([600.0], dtype=torch.float32)

        with torch.no_grad():
            got = bernini_packed_forward(
                model,
                cond_latents=[],
                cond_source_ids=[],
                target_latent=target,
                timestep=timestep,
                encoder_hidden_states=text,
                return_dict=False,
            )[0]
            stock = model(
                hidden_states=target,
                timestep=timestep,
                encoder_hidden_states=text,
                return_dict=False,
            )[0]

        maxdiff = (got - stock).abs().max().item()
        assert torch.allclose(got, stock, atol=ATOL), (
            f"degenerate t2v diverges from stock WanTransformer3DModel.forward; "
            f"max|diff|={maxdiff:.3e}"
        )
