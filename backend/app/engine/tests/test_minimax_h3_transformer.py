"""minimax_h3 transformer contract — on the INSTALLED diffusers classes.

Plan row 1.8 retired the family's ``vendor/`` fork (a divergent copy of
diffusers PR #14355 at SHA 245d78fb). diffusers 0.40.0 — the pinned floor in
``requirements.txt`` — ships the whole H3 stack natively, so this module now
pins three contracts:

  1. Installed-dependency contract: the four H3 classes import from
     ``diffusers`` and the installed version equals the ``requirements.txt``
     pin (the intent the deleted ``vendor/REVISION`` used to carry).
  2. Vendor parity, frozen: the vendored transformer's fp32 CPU forward on
     the tiny arch was recorded ONCE — same seed, same state dict, same
     inputs — into ``fixtures/h3_vendor_parity.npz`` before the fork was
     deleted; the upstream class must reproduce it bit-exact (``torch.equal``)
     for the video AND the audio output. Grad checkpointing arming both
     sites on the upstream class is pinned by the convention suite
     (``checkpointing_arms_both_sites``, row 1.5).
  3. Source guard: no ``families.minimax_h3.vendor`` import anywhere under
     ``backend/`` and the directory itself is gone.

The tiny config is deliberately un-real. It preserves the relationships that
matter (patch_size vs spatial dims, in_channels vs VAE latent width) while
shrinking everything else, so this runs on CPU in seconds with no weights.

Reconciled against the REAL ``MiniMaxH3Transformer3DModel`` (``__init__``
and ``.forward``), which differ from a naive first guess in two load-bearing
ways:

  * ``MiniMaxH3Transformer3DModel`` does NOT patchify video latents itself --
    ``proj_in`` is a plain ``nn.Linear(video_patch_dim, hidden_size)``. The
    driver's ``packing.py`` patchifies before calling forward.
    ``_patchify_video`` below reproduces that math (matching ``patch_size``
    exactly) so this smoke test can feed a correctly-shaped tensor without
    the driver.
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

import pathlib
import re

import torch

from app.engine.models.families.minimax_h3.loader import diffusers_ships_native_h3

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parents[2]
_FAMILY_DIR = _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3"
# ``.npz`` (numpy, allow_pickle=False), not ``.pt``: the repo's .gitignore
# refuses every weight-file extension (*.pt, *.bin, *.safetensors, ...) so
# weights never enter git; a fixture must live in a format that rule admits.
_PARITY_FIXTURE = _TESTS_DIR / "fixtures" / "h3_vendor_parity.npz"
_REQUIREMENTS = _BACKEND_DIR / "requirements.txt"

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


def build_tiny_transformer():
    """A tiny CPU transformer (the INSTALLED diffusers class) for structural
    tests. Shared with the definitions test so both derive targets from the
    SAME structure."""
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER_KWARGS).eval()


def _patchify_video(latents: torch.Tensor, patch_size: list[int]) -> torch.Tensor:
    """Reproduce the patchify math the driver's ``packing.py`` does before
    calling ``forward`` -- the transformer's ``proj_in`` is a plain
    ``nn.Linear(video_patch_dim, hidden_size)``, it does not patchify raw
    ``(batch, channels, frames, height, width)`` latents itself.

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


def packed_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    """The two-noise-level packed sequence every forward test here feeds.

    Seeded so the parity fixture (recorded from the vendored class before its
    deletion) and the live upstream forward see the SAME inputs.
    """
    torch.manual_seed(seed)
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

    return {
        "hidden_states": video_hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def _outputs(out) -> tuple[torch.Tensor, torch.Tensor]:
    sample = out.sample if hasattr(out, "sample") else out[0]
    audio_sample = out.audio_sample if hasattr(out, "audio_sample") else out[1]
    return sample, audio_sample


def _load_parity_fixture() -> dict:
    """The vendored forward, recorded once (seed 0, fp32, CPU, torch 2.12.1)
    from the fork at diffusers SHA 245d78fb before its deletion. Flat npz:
    ``state_dict/<key>``, ``inputs/<key>``, ``sample``, ``audio_sample`` and a
    ``meta_json`` string carrying ``kwargs`` + provenance."""
    import json

    import numpy as np

    with np.load(_PARITY_FIXTURE, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        tensors = {k: torch.from_numpy(np.ascontiguousarray(npz[k])) for k in npz.files if k != "meta_json"}
    split = lambda prefix: {  # noqa: E731
        k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)
    }
    return {
        "kwargs": meta["kwargs"],
        "provenance": meta["provenance"],
        "state_dict": split("state_dict/"),
        "inputs": split("inputs/"),
        "sample": tensors["sample"],
        "audio_sample": tensors["audio_sample"],
    }


def _pinned_diffusers_version() -> str:
    text = _REQUIREMENTS.read_text(encoding="utf-8")
    match = re.search(r"^diffusers==([^\s#]+)", text, flags=re.MULTILINE)
    assert match, "requirements.txt no longer pins diffusers with =="
    return match.group(1)


# ── 1. Installed-dependency contract ─────────────────────────────────────


def test_installed_dependency_contract():
    """The four H3 classes come from the INSTALLED diffusers, whose version is
    the requirements.txt pin — the known-revision intent the deleted
    ``vendor/REVISION`` carried now lives in the dependency pin."""
    import diffusers
    from packaging.version import Version

    pinned = _pinned_diffusers_version()
    assert Version(diffusers.__version__) == Version(pinned), (
        f"installed diffusers {diffusers.__version__} != requirements.txt pin {pinned}"
    )
    assert diffusers_ships_native_h3(), (
        f"{pinned} exports MiniMaxH3Transformer3DModel; the probe says it does not"
    )
    for name in (
        "MiniMaxH3Transformer3DModel",
        "AutoencoderKLMiniMaxH3",
        "AutoencoderKLMiniMaxH3Audio",
        "MiniMaxH3Scheduler",
    ):
        cls = getattr(diffusers, name)
        assert cls.__module__.startswith("diffusers."), f"{name} is not diffusers' own: {cls.__module__}"


def test_native_probe_negative_control():
    """The probe is a real lookup, not a constant: a name diffusers does not
    export answers ``False``."""
    assert diffusers_ships_native_h3("MiniMaxH3ClassThatDoesNotExist") is False
    assert diffusers_ships_native_h3("MiniMaxH3Transformer3DModel") is True


# ── 2. Vendor parity, frozen ─────────────────────────────────────────────


def test_vendored_and_upstream_agree_bit_exact():
    """The vendored fork's fp32 CPU forward (recorded into the fixture before
    ``vendor/`` was deleted: seed 0, the tiny arch, ``packed_inputs(0)``)
    equals the installed class's forward bit-exact, video AND audio."""
    from diffusers import MiniMaxH3Transformer3DModel

    assert _PARITY_FIXTURE.is_file(), f"parity fixture missing: {_PARITY_FIXTURE}"
    fixture = _load_parity_fixture()
    assert fixture["kwargs"] == TINY_TRANSFORMER_KWARGS, "fixture arch != TINY_TRANSFORMER_KWARGS"

    model = MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER_KWARGS).eval()
    missing, unexpected = model.load_state_dict(fixture["state_dict"], strict=True)
    assert not missing and not unexpected

    inputs = packed_inputs(0)
    assert set(fixture["inputs"]) == set(inputs), "fixture inputs != packed_inputs keys"
    for key, recorded in fixture["inputs"].items():
        assert torch.equal(inputs[key], recorded), f"packed_inputs drifted from the fixture at {key!r}"

    with torch.no_grad():
        sample, audio_sample = _outputs(model(**inputs))

    assert sample.dtype == torch.float32 and audio_sample.dtype == torch.float32
    assert torch.equal(sample, fixture["sample"]), (
        f"video output differs from the vendored forward: max |Δ| = "
        f"{(sample - fixture['sample']).abs().max().item():.3e}"
    )
    assert torch.equal(audio_sample, fixture["audio_sample"]), (
        f"audio output differs from the vendored forward: max |Δ| = "
        f"{(audio_sample - fixture['audio_sample']).abs().max().item():.3e}"
    )


# ── 3. Source guard ──────────────────────────────────────────────────────


def test_no_vendor_import_remains():
    """No module under ``backend/`` imports the retired fork, and the fork's
    directory is gone (a stale ``vendor/`` would still be importable)."""
    assert not (_FAMILY_DIR / "vendor").exists(), "families/minimax_h3/vendor/ still exists"
    pattern = re.compile(r"families\.minimax_h3\.vendor|minimax_h3/vendor")
    offenders = []
    for path in _BACKEND_DIR.rglob("*.py"):
        if "venv" in path.parts or "__pycache__" in path.parts or path == pathlib.Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_BACKEND_DIR)}:{lineno}")
    assert not offenders, f"vendor references remain: {offenders}"


def test_loader_uses_diffusers_h3_classes():
    """Every H3 model class in the manifest resolves inside the installed
    ``diffusers`` package — no dotted path into a family-owned copy."""
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    definition = ModelRegistry._definitions["minimax-h3-t2va"]
    loader = MiniMaxH3Loader.__new__(MiniMaxH3Loader)
    specs = {s.key: s for s in loader.get_component_manifest(definition)}
    for key in ("vae", "audio_vae", "transformer"):
        assert specs[key].hf_class.startswith("diffusers."), f"{key}: {specs[key].hf_class}"
        # The path must actually resolve — a typo fails only after ~100 GB otherwise.
        assert MiniMaxH3Loader._import_class(specs[key].hf_class).__module__.startswith("diffusers.")


# ── Structural smoke (kept from the vendor era, now on the installed class) ──


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
    inputs = packed_inputs(0)
    with torch.no_grad():
        sample, audio_sample = _outputs(model(**inputs))

    assert sample.shape == inputs["hidden_states"].shape
    assert audio_sample.shape == inputs["audio_hidden_states"].shape
    assert torch.isfinite(sample).all(), "forward produced NaN/Inf (video)"
    assert torch.isfinite(audio_sample).all(), "forward produced NaN/Inf (audio)"
