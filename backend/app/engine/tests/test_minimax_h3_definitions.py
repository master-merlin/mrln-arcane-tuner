"""minimax_h3 definition-YAML pins.

Guards the boogu_image/dreamlite precedent: a definition whose
``lora_targetable_modules`` does not EXACTLY match what the real vendored
transformer offers either silently starts empty and gets overwritten by the
introspector's exhaustive catalog at first model load, or omits real
weight-bearing Linears — leaving part of the network un-adapted at training
time, which only shows up as a weak LoRA after a full GPU run.

Method: instantiate the vendored transformer TINY on CPU, walk named_modules()
to discover the real Linear suffix set, and assert the shipped YAML matches
EXACTLY.
"""

from __future__ import annotations

import pathlib

import torch.nn as nn
import yaml

from app.engine.tests.test_minimax_h3_vendor import build_tiny_transformer

DEF_IDS = ("minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va")

_DEFINITIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models" / "families" / "minimax_h3" / "definitions"
)

# Real checkpoint block counts — transformer/config.json, verified 2026-08-05.
NUM_LAYERS = 50
NUM_REFINER_LAYERS = 2


def _load(def_id: str) -> dict:
    for path in _DEFINITIONS_DIR.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data.get("id") == def_id:
            return data
    raise AssertionError(f"no definition YAML with id {def_id!r}")


def _real_linear_suffixes() -> set[str]:
    model = build_tiny_transformer()
    suffixes = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            parts = name.split(".")
            idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
            if idx is not None:
                suffixes.add(".".join(parts[idx + 1:]))
    return suffixes


def _real_refiner_linear_suffixes() -> set[str]:
    """Same derivation as ``_real_linear_suffixes`` but restricted to modules
    under ``token_refiner.refiner_blocks`` — main blocks and refiner blocks
    collapse to the same suffix shape (both index at the first digit
    segment), so this is the only way to see refiner structure in isolation.
    """
    model = build_tiny_transformer()
    suffixes = set()
    for name, mod in model.named_modules():
        if not name.startswith("token_refiner.refiner_blocks."):
            continue
        if isinstance(mod, nn.Linear):
            parts = name.split(".")
            idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
            if idx is not None:
                suffixes.add(".".join(parts[idx + 1:]))
    return suffixes


# Linear-bearing suffixes present in the REAL checkpoint, read from
# transformer/diffusion_pytorch_model.safetensors.index.json on 2026-08-08.
# adaln_proj.linear is real and deliberately NOT targeted — see
# test_adaln_branch_is_excluded_from_targeting for the rationale.
CHECKPOINT_BLOCK_LINEARS = {
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "ff.net.0.proj", "ff.net.2", "adaln_proj.linear",
}
CHECKPOINT_REFINER_LINEARS = CHECKPOINT_BLOCK_LINEARS - {"adaln_proj.linear"}


def test_all_three_definitions_exist():
    for def_id in DEF_IDS:
        assert _load(def_id)["family"] == "minimax_h3"


def test_vendored_class_module_names_match_the_checkpoint():
    """The vendored class and the checkpoint must agree on module naming, in
    BOTH directions.

    A missing module means ``from_pretrained`` either raises or silently
    skips weights after a ~62 GB download. A vendored class that INVENTS
    extra Linears absent from the checkpoint is just as dangerous the other
    way: it would load with random-initialized weights that the checkpoint
    never trained, and nothing about that failure mode is loud. Catching
    either direction here costs a CPU instantiate.
    """
    derived = _real_linear_suffixes()
    missing = CHECKPOINT_BLOCK_LINEARS - derived
    extra = derived - CHECKPOINT_BLOCK_LINEARS
    assert derived == CHECKPOINT_BLOCK_LINEARS, (
        f"vendored class and checkpoint disagree on module naming — "
        f"the vendor drop and the weights are out of sync.\n"
        f"  vendor lacks (checkpoint ships): {sorted(missing)}\n"
        f"  vendor invents (checkpoint lacks): {sorted(extra)}"
    )


def test_target_list_is_non_empty():
    """An EMPTY list is the specific failure the boogu_image/dreamlite incident
    documents: it passes every subset check vacuously, then gets silently
    overwritten by the introspector's exhaustive catalog at first model load,
    so the run adapts modules nobody curated. Pin non-emptiness explicitly."""
    for def_id in DEF_IDS:
        shipped = _load(def_id)["lora_targetable_modules"]
        assert shipped, f"{def_id} ships an EMPTY lora_targetable_modules"
        # Attention alone contributes q/k/v/out; a real list cannot be tiny.
        assert len(shipped) >= 4, f"{def_id} ships only {len(shipped)} targets"


def test_target_list_contains_no_module_that_does_not_exist():
    real = _real_linear_suffixes()
    for def_id in DEF_IDS:
        shipped = set(_load(def_id)["lora_targetable_modules"])
        missing = shipped - real
        assert not missing, f"{def_id} targets non-existent modules: {sorted(missing)}"


def test_target_list_is_exactly_the_non_adaln_checkpoint_linears():
    """The other half of EXACT: not just 'nothing fake' but 'nothing omitted'.

    A silently omitted weight-bearing Linear leaves part of the network
    un-adapted, which no unit test catches later and which only shows up as a
    weak LoRA after a full GPU run. Asserted against the REAL checkpoint key
    set, not against whatever the tiny instance happens to expose.
    """
    expected = CHECKPOINT_BLOCK_LINEARS - {"adaln_proj.linear"}
    for def_id in DEF_IDS:
        shipped = set(_load(def_id)["lora_targetable_modules"])
        assert shipped == expected, (
            f"{def_id} target list drifted from the checkpoint's non-AdaLN "
            f"Linears.\n  missing: {sorted(expected - shipped)}"
            f"\n  unexpected: {sorted(shipped - expected)}"
        )


def test_refiner_blocks_match_checkpoint_minus_adaln():
    """Refiner blocks carry the checkpoint's block Linears MINUS
    ``adaln_proj.linear`` — refiners have no per-row AdaLN table, only the
    main transformer blocks do (see the module docstring). A vendored
    refiner block that gained or lost a Linear relative to this would
    silently escape every other test here, since those only check the
    combined main+refiner suffix set or the shipped YAML, never refiner
    structure in isolation.
    """
    derived_refiner = _real_refiner_linear_suffixes()
    assert derived_refiner == CHECKPOINT_REFINER_LINEARS, (
        f"refiner block Linears drifted from the checkpoint.\n"
        f"  missing: {sorted(CHECKPOINT_REFINER_LINEARS - derived_refiner)}\n"
        f"  unexpected: {sorted(derived_refiner - CHECKPOINT_REFINER_LINEARS)}"
    )
    assert "adaln_proj.linear" not in derived_refiner, (
        "refiner blocks must NOT carry adaln_proj — they have no per-row "
        "AdaLN table, unlike the main transformer blocks"
    )


def test_adaln_branch_is_excluded_from_targeting():
    # Spec 8.1 — the ~13B AdaLN/time-embed branch is timestep-conditioned,
    # not content-conditioned. Adapting it is expensive and off-objective.
    for def_id in DEF_IDS:
        shipped = set(_load(def_id)["lora_targetable_modules"])
        offenders = {s for s in shipped if "time_embed" in s or "adaln" in s.lower()}
        assert not offenders, f"{def_id} targets the AdaLN branch: {sorted(offenders)}"


def test_block_topology_covers_main_and_refiner_blocks():
    for def_id in DEF_IDS:
        topo = _load(def_id)["block_topology"]
        counts = {entry["name"]: entry["count"] for entry in topo}
        assert sum(counts.values()) == NUM_LAYERS + NUM_REFINER_LAYERS, (
            f"{def_id} block_topology sums to {sum(counts.values())}, "
            f"expected {NUM_LAYERS + NUM_REFINER_LAYERS} — refiner blocks are "
            "SEPARATE from num_layers and silently escape targeting if omitted"
        )


def test_cross_component_latent_widths_agree():
    for def_id in DEF_IDS:
        arch = _load(def_id)["architecture_params"]
        assert arch["transformer.in_channels"] == arch["vae.latent_channels"] == 24
        assert arch["transformer.audio_in_channels"] == arch["audio_vae.latent_channels"] == 32


def test_audio_latent_rate_is_derived_not_asserted():
    # 32000 / (2*4*4*5*5) == 40 Hz. A future config change that breaks this
    # must fail loudly rather than silently desync audio.
    import math

    for def_id in DEF_IDS:
        arch = _load(def_id)["architecture_params"]
        rates = arch["audio_vae.encoder_rates"]
        assert arch["audio.sampling_rate"] / math.prod(rates) == arch["audio.latent_rate"]


def test_ref2va_uses_the_separate_ref_checkpoint():
    assert _load("minimax-h3-ref2va")["architecture_params"]["transformer.subfolder"] == "transformer_ref"
    for def_id in ("minimax-h3-t2va", "minimax-h3-fl2va"):
        assert _load(def_id)["architecture_params"]["transformer.subfolder"] == "transformer"
