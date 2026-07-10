"""boogu_image ecosystem LoRA conversion — Boogu-adapted counterpart of the
vendored (stock-Lumina2) ``_convert_non_diffusers_lumina2_lora_to_diffusers``.

Task 7 MUST-FIX (carried from the Task 1 review): the vendored
``vendor/lora_conversion.py`` is byte-faithful diffusers 0.39 and hardcodes
stock Lumina2's non-GQA qkv split — it cannot be used as-is against Boogu's
real geometry or module names. This module documents WHY (with upstream
file:line evidence) and provides the Boogu-adapted forward (ecosystem ->
diffusers) and reverse (diffusers -> ecosystem) converters actually usable
by this family's saver/portability tests.

## Evidence trail: what the upstream mixin actually does with
``diffusion_model.*``-prefixed state dicts

``BooguImageLoraLoaderMixin.lora_state_dict()``
(``.agent/workdir/sdd-boogu/upstream/boogu/pipelines/lora_pipeline.py:164-167``)::

    # conversion.
    non_diffusers = any(k.startswith("diffusion_model.") for k in state_dict)
    if non_diffusers:
        state_dict = _convert_non_diffusers_lumina2_lora_to_diffusers(state_dict)

ANY checkpoint whose keys start with ``"diffusion_model."`` — i.e. EXACTLY
the house ``GenericLoRASaver`` convention this family's ``BooguImageSaver``
(``saver.py``) uses — is unconditionally routed through the vendored Lumina2
conversion helper before being handed to
``load_lora_into_transformer -> transformer.load_lora_adapter(state_dict,
prefix="transformer")`` (``lora_pipeline.py:361-427``, esp. :419-427).

Naming scheme: the vendored converter pops/writes ``.lora_A.weight`` /
``.lora_B.weight`` suffixes throughout (``vendor/lora_conversion.py:50-51,
58-63, 67-72, 75-80``) — i.e. ai-toolkit/PEFT-style naming, NOT kohya
``lora_down``/``lora_up``. **No key rename is required on that front** —
``GenericLoRASaver``'s output already matches upstream's expected suffix
scheme byte for byte; the MUST-FIX is entirely about the *module-path*
mapping (qkv split + block names), not the naming *scheme*.

## Why the vendored converter cannot be used verbatim

1. **Hardcoded stock Lumina2 qkv split.** ``vendor/lora_conversion.py:54``
   splits a FUSED ``attention.qkv`` LoRA output
   (``torch.split(lora_up, [2304, 768, 768], dim=0)``) — stock Lumina2's
   widths. Boogu's real checkpoint (``definitions/base.yaml
   architecture_params``) is GQA: ``num_attention_heads=28,
   num_kv_heads=7, attention_head_dim=120`` -> ``to_q width = 28*120 =
   3360``, ``to_k/to_v width = 7*120 = 840``. Using the stock split against
   a Boogu tensor produces WRONG shapes / silently wrong weight slices, not
   even a clean crash.

2. **Block-attribute-name mismatch.** The vendored converter's third stage
   processes the core transformer stack under the literal prefix
   ``"layers"`` (``vendor/lora_conversion.py:92-95``,
   ``re.search(r"layers\\.(\\d+)\\.", key)`` + ``process_block("layers",
   ...)``) — because stock diffusers' ``Lumina2Transformer2DModel`` names
   that stack ``self.layers``. Boogu's vendored
   ``BooguImageTransformer2DModel`` names the structurally-equivalent stack
   ``self.single_stream_layers`` (``transformer_boogu.py:946``);
   ``self.layers`` IS assigned (``transformer_boogu.py:991``) but as a
   plain Python ``list``, not an ``nn.ModuleList`` — it is NOT a registered
   submodule and produces no ``named_modules()`` path, so no real
   checkpoint or LoRA-targetable module is ever named ``layers.N.*``.
   Upstream's own (unanchored) ``re.search`` pattern would actually
   MIS-MATCH the substring ``"layers.0."`` inside
   ``"single_stream_layers.0.attn..."`` and then try to
   ``state_dict.pop("layers.0.attention.qkv...")``, which does not exist in
   a Boogu export -> ``KeyError``. This module's patterns are anchored
   (``^{prefix}\\.``) and use Boogu's real attribute name
   (``single_stream_layers``) to avoid this.

3. **Architectural extensions with NO upstream analogue at all.** Boogu
   adds two whole stages stock Lumina2 (and therefore the vendored
   converter) has zero knowledge of:

   - ``ref_image_refiner`` (``transformer_boogu.py:898-911``) — a third
     refiner stack. Stock Lumina2 (and the vendored converter) only ever
     understood two refiner families (``noise_refiner``, ``context_refiner``
     — ``vendor/lora_conversion.py:82-90``).
   - ``double_stream_layers`` (``BooguImageDoubleStreamTransformerBlock``,
     ``transformer_boogu.py:399-774``) — an instruction/image dual-stream
     mixing stage that DELETES its ``Attention`` module's own
     ``to_q``/``to_k``/``to_v`` (``transformer_boogu.py:538-547``) and
     re-homes them on the attention PROCESSOR as plain ``nn.Linear``s
     (``img_to_q``/``img_to_k``/``img_to_v``/``instruct_to_q``/
     ``instruct_to_k``/``instruct_to_v`` —
     ``attention_processor.py:72-78,551-557`` — plus ``img_out``/
     ``instruct_out`` output projections, same file). Stock Lumina2 has no
     dual-stream stage and no processor-owned linears whatsoever; the
     vendored converter's three ``process_block`` calls (``noise_refiner``,
     ``context_refiner``, ``layers``) never touch these keys, and
     upstream's own leftover-key guard (``vendor/lora_conversion.py:97-98``,
     ``raise ValueError(f"...has {state_dict.keys()=}")``) would trip on
     them if an ecosystem checkpoint containing them were ever fed to it.

## Finding: which of the 418 curated targets round-trip through the
## (Boogu-adapted) ecosystem converter

Only modules belonging to ``noise_refiner``, ``context_refiner``, and
``single_stream_layers`` have a structural analogue in stock Lumina2:

    14 (noise_refiner) + 14 (context_refiner) + 224 (single_stream_layers)
    = 252 modules = 504 lora_A/B keys  -- PORTABLE

The remaining 166 modules — ``ref_image_refiner`` (14) and the entire
``double_stream_layers`` processor-owned + ``img_self_attn``/feed-forward
surface (152) — have NO representation in stock Lumina2 / the vendored
converter at all; they are Boogu-native architecture extensions:

    14 (ref_image_refiner) + 152 (double_stream_layers) = 166 modules
    = 332 lora_A/B keys  -- NOT PORTABLE (boogu_image-native only)

252 + 166 = 418 modules; 504 + 332 = 836 keys (the pinned canonical count).
A LoRA trained against the non-portable targets can ONLY be loaded via this
family's own loader/model directly against a house-format checkpoint — NOT
via ``BooguImageLoraLoaderMixin.lora_state_dict()``'s ecosystem-conversion
path — until Boogu's own upstream ships a converter update for these
extensions (out of scope here: we do not vendor a fix for someone else's
unshipped converter).

## Structural caveat on the export (diffusers -> ecosystem) direction

The ecosystem's fused-qkv format encodes ONE shared low-rank factor
``lora_A`` for the whole ``attention.qkv`` Linear
(``vendor/lora_conversion.py:50-56``: the SAME ``lora_down`` tensor is
reused for ``to_q``/``to_k``/``to_v``, only ``lora_up`` is split). That is
only mathematically exact if a checkpoint's independently-PEFT-trained
``to_q``/``to_k``/``to_v`` adapters happen to share an identical
``lora_A``. Boogu's curated ``lora_targetable_modules`` list targets
``to_q``/``to_k``/``to_v`` as three SEPARATE PEFT target modules
(``driver.py::get_lora_targets``), so real trained checkpoints generally do
NOT share a common ``lora_A`` across the three.
:func:`convert_diffusers_to_ecosystem` verifies this precondition
explicitly and raises a clear ``ValueError`` (naming the offending block)
rather than silently fusing mismatched adapters into a lossy/wrong
checkpoint.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch

# Boogu block families with a structural analogue in stock Lumina2 — see
# module docstring, evidence points 1-2. This is the intersection the
# Boogu-adapted converters below actually handle.
PORTABLE_BLOCK_PREFIXES: tuple[str, ...] = (
    "noise_refiner",
    "context_refiner",
    "single_stream_layers",
)

# Boogu-native architecture extensions with NO stock-Lumina2 / vendored-
# converter analogue (see module docstring, evidence point 3). Not consumed
# by the converters below; kept here so the "which keys don't port" finding
# lives next to the code that enforces it.
NON_PORTABLE_BLOCK_PREFIXES: tuple[str, ...] = (
    "ref_image_refiner",
    "double_stream_layers",
)

# Blocks that carry an AdaLN modulation LoRA target in the *fused* ecosystem
# convention (`adaLN_modulation.1`) -- mirrors vendor/lora_conversion.py's
# per-family `convert_norm` flags. Boogu's curated lora_targetable_modules
# never actually ships norm*.linear targets (base.yaml: "excluding AdaLN
# modulation"), so this only matters if a foreign ecosystem checkpoint
# happens to carry one; guarded with `.get`/`in` checks below, never a bare
# `.pop()` like the stock function.
_CONVERT_NORM = {
    "noise_refiner": True,
    "context_refiner": False,
    "single_stream_layers": True,
}


def qkv_split_from_config(config) -> tuple[int, int, int]:
    """Derive ``(to_q_width, to_k_width, to_v_width)`` from a transformer
    config (real checkpoint or the tiny test model alike) instead of
    hardcoding stock Lumina2's ``(2304, 768, 768)``.

    ``to_q`` width = ``num_attention_heads * head_dim`` = ``hidden_size``;
    ``to_k``/``to_v`` width = ``num_kv_heads * head_dim`` (GQA) — matches
    the ``Attention(dim_head=hidden_size // num_attention_heads,
    heads=num_attention_heads, kv_heads=num_kv_heads, ...)`` construction in
    ``transformer_boogu.py`` (e.g. ``BooguImageTransformerBlock.__init__``).
    """
    head_dim = config.hidden_size // config.num_attention_heads
    to_q = config.num_attention_heads * head_dim
    to_kv = config.num_kv_heads * head_dim
    return (to_q, to_kv, to_kv)


def _get_num_layers(keys: Iterable[str], pattern: str) -> int:
    layers: set[int] = set()
    for key in keys:
        match = re.search(pattern, key)
        if match:
            layers.add(int(match.group(1)))
    return len(layers)


def _strip_diffusion_model_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        (k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k): v
        for k, v in state_dict.items()
    }


def convert_ecosystem_to_diffusers(
    state_dict: dict[str, torch.Tensor],
    qkv_split: tuple[int, int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Boogu-adapted counterpart of the vendored
    ``_convert_non_diffusers_lumina2_lora_to_diffusers`` — the IMPORT
    direction (ecosystem fused-qkv -> diffusers-native).

    Args:
        state_dict: Raw state dict, MAY be "diffusion_model."-prefixed
            (stripped internally, mirroring
            ``lora_pipeline.py:165-167``).
        qkv_split: ``(to_q_width, to_k_width, to_v_width)`` — Boogu's real
            GQA widths (see :func:`qkv_split_from_config`), NOT stock
            Lumina2's hardcoded ``(2304, 768, 768)``.

    Returns:
        ``(converted, unconverted)``. ``converted`` uses diffusers-native
        ``transformer.<block>.<index>.attn.to_*``/``feed_forward.*``/
        ``norm1.linear`` keys (matching upstream's own output convention,
        ``vendor/lora_conversion.py:100-102``). ``unconverted`` holds every
        key that does not belong to a portable block family (e.g.
        ``ref_image_refiner``/``double_stream_layers`` keys), keyed by their
        original (post-prefix-strip) name — unlike upstream, this does NOT
        raise on leftovers (see module docstring, evidence point 3).
    """
    working = _strip_diffusion_model_prefix(state_dict)
    converted: dict[str, torch.Tensor] = {}

    def process_block(prefix: str, index: int) -> None:
        lora_down = working.pop(f"{prefix}.{index}.attention.qkv.lora_A.weight")
        lora_up = working.pop(f"{prefix}.{index}.attention.qkv.lora_B.weight")
        for attn_key in ("to_q", "to_k", "to_v"):
            converted[f"{prefix}.{index}.attn.{attn_key}.lora_A.weight"] = lora_down
        for attn_key, weight in zip(
            ("to_q", "to_k", "to_v"), torch.split(lora_up, list(qkv_split), dim=0),
        ):
            converted[f"{prefix}.{index}.attn.{attn_key}.lora_B.weight"] = weight

        converted[f"{prefix}.{index}.attn.to_out.0.lora_A.weight"] = working.pop(
            f"{prefix}.{index}.attention.out.lora_A.weight",
        )
        converted[f"{prefix}.{index}.attn.to_out.0.lora_B.weight"] = working.pop(
            f"{prefix}.{index}.attention.out.lora_B.weight",
        )

        for layer in range(1, 4):
            converted[
                f"{prefix}.{index}.feed_forward.linear_{layer}.lora_A.weight"
            ] = working.pop(f"{prefix}.{index}.feed_forward.w{layer}.lora_A.weight")
            converted[
                f"{prefix}.{index}.feed_forward.linear_{layer}.lora_B.weight"
            ] = working.pop(f"{prefix}.{index}.feed_forward.w{layer}.lora_B.weight")

        if _CONVERT_NORM[prefix]:
            norm_a = f"{prefix}.{index}.adaLN_modulation.1.lora_A.weight"
            norm_b = f"{prefix}.{index}.adaLN_modulation.1.lora_B.weight"
            if norm_a in working:
                converted[f"{prefix}.{index}.norm1.linear.lora_A.weight"] = working.pop(norm_a)
                converted[f"{prefix}.{index}.norm1.linear.lora_B.weight"] = working.pop(norm_b)

    for prefix in PORTABLE_BLOCK_PREFIXES:
        pattern = rf"^{re.escape(prefix)}\.(\d+)\."
        num_layers = _get_num_layers(working.keys(), pattern)
        for i in range(num_layers):
            process_block(prefix, i)

    for key in list(converted.keys()):
        converted[f"transformer.{key}"] = converted.pop(key)

    return converted, working


def convert_diffusers_to_ecosystem(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Reverse of :func:`convert_ecosystem_to_diffusers` — our EXPORT-side
    fusion into the ecosystem's shared-``lora_A`` fused-qkv convention (see
    module docstring's structural caveat).

    Args:
        state_dict: House-format (``diffusion_model.``-prefixed or bare)
            diffusers-native state dict, e.g. ``BooguImageSaver``'s output.

    Returns:
        ``(converted, unconverted)`` in the ecosystem's fused-qkv
        ``"diffusion_model."``-prefixed convention (matching what
        ``BooguImageLoraLoaderMixin.lora_state_dict``'s ``non_diffusers``
        branch expects as SOURCE format) for the portable block families;
        ``unconverted`` holds every other key verbatim (non-portable block
        keys, plus anything not recognized).

    Raises:
        ValueError: if a portable block's ``to_q``/``to_k``/``to_v``
            ``lora_A`` tensors are not identical — the fused-qkv format
            cannot represent independently-trained per-target adapters.
    """
    working = _strip_diffusion_model_prefix(state_dict)
    converted: dict[str, torch.Tensor] = {}

    for prefix in PORTABLE_BLOCK_PREFIXES:
        pattern = rf"^{re.escape(prefix)}\.(\d+)\."
        num_layers = _get_num_layers(working.keys(), pattern)
        for i in range(num_layers):
            q_a_key = f"{prefix}.{i}.attn.to_q.lora_A.weight"
            if q_a_key not in working:
                continue

            a_tensors = {
                attn_key: working.pop(f"{prefix}.{i}.attn.{attn_key}.lora_A.weight")
                for attn_key in ("to_q", "to_k", "to_v")
            }
            b_tensors = {
                attn_key: working.pop(f"{prefix}.{i}.attn.{attn_key}.lora_B.weight")
                for attn_key in ("to_q", "to_k", "to_v")
            }

            shared_a = a_tensors["to_q"]
            for attn_key, tensor in a_tensors.items():
                if not torch.equal(tensor, shared_a):
                    raise ValueError(
                        f"{prefix}.{i}: to_q/to_k/to_v lora_A tensors differ -- "
                        "cannot losslessly fuse into the ecosystem's shared-A "
                        "fused-qkv format (independently-trained adapters). "
                        "See lora_ecosystem.py module docstring "
                        "('Structural caveat on the export direction')."
                    )

            converted[f"{prefix}.{i}.attention.qkv.lora_A.weight"] = shared_a
            converted[f"{prefix}.{i}.attention.qkv.lora_B.weight"] = torch.cat(
                [b_tensors["to_q"], b_tensors["to_k"], b_tensors["to_v"]], dim=0,
            )

            converted[f"{prefix}.{i}.attention.out.lora_A.weight"] = working.pop(
                f"{prefix}.{i}.attn.to_out.0.lora_A.weight",
            )
            converted[f"{prefix}.{i}.attention.out.lora_B.weight"] = working.pop(
                f"{prefix}.{i}.attn.to_out.0.lora_B.weight",
            )

            for layer in range(1, 4):
                converted[
                    f"{prefix}.{i}.feed_forward.w{layer}.lora_A.weight"
                ] = working.pop(f"{prefix}.{i}.feed_forward.linear_{layer}.lora_A.weight")
                converted[
                    f"{prefix}.{i}.feed_forward.w{layer}.lora_B.weight"
                ] = working.pop(f"{prefix}.{i}.feed_forward.linear_{layer}.lora_B.weight")

    for key in list(converted.keys()):
        converted[f"diffusion_model.{key}"] = converted.pop(key)

    return converted, working
