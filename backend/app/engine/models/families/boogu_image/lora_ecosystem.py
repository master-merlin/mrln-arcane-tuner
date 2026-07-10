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
(``saver.py``) uses — is unconditionally routed through the Lumina2
conversion helper before being handed to
``load_lora_into_transformer -> transformer.load_lora_adapter(state_dict,
prefix="transformer")`` (``lora_pipeline.py:361-427``, esp. :419-427).

CRITICALLY (Task 7 review Finding 2): upstream imports that helper from
STOCK diffusers, not a Boogu-corrected fork —
``lora_pipeline.py:26-28``::

    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_lumina2_lora_to_diffusers,
    )

So TODAY's real upstream mixin runs the stock converter with the stock
Lumina2 hardcodes against any ``diffusion_model.*`` checkpoint — meaning
even the "portable" module set below cannot load through it as shipped
(wrong qkv split 2304/768/768 vs Boogu's 3360/840/840, and the stock
``"layers"`` block-name assumption mis-matching Boogu's
``single_stream_layers`` — see "Why the vendored converter cannot be used
verbatim"). The PORTABLE/NON-PORTABLE split documented here therefore
describes what the fused ecosystem FORMAT (processed by a Boogu-CORRECTED
converter, i.e. this module's :func:`convert_ecosystem_to_diffusers`) can
and cannot represent — NOT what today's stock upstream loader accepts.

What a ComfyUI-Boogu user can actually do TODAY: the ``diffusion_model.``
prefix is the ONLY trigger for upstream's conversion branch
(``lora_pipeline.py:165``). A state dict carrying diffusers-native
``transformer.<module_path>.lora_A/B.weight`` keys bypasses the broken
stock converter entirely and goes straight to
``transformer.load_lora_adapter(..., prefix="transformer")`` — the
format-supported route for our exports (portable AND non-portable modules
alike, since the module paths are the transformer's own ``named_modules``
paths) until Boogu's upstream ships a corrected converter for the fused
``diffusion_model.*`` format.

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

3. **Architectural extensions with NO stock-Lumina2 analogue at all.**
   Boogu adds two whole stages stock Lumina2 (and therefore the vendored
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

## Finding: which of the 418 curated targets the fused ecosystem FORMAT
## can represent (via this module's Boogu-corrected converters)

Only modules belonging to ``noise_refiner``, ``context_refiner``, and
``single_stream_layers`` have a structural analogue in stock Lumina2's
fused format:

    14 (noise_refiner) + 14 (context_refiner) + 224 (single_stream_layers)
    = 252 modules = 504 lora_A/B keys  -- REPRESENTABLE in the fused format

The remaining 166 modules — ``ref_image_refiner`` (14) and the entire
``double_stream_layers`` processor-owned + ``img_self_attn``/feed-forward
surface (152) — have NO representation in the fused format at all; they
are Boogu-native architecture extensions:

    14 (ref_image_refiner) + 152 (double_stream_layers) = 166 modules
    = 332 lora_A/B keys  -- NOT representable (boogu_image-native only)

252 + 166 = 418 modules; 504 + 332 = 836 keys (the pinned canonical count).
Remember (Finding 2 above): even the representable 252 do NOT load through
today's stock upstream mixin — the split describes the format's ceiling
once a Boogu-corrected converter exists upstream; until then the
``transformer.*``-prefixed diffusers-native route above is the loadable
path for everything.

## Export direction: lossless rank-stacking (Task 7 review Finding 1)

The ecosystem's fused-qkv format natively encodes ONE shared low-rank
factor ``lora_A`` for the whole ``attention.qkv`` Linear
(``vendor/lora_conversion.py:50-56``: the SAME ``lora_down`` tensor is
reused for ``to_q``/``to_k``/``to_v``, only ``lora_up`` is split). Boogu's
curated ``lora_targetable_modules`` list targets ``to_q``/``to_k``/``to_v``
as three SEPARATE PEFT target modules (``driver.py::get_lora_targets``), so
real trained checkpoints generally do NOT share a common ``lora_A`` across
the three. :func:`convert_diffusers_to_ecosystem` handles this LOSSLESSLY
via rank-stacking::

    A_fused = cat([A_q, A_k, A_v], dim=0)                  # [r_q+r_k+r_v, in]
    B'_q = [B_q | 0 | 0]  (B_q in its own rank columns)     # [out_q, r_q+r_k+r_v]
    B'_k = [0 | B_k | 0], B'_v = [0 | 0 | B_v]
    fused_B = cat([B'_q, B'_k, B'_v], dim=0)

so that ``B'_x @ A_fused == B_x @ A_x`` bit-exactly (the zero blocks
contribute exact zeros). The import direction (shared ``A_fused`` assigned
to all three, ``fused_B`` split by output rows) round-trips this exactly.
A fast path keeps the original rank when the three ``lora_A`` tensors are
already identical (no rank inflation).

Alpha: the fused ecosystem format is alpha-free — a consumer loading it
(e.g. upstream's ``load_lora_adapter`` with ``network_alphas=None``)
defaults ``lora_alpha = rank`` => scale 1. Our training scale is
``network_alpha / network_rank`` (``pipeline_optimization.py::_apply_peft``,
``alpha = float(self.config.get("network_alpha", rank))`` — defaults to
rank but is user-configurable). When the caller passes the trained
``alpha`` (from the saver's ``ss_network_alpha`` safetensors metadata) and
it differs from a module's rank, :func:`convert_diffusers_to_ecosystem`
folds the ``alpha/rank`` scale into ``lora_B`` at export so the consumer's
scale-1 read reproduces the trained effective delta.

## Graceful partial-checkpoint contract (Task 7 review Finding 3)

BOTH converters iterate the layer indices actually present (not
``range(count)`` — non-contiguous indices are fine) and only convert a
sub-target when its complete key group is present (qkv triple: all 6
tensors; out / feed-forward / adaLN: their full A+B pair). Anything
incomplete or unrecognized is returned in ``unconverted`` verbatim — no
``KeyError``, and no error path aborts unaffected keys. The one deliberate
exception: a fused-qkv ``lora_B`` whose row count does not match the given
``qkv_split`` raises (``torch.split``) — that is a geometry mismatch
(corrupt input or stock-Lumina2 dims fed to a Boogu model), not a partial
checkpoint, and silently passing it through would hide exactly the MUST-FIX
bug class.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch

# Boogu block families with a structural analogue in stock Lumina2's fused
# format — see module docstring. This is the intersection the Boogu-adapted
# converters below actually handle. NOTE (review Finding 2): "portable"
# means representable in the fused format via THIS module's corrected
# converters — today's stock upstream converter loads neither set (see
# module docstring).
PORTABLE_BLOCK_PREFIXES: tuple[str, ...] = (
    "noise_refiner",
    "context_refiner",
    "single_stream_layers",
)

# Boogu-native architecture extensions with NO stock-Lumina2 / fused-format
# analogue (see module docstring, evidence point 3). Not consumed by the
# converters below; kept here so the "which keys can't be represented"
# finding lives next to the code that enforces it.
NON_PORTABLE_BLOCK_PREFIXES: tuple[str, ...] = (
    "ref_image_refiner",
    "double_stream_layers",
)

# Blocks that carry an AdaLN modulation LoRA target in the *fused* ecosystem
# convention (`adaLN_modulation.1`) -- mirrors vendor/lora_conversion.py's
# per-family `convert_norm` flags. Boogu's curated lora_targetable_modules
# never actually ships norm*.linear targets (base.yaml: "excluding AdaLN
# modulation"), so this only matters for foreign ecosystem checkpoints.
_CONVERT_NORM = {
    "noise_refiner": True,
    "context_refiner": False,
    "single_stream_layers": True,
}

_QKV_ORDER = ("to_q", "to_k", "to_v")


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


def _indices_for_prefix(keys: Iterable[str], prefix: str) -> list[int]:
    """The layer indices ACTUALLY present under ``prefix`` — supports
    non-contiguous/partial checkpoints (review Finding 3), unlike the
    vendored converter's distinct-count + ``range(n)`` walk."""
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    found: set[int] = set()
    for key in keys:
        match = pattern.match(key)
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def _strip_diffusion_model_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        (k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k): v
        for k, v in state_dict.items()
    }


def _pop_pair(
    working: dict[str, torch.Tensor], key_a: str, key_b: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pop an (lora_A, lora_B) pair only if BOTH are present (graceful
    partial-checkpoint contract) — else leave both untouched."""
    if key_a in working and key_b in working:
        return working.pop(key_a), working.pop(key_b)
    return None


def convert_ecosystem_to_diffusers(
    state_dict: dict[str, torch.Tensor],
    qkv_split: tuple[int, int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Boogu-adapted counterpart of the vendored
    ``_convert_non_diffusers_lumina2_lora_to_diffusers`` — the IMPORT
    direction (ecosystem fused-qkv -> diffusers-native).

    Args:
        state_dict: Raw state dict, MAY be "diffusion_model."-prefixed
            (stripped internally, mirroring ``lora_pipeline.py:165-167``).
        qkv_split: ``(to_q_width, to_k_width, to_v_width)`` — Boogu's real
            GQA widths (see :func:`qkv_split_from_config`), NOT stock
            Lumina2's hardcoded ``(2304, 768, 768)``.

    Returns:
        ``(converted, unconverted)``. ``converted`` uses diffusers-native
        ``transformer.<block>.<index>.attn.to_*``/``feed_forward.*``/
        ``norm1.linear`` keys (matching upstream's own output convention,
        ``vendor/lora_conversion.py:100-102``). ``unconverted`` holds every
        key that does not belong to a portable block family OR whose
        A/B key group is incomplete (partial checkpoints — see module
        docstring "Graceful partial-checkpoint contract"), keyed by its
        original (post-prefix-strip) name. Unlike upstream, this never
        raises on leftovers.

    Raises:
        RuntimeError: (via ``torch.split``) if a fused ``lora_B``'s row
            count does not match ``qkv_split`` — a geometry mismatch, not
            a partial checkpoint (see module docstring).
    """
    working = _strip_diffusion_model_prefix(state_dict)
    converted: dict[str, torch.Tensor] = {}

    for prefix in PORTABLE_BLOCK_PREFIXES:
        for i in _indices_for_prefix(working.keys(), prefix):
            base = f"{prefix}.{i}"

            qkv = _pop_pair(
                working,
                f"{base}.attention.qkv.lora_A.weight",
                f"{base}.attention.qkv.lora_B.weight",
            )
            if qkv is not None:
                lora_down, lora_up = qkv
                for attn_key in _QKV_ORDER:
                    converted[f"{base}.attn.{attn_key}.lora_A.weight"] = lora_down
                for attn_key, weight in zip(
                    _QKV_ORDER, torch.split(lora_up, list(qkv_split), dim=0),
                ):
                    converted[f"{base}.attn.{attn_key}.lora_B.weight"] = weight

            out = _pop_pair(
                working,
                f"{base}.attention.out.lora_A.weight",
                f"{base}.attention.out.lora_B.weight",
            )
            if out is not None:
                converted[f"{base}.attn.to_out.0.lora_A.weight"] = out[0]
                converted[f"{base}.attn.to_out.0.lora_B.weight"] = out[1]

            for layer in range(1, 4):
                ff = _pop_pair(
                    working,
                    f"{base}.feed_forward.w{layer}.lora_A.weight",
                    f"{base}.feed_forward.w{layer}.lora_B.weight",
                )
                if ff is not None:
                    converted[f"{base}.feed_forward.linear_{layer}.lora_A.weight"] = ff[0]
                    converted[f"{base}.feed_forward.linear_{layer}.lora_B.weight"] = ff[1]

            if _CONVERT_NORM[prefix]:
                norm = _pop_pair(
                    working,
                    f"{base}.adaLN_modulation.1.lora_A.weight",
                    f"{base}.adaLN_modulation.1.lora_B.weight",
                )
                if norm is not None:
                    converted[f"{base}.norm1.linear.lora_A.weight"] = norm[0]
                    converted[f"{base}.norm1.linear.lora_B.weight"] = norm[1]

    for key in list(converted.keys()):
        converted[f"transformer.{key}"] = converted.pop(key)

    return converted, working


def convert_diffusers_to_ecosystem(
    state_dict: dict[str, torch.Tensor],
    alpha: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Reverse of :func:`convert_ecosystem_to_diffusers` — our EXPORT-side
    fusion into the ecosystem's fused-qkv convention, LOSSLESS for
    independently-trained per-target adapters via rank-stacking (see module
    docstring "Export direction" — Task 7 review Finding 1).

    Args:
        state_dict: House-format (``diffusion_model.``-prefixed or bare)
            diffusers-native state dict, e.g. ``BooguImageSaver``'s output.
        alpha: The checkpoint's trained ``network_alpha`` (available in the
            saver's ``ss_network_alpha`` safetensors metadata). The fused
            ecosystem format is alpha-free (consumers read it at scale 1,
            i.e. alpha == rank); when ``alpha`` is given and differs from a
            module's rank, the ``alpha/rank`` scale is folded into that
            module's ``lora_B`` so the consumer reproduces the trained
            effective delta. ``None`` (default) assumes ``alpha == rank``
            (the house default — ``pipeline_optimization.py::_apply_peft``).

    Returns:
        ``(converted, unconverted)`` in the ecosystem's fused-qkv
        ``"diffusion_model."``-prefixed convention (the SOURCE format of
        ``BooguImageLoraLoaderMixin.lora_state_dict``'s ``non_diffusers``
        branch — loadable through upstream only once upstream corrects its
        stock converter; see module docstring Finding 2 note) for the
        portable block families. ``unconverted`` holds every other key
        verbatim: non-portable block keys, unrecognized keys, and any
        module whose key group is incomplete (e.g. a qkv triple missing
        one tensor) — incomplete groups never abort conversion of the
        remaining modules.
    """
    working = _strip_diffusion_model_prefix(state_dict)
    converted: dict[str, torch.Tensor] = {}

    def _scaled(b: torch.Tensor, rank: int) -> torch.Tensor:
        if alpha is None or float(alpha) == float(rank):
            return b
        return b * (float(alpha) / float(rank))

    def _fuse_qkv(base: str) -> None:
        keys_a = {x: f"{base}.attn.{x}.lora_A.weight" for x in _QKV_ORDER}
        keys_b = {x: f"{base}.attn.{x}.lora_B.weight" for x in _QKV_ORDER}
        needed = list(keys_a.values()) + list(keys_b.values())
        if not all(k in working for k in needed):
            return  # incomplete triple — leave everything in `unconverted`.

        a = {x: working.pop(keys_a[x]) for x in _QKV_ORDER}
        b = {x: _scaled(working.pop(keys_b[x]), a[x].shape[0]) for x in _QKV_ORDER}

        if torch.equal(a["to_q"], a["to_k"]) and torch.equal(a["to_q"], a["to_v"]):
            # Fast path: already the ecosystem's native shared-A shape —
            # no rank inflation.
            fused_a = a["to_q"]
            fused_b = torch.cat([b[x] for x in _QKV_ORDER], dim=0)
        else:
            # Lossless rank-stacking: A_fused = cat(A_q, A_k, A_v);
            # each B block-placed into its own rank columns so
            # B'_x @ A_fused == B_x @ A_x bit-exactly (the zero blocks
            # contribute exact zeros).
            ranks = [a[x].shape[0] for x in _QKV_ORDER]
            total_rank = sum(ranks)
            fused_a = torch.cat([a[x] for x in _QKV_ORDER], dim=0)

            padded_blocks: list[torch.Tensor] = []
            col_offset = 0
            for x, r in zip(_QKV_ORDER, ranks):
                block = b[x].new_zeros(b[x].shape[0], total_rank)
                block[:, col_offset:col_offset + r] = b[x]
                padded_blocks.append(block)
                col_offset += r
            fused_b = torch.cat(padded_blocks, dim=0)

        converted[f"{base}.attention.qkv.lora_A.weight"] = fused_a
        converted[f"{base}.attention.qkv.lora_B.weight"] = fused_b

    def _move_pair(src_base: str, dst_base: str) -> None:
        pair = _pop_pair(
            working, f"{src_base}.lora_A.weight", f"{src_base}.lora_B.weight",
        )
        if pair is not None:
            a_tensor, b_tensor = pair
            converted[f"{dst_base}.lora_A.weight"] = a_tensor
            converted[f"{dst_base}.lora_B.weight"] = _scaled(
                b_tensor, a_tensor.shape[0],
            )

    for prefix in PORTABLE_BLOCK_PREFIXES:
        for i in _indices_for_prefix(working.keys(), prefix):
            base = f"{prefix}.{i}"
            _fuse_qkv(base)
            _move_pair(f"{base}.attn.to_out.0", f"{base}.attention.out")
            for layer in range(1, 4):
                _move_pair(
                    f"{base}.feed_forward.linear_{layer}",
                    f"{base}.feed_forward.w{layer}",
                )
            if _CONVERT_NORM[prefix]:
                _move_pair(f"{base}.norm1.linear", f"{base}.adaLN_modulation.1")

    for key in list(converted.keys()):
        converted[f"diffusion_model.{key}"] = converted.pop(key)

    return converted, working
