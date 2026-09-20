"""minimax_h3 — the native ↔ diffusers key plan (plan row 1.6, finding F-1; row 2.8 extends it).

# Method re-derived from diffusers `scripts/convert_minimax_h3_to_diffusers.py`
# @ c419dac0 (Apache-2.0) and measured against the on-disk indexes; no line copied.

The checkpoint ships in TWO naming conventions and the rest of the ecosystem
speaks the native one (research §3.9): ComfyUI, ai-toolkit and diffusion-pipe
all address `blocks.N.attn.qkv_proj`, while the installed diffusers class (and
therefore every PEFT module name this app trains) addresses
`transformer_blocks.N.attn.to_q`. This module is the ONE place that relates
the two — a LoRA saver that invents its own rename drifts silently and the
artifact loads nowhere.

Rules (each is a per-tensor transform, not just a rename):

* ``copy``        — same tensor, new name (block prefixes, norms, AdaLN,
                    projections, heads, time embedder, `condition_proj` →
                    `context_embedder`);
* ``qkv_split``   — the fused `attn.qkv_proj.weight` becomes `to_q`, `to_k`,
                    `to_v` as contiguous thirds along dim 0;
* ``swiglu_swap`` — `mlp.fc1.weight` keeps its fused shape as
                    `ff.net.0.proj.weight` with its two halves swapped
                    (diffusers' `SwiGLU` gates the OTHER half);
* ``drop``        — `rope.inv_freq`, a pure function of the config the port
                    recomputes.

Every rule is anchored (``^``/``$``) so a key that matches nothing raises —
an unmapped key is exactly the failure F-1 exists to catch.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

DROPPED_NATIVE_KEYS: tuple[str, ...] = ("rope.inv_freq",)

# (native regex, diffusers replacement templates, transform). ``\g<n>`` in a
# template is the block index captured by the regex. Order matters only for
# readability: the anchors make the rules disjoint.
_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...], str], ...] = tuple(
    (re.compile(pattern), templates, transform)
    for pattern, templates, transform in (
        # ── block stacks: `blocks.N.` → `transformer_blocks.N.`,
        #    `token_refiner.blocks.N.` → `token_refiner.refiner_blocks.N.`
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.attn\.qkv_proj\.weight$",
            (
                r"\g<p>{stack}.\g<n>.attn.to_q.weight",
                r"\g<p>{stack}.\g<n>.attn.to_k.weight",
                r"\g<p>{stack}.\g<n>.attn.to_v.weight",
            ),
            "qkv_split",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.mlp\.fc1\.weight$",
            (r"\g<p>{stack}.\g<n>.ff.net.0.proj.weight",),
            "swiglu_swap",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.mlp\.fc2\.weight$",
            (r"\g<p>{stack}.\g<n>.ff.net.2.weight",),
            "copy",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.attn\.out_proj\.weight$",
            (r"\g<p>{stack}.\g<n>.attn.to_out.0.weight",),
            "copy",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.attn\.q_norm\.weight$",
            (r"\g<p>{stack}.\g<n>.attn.norm_q.weight",),
            "copy",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.attn\.k_norm\.weight$",
            (r"\g<p>{stack}.\g<n>.attn.norm_k.weight",),
            "copy",
        ),
        (
            r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\.(?P<rest>norm1\.weight|norm2\.weight)$",
            (r"\g<p>{stack}.\g<n>.\g<rest>",),
            "copy",
        ),
        # AdaLN tables exist on the main blocks only (refiners carry none).
        (
            r"^blocks\.(?P<n>\d+)\.(?P<rest>adaln_proj\.linear\.(?:weight|bias))$",
            (r"transformer_blocks.\g<n>.\g<rest>",),
            "copy",
        ),
        # ── stem / head / time embedder
        (r"^video_patch_proj\.(?P<t>weight|bias)$", (r"proj_in.\g<t>",), "copy"),
        (r"^audio_patch_proj\.(?P<t>weight|bias)$", (r"audio_proj_in.\g<t>",), "copy"),
        (r"^condition_proj\.(?P<t>weight|bias)$", (r"context_embedder.\g<t>",), "copy"),
        (r"^time_embedder\.proj_in\.(?P<t>weight|bias)$", (r"time_embedder.linear_1.\g<t>",), "copy"),
        (r"^time_embedder\.proj_out\.(?P<t>weight|bias)$", (r"time_embedder.linear_2.\g<t>",), "copy"),
        (r"^token_refiner\.final_norm\.weight$", (r"token_refiner.final_norm.weight",), "copy"),
        (r"^final_layer\.norm\.weight$", (r"norm_out.norm.weight",), "copy"),
        (r"^final_layer\.adaln_proj\.linear\.(?P<t>weight|bias)$", (r"norm_out.linear.\g<t>",), "copy"),
        (r"^final_layer\.video_out\.(?P<t>weight|bias)$", (r"proj_out.\g<t>",), "copy"),
        (r"^final_layer\.audio_out\.(?P<t>weight|bias)$", (r"audio_proj_out.\g<t>",), "copy"),
        (r"^rope\.inv_freq$", (), "drop"),
    )
)


@dataclass(frozen=True)
class KeyMapping:
    """One native tensor → its diffusers targets and the transform between them."""

    native: str
    targets: tuple[str, ...]
    transform: str  # "copy" | "qkv_split" | "swiglu_swap" | "drop"


def _stack_for(prefix: str) -> str:
    return "refiner_blocks" if prefix else "transformer_blocks"


def map_native_key(native: str) -> KeyMapping:
    """Map ONE native checkpoint key. Raises ``KeyError`` on a key no rule
    covers — never guess a name the converter would not produce."""
    for pattern, templates, transform in _RULES:
        match = pattern.match(native)
        if match is None:
            continue
        stack = _stack_for(match.groupdict().get("p") or "")
        targets = tuple(match.expand(t.replace("{stack}", stack)) for t in templates)
        return KeyMapping(native=native, targets=targets, transform=transform)
    raise KeyError(f"no F-1 rule maps native key {native!r}")


def native_to_diffusers(native_keys: Iterable[str]) -> dict[str, KeyMapping]:
    """Map every native key. A key no rule covers is returned with
    ``transform="unmapped"`` and no targets so a caller can report ALL of
    them at once (F-1 names every unmapped key, not the first)."""
    plan: dict[str, KeyMapping] = {}
    for key in native_keys:
        try:
            plan[key] = map_native_key(key)
        except KeyError:
            plan[key] = KeyMapping(native=key, targets=(), transform="unmapped")
    return plan


def unmapped_keys(plan: dict[str, KeyMapping]) -> list[str]:
    return sorted(k for k, m in plan.items() if m.transform == "unmapped")


# ── Row 2.8: PEFT adapters → the ORIGINAL-checkpoint LoRA artifact ───────────
#
# The artifact every H3 producer emits and ComfyUI consumes, and what diffusers
# 0.40 converts back (`loaders/lora_conversion_utils.py:3122`): native module
# names under a `diffusion_model.` prefix, `lora_A` / `lora_B`, PEFT's
# `alpha / r` scaling FOLDED into `lora_B`, no `.alpha` key — so a loader that
# applies scale 1 reproduces our delta exactly. Reserved as a public id
# (ECOSYSTEM §6, REQUEST-16); `ARTIFACT_LAYOUT_VERSION` is frozen once shipped.
#
# The three per-tensor transforms are the inverse of the F-1 rules above:
#   * `qkv_fuse`   — three independent rank-r adapters on `to_q`/`to_k`/`to_v`
#                    become ONE fused `attn.qkv_proj` adapter of rank 3r:
#                    `A_fused = [A_q; A_k; A_v]` (3r × hidden) and `B_fused`
#                    block-diagonal (3·inner × 3r) — exact, no approximation;
#                    the consumer chunks `lora_B` in thirds and hands every
#                    third the whole `lora_A`, so each third sees only its own
#                    rank columns (the zero blocks kill the others).
#   * `swiglu_swap` — `ff.net.0.proj` (diffusers rows `[value; gate]`) becomes
#                    `mlp.fc1` (checkpoint rows `[gate; value]`): a permutation
#                    of OUTPUT rows, so it touches `lora_B` alone.
#   * `rename`     — everything else is a name change.

ARTIFACT_LAYOUT_VERSION = 1
ARTIFACT_PREFIX = "diffusion_model."

_QKV_SLOT = {"to_q": 0, "to_k": 1, "to_v": 2}
_BLOCK_MODULE = re.compile(
    r"^(?:(?P<refiner>token_refiner\.)refiner_blocks|transformer_blocks)\.(?P<n>\d+)\.(?P<rest>.+)$"
)
_PLAIN_RENAMES = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.2": "mlp.fc2",
    "adaln_proj.linear": "adaln_proj.linear",
}


@dataclass(frozen=True)
class Adapter:
    """One PEFT LoRA pair on a diffusers module, with the scaling PEFT applies
    in its forward (`alpha / r`, or `alpha / sqrt(r)` under rsLoRA — read off
    the layer, never recomputed)."""

    lora_A: Any
    lora_B: Any
    scaling: float


def native_module_for(diffusers_module: str) -> tuple[str, str]:
    """A diffusers LoRA module path → ``(native module path, transform)`` with
    transform one of ``"q"`` / ``"k"`` / ``"v"`` (a slot of the fused
    ``qkv_proj``), ``"fc1"`` (the SwiGLU swap) or ``"rename"``. Raises
    ``KeyError`` on a module no rule covers — never invent a name."""
    match = _BLOCK_MODULE.match(diffusers_module)
    if match is None:
        raise KeyError(f"no row-2.8 rule maps diffusers module {diffusers_module!r}")
    prefix = "token_refiner." if match.group("refiner") else ""
    block = f"{prefix}blocks.{match.group('n')}"
    rest = match.group("rest")
    if rest.startswith("attn.to_") and rest.removeprefix("attn.") in _QKV_SLOT:
        return f"{block}.attn.qkv_proj", rest.removeprefix("attn.")[-1]
    if rest == "ff.net.0.proj":
        return f"{block}.mlp.fc1", "fc1"
    if rest in _PLAIN_RENAMES:
        return f"{block}.{_PLAIN_RENAMES[rest]}", "rename"
    raise KeyError(f"no row-2.8 rule maps diffusers module {diffusers_module!r}")


def adapters_from_peft_model(model: Any, adapter_name: str = "default") -> dict[str, Adapter]:
    """Every LoRA pair on a PEFT-wrapped transformer, keyed by the diffusers
    module path (``base_model.model.`` stripped), with each layer's own
    ``scaling``. Incomplete pairs and layers without a scaling are refused."""
    import torch
    from peft import get_peft_model_state_dict

    scalings: dict[str, float] = {}
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling and hasattr(module, "lora_A"):
            scalings[name.removeprefix("base_model.model.")] = float(scaling[adapter_name])

    pairs: dict[str, dict[str, Any]] = {}
    for key, value in get_peft_model_state_dict(model, adapter_name=adapter_name).items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = key.removeprefix("base_model.model.")
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if clean.endswith(suffix):
                pairs.setdefault(clean[: -len(suffix)], {})[slot] = value.detach()
                break

    adapters: dict[str, Adapter] = {}
    for module, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"minimax_h3 lora_keys: incomplete LoRA pair on {module!r}")
        if module not in scalings:
            raise ValueError(f"minimax_h3 lora_keys: no PEFT scaling found for {module!r}")
        adapters[module] = Adapter(lora_A=pair["A"], lora_B=pair["B"], scaling=scalings[module])
    return adapters


def build_artifact(adapters: dict[str, Adapter]) -> dict[str, Any]:
    """The artifact tensors (float32; the saver casts) in the ORIGINAL layout.
    A projection missing from a fused triple (adaptive rebuild narrowed the
    targets) contributes a zero block, so the fused delta stays exact."""
    import torch

    out: dict[str, Any] = {}
    fused: dict[str, dict[int, tuple[Any, Any]]] = {}
    for module, adapter in adapters.items():
        native, transform = native_module_for(module)
        a = adapter.lora_A.detach().float()
        b = adapter.lora_B.detach().float() * float(adapter.scaling)  # the fold
        if transform in ("q", "k", "v"):
            fused.setdefault(native, {})[_QKV_SLOT[f"to_{transform}"]] = (a, b)
            continue
        if transform == "fc1":
            half = b.shape[0] // 2
            b = torch.cat([b[half:], b[:half]], dim=0)  # [value; gate] → [gate; value]
        out[f"{ARTIFACT_PREFIX}{native}.lora_A.weight"] = a.contiguous()
        out[f"{ARTIFACT_PREFIX}{native}.lora_B.weight"] = b.contiguous()

    for native, slots in fused.items():
        any_a, any_b = next(iter(slots.values()))
        inner, hidden = any_b.shape[0], any_a.shape[1]
        total_rank = sum(slots[i][0].shape[0] for i in range(3) if i in slots)
        a_fused = torch.cat([slots[i][0] for i in range(3) if i in slots], dim=0)
        b_fused = torch.zeros((3 * inner, total_rank), dtype=any_b.dtype)
        col = 0
        for i in range(3):
            if i not in slots:
                continue
            a_i, b_i = slots[i]
            r_i = a_i.shape[0]
            b_fused[i * inner : (i + 1) * inner, col : col + r_i] = b_i
            col += r_i
        assert a_fused.shape == (total_rank, hidden)
        out[f"{ARTIFACT_PREFIX}{native}.lora_A.weight"] = a_fused.contiguous()
        out[f"{ARTIFACT_PREFIX}{native}.lora_B.weight"] = b_fused.contiguous()
    return out


# ── Row 2.9: loading an artifact BACK (the interop proof, both directions) ───
#
# `load_via_diffusers` is the independent oracle: diffusers 0.40's own
# converter (`loaders/lora_conversion_utils.py:3122`, a PRIVATE symbol) then
# `PeftAdapterMixin.load_lora_adapter`. When that symbol is gone (a diffusers
# bump), `artifact_to_diffusers` — OUR inverse of `build_artifact` — runs as
# the production fallback and says so (`fallback_used`). A proof must run with
# the fallback DISABLED: an exporter error and its own inverse cancel.
#
# `load_via_lora_keys` is the inbound path the sampler uses for a LoRA from
# outside this app: kohya / musubi (`lora_unet_` flattened names,
# `lora_down` / `lora_up`, per-module `.alpha`), bare ComfyUI (native names,
# `lora_A` / `lora_B`, alpha-less) and DiffSynth (`.default.` infix), with or
# without the `diffusion_model.` prefix — every layout the diffusers converter
# accepts (`:3141-3182`), re-derived here so the two never disagree.

_KOHYA_FLATTENED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^blocks_(\d+)_attn_(qkv|out)_proj$"), r"blocks.\1.attn.\2_proj"),
    (re.compile(r"^blocks_(\d+)_mlp_fc([12])$"), r"blocks.\1.mlp.fc\2"),
    (re.compile(r"^blocks_(\d+)_adaln_proj_linear$"), r"blocks.\1.adaln_proj.linear"),
    (re.compile(r"^token_refiner_blocks_(\d+)_attn_(qkv|out)_proj$"), r"token_refiner.blocks.\1.attn.\2_proj"),
    (re.compile(r"^token_refiner_blocks_(\d+)_mlp_fc([12])$"), r"token_refiner.blocks.\1.mlp.fc\2"),
)
_NATIVE_BLOCK_MODULE = re.compile(
    r"^(?P<p>|token_refiner\.)blocks\.(?P<n>\d+)\."
    r"(?P<rest>attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2|adaln_proj\.linear)$"
)
_NATIVE_RENAMES = {
    "attn.out_proj": "attn.to_out.0",
    "mlp.fc2": "ff.net.2",
    "adaln_proj.linear": "adaln_proj.linear",
}
_DIFFUSERS_CONVERTER = "_convert_non_diffusers_minimax_h3_lora_to_diffusers"


@dataclass(frozen=True)
class LoraLoadResult:
    """What a load did: which path ran (`via`), whether the production
    fallback was the one that ran, and how many NATIVE modules the file
    carried (a fused `qkv_proj` counts once)."""

    fallback_used: bool
    modules: int
    adapter_name: str
    via: str  # "diffusers" | "lora_keys"


def _native_pairs(state_dict: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Normalise ANY inbound layout to ``{native module: (down, up)}`` with the
    per-module ``.alpha`` already folded into ``up`` (``alpha / r``)."""
    sd = {k.removeprefix(ARTIFACT_PREFIX): v for k, v in state_dict.items()}
    sd = {
        k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): v
        for k, v in sd.items()
    }
    dotted_sd: dict[str, Any] = {}
    for key, value in sd.items():
        if key.startswith("lora_unet_"):
            module, _, suffix = key.removeprefix("lora_unet_").partition(".")
            for pattern, replacement in _KOHYA_FLATTENED:
                if pattern.match(module):
                    key = f"{pattern.sub(replacement, module)}.{suffix}"
                    break
            else:
                raise KeyError(f"no row-2.9 rule unflattens kohya module {key!r}")
        dotted_sd[key] = value

    downs: dict[str, Any] = {}
    ups: dict[str, Any] = {}
    alphas: dict[str, Any] = {}
    buckets = (
        (".lora_down.weight", downs),
        (".lora_A.weight", downs),
        (".lora_up.weight", ups),
        (".lora_B.weight", ups),
        (".alpha", alphas),
    )
    for key, value in dotted_sd.items():
        for suffix, bucket in buckets:
            if key.endswith(suffix):
                bucket[key[: -len(suffix)]] = value
                break
        else:
            raise KeyError(f"unrecognised LoRA key {key!r}")
    stray = sorted(set(ups) - set(downs))
    if stray:
        raise ValueError(f"lora_up/lora_B without a lora_down/lora_A: {stray}")

    pairs: dict[str, tuple[Any, Any]] = {}
    for module, down in downs.items():
        if module not in ups:
            raise ValueError(f"lora_down/lora_A without a lora_up/lora_B: {module!r}")
        down = down.detach().float()
        up = ups[module].detach().float()
        alpha = alphas.get(module)
        if alpha is not None:
            up = up * (float(alpha) / down.shape[0])
        pairs[module] = (down, up)
    return pairs


def artifact_to_diffusers(state_dict: dict[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`build_artifact`: any inbound layout →
    ``{diffusers module}.lora_A|B.weight`` (no component prefix), scale 1."""
    import torch

    out: dict[str, Any] = {}
    for module, (down, up) in _native_pairs(state_dict).items():
        match = _NATIVE_BLOCK_MODULE.match(module)
        if match is None:
            raise KeyError(f"no row-2.9 rule maps native module {module!r}")
        stack = "token_refiner.refiner_blocks" if match.group("p") else "transformer_blocks"
        block = f"{stack}.{match.group('n')}"
        rest = match.group("rest")
        if rest == "attn.qkv_proj":
            if up.shape[0] % 3:
                raise ValueError(f"{module}: {up.shape[0]} output rows is not a fused qkv projection")
            for proj, part in zip(("to_q", "to_k", "to_v"), up.chunk(3, dim=0)):
                out[f"{block}.attn.{proj}.lora_A.weight"] = down.clone()
                out[f"{block}.attn.{proj}.lora_B.weight"] = part.contiguous()
        elif rest == "mlp.fc1":
            if up.shape[0] % 2:
                raise ValueError(f"{module}: {up.shape[0]} output rows is not a fused SwiGLU projection")
            half = up.shape[0] // 2
            out[f"{block}.ff.net.0.proj.lora_A.weight"] = down.contiguous()
            out[f"{block}.ff.net.0.proj.lora_B.weight"] = torch.cat([up[half:], up[:half]], dim=0).contiguous()
        else:
            target = _NATIVE_RENAMES[rest]
            out[f"{block}.{target}.lora_A.weight"] = down.contiguous()
            out[f"{block}.{target}.lora_B.weight"] = up.contiguous()
    return out


def _read_artifact(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        return dict(artifact)
    from safetensors.torch import load_file

    return load_file(str(artifact))


def _inject(model: Any, diffusers_sd: dict[str, Any], adapter_name: str) -> None:
    """Hand diffusers-named tensors to ``PeftAdapterMixin.load_lora_adapter``
    under the ``transformer.`` prefix it strips by default."""
    model.load_lora_adapter(
        {f"transformer.{k}": v for k, v in diffusers_sd.items()},
        prefix="transformer",
        adapter_name=adapter_name,
    )


def load_via_diffusers(
    model: Any,
    artifact: Any,
    *,
    allow_fallback: bool = True,
    adapter_name: str = "default",
) -> LoraLoadResult:
    """Load an artifact (a path or a state dict) onto a diffusers
    ``MiniMaxH3Transformer3DModel`` through diffusers' OWN converter. With the
    private symbol absent: raise when ``allow_fallback`` is False, otherwise
    run :func:`artifact_to_diffusers` and report ``fallback_used=True``. A
    converter that REJECTS the file (a ``ValueError``) is never papered over."""
    sd = _read_artifact(artifact)
    modules = len(_native_pairs(sd))
    try:
        from diffusers.loaders import lora_conversion_utils as _conversion

        converter = getattr(_conversion, _DIFFUSERS_CONVERTER)
        converted = converter(dict(sd))
    except (ImportError, AttributeError) as exc:
        if not allow_fallback:
            raise ImportError(
                f"diffusers' MiniMax-H3 LoRA converter is unavailable ({exc}) and the fallback is disabled"
            ) from exc
        _log.warning("minimax_h3_lora_converter_fallback", error=str(exc), modules=modules)
        _inject(model, artifact_to_diffusers(sd), adapter_name)
        return LoraLoadResult(fallback_used=True, modules=modules, adapter_name=adapter_name, via="lora_keys")
    model.load_lora_adapter(converted, prefix="transformer", adapter_name=adapter_name)
    return LoraLoadResult(fallback_used=False, modules=modules, adapter_name=adapter_name, via="diffusers")


def load_via_lora_keys(model: Any, artifact: Any, *, adapter_name: str = "default") -> LoraLoadResult:
    """The inbound path: any supported layout → OUR inverse → the model."""
    sd = _read_artifact(artifact)
    diffusers_sd = artifact_to_diffusers(sd)
    _inject(model, diffusers_sd, adapter_name)
    return LoraLoadResult(
        fallback_used=False, modules=len(_native_pairs(sd)), adapter_name=adapter_name, via="lora_keys"
    )
