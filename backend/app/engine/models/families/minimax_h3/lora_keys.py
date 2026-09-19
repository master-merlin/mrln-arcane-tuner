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
