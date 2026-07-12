"""WAN 2.2 load-time host-RAM contract (no GPU / no weights).

WAN 2.2 A14B is the repo's only dual-transformer family: TWO ~14B experts, and
the stock ``Wan-AI/Wan2.2-*-A14B-Diffusers`` checkpoints ship each expert in
**fp32** (~53 GB on disk PER expert). The phased loader stages every component
on CPU first (``initial_device="cpu"``), so a ``both``-expert run already pins
~2×28 GB of bf16 transformer in host RAM before Phase B moves them to the GPU.

If ``low_cpu_mem_usage`` is NOT active, ``from_pretrained`` additionally
materialises the full fp32 ``state_dict`` (~53 GB) per expert BEFORE casting to
bf16 — turning the staging peak into ~2×(53 GB + 28 GB) of transient host RAM,
which fills system memory and hangs the machine (the reported bug).

These tests pin that **every** WAN 2.2 transformer spec forces the streamed
(``low_cpu_mem_usage=True``) load path, and that the flag actually reaches
``WanTransformer3DModel.from_pretrained`` for BOTH experts.
"""

from __future__ import annotations

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.models.families.wan22.loader import Wan22Loader

import torch


class _Defn:
    architecture_params = {"mode": "t2v", "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []


def _specs(expert_mode: str) -> dict[str, ComponentSpec]:
    loader = Wan22Loader(torch.device("cpu"), expert_mode=expert_mode)
    return {s.key: s for s in loader.get_component_manifest(_Defn())}


# ── Manifest carries the streamed-load pin on EVERY expert spec ────────────


def test_both_experts_pin_low_cpu_mem_usage():
    specs = _specs("both")
    for key in ("unet", "unet_low"):
        assert specs[key].load_kwargs.get("low_cpu_mem_usage") is True, key


def test_single_expert_high_pins_low_cpu_mem_usage():
    specs = _specs("high")
    assert "unet_low" not in specs
    assert specs["unet"].load_kwargs.get("low_cpu_mem_usage") is True


def test_single_expert_low_pins_low_cpu_mem_usage():
    specs = _specs("low")
    # transformer_2/ is loaded AS the single primary "unet".
    assert specs["unet"].subfolder == "transformer_2"
    assert specs["unet"].load_kwargs.get("low_cpu_mem_usage") is True


def test_non_transformer_specs_do_not_force_low_cpu_mem_usage():
    """Only the big experts are pinned — TE/VAE/tokenizer keep loader defaults."""
    specs = _specs("both")
    for key in ("text_encoder", "vae", "tokenizer"):
        assert "low_cpu_mem_usage" not in specs[key].load_kwargs, key


# ── The pin actually reaches from_pretrained for BOTH experts ──────────────


def test_low_cpu_mem_usage_reaches_from_pretrained_both_experts():
    """``spec.load_kwargs`` is merged into the real ``from_pretrained`` call.

    Exercises the REAL ``GenericComponentLoader._load_component`` (the same code
    the training loop runs) with a recording fake class, once per expert spec,
    and asserts the streamed-load flag is forwarded alongside the bf16 dtype.
    """
    calls: list[dict] = []

    class _RecordingModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append({"path": path, **kwargs})
            return object()

    specs = _specs("both")
    for key in ("unet", "unet_low"):
        GenericComponentLoader._load_component(
            _RecordingModel,
            f"/fake/{key}",
            torch.bfloat16,
            specs[key],
        )

    assert len(calls) == 2
    for call in calls:
        assert call["low_cpu_mem_usage"] is True
        assert call["torch_dtype"] is torch.bfloat16
