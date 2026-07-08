"""hv15 saver + t2v↔i2v LoRA portability round trip.

diffusers 0.39 has NO pipeline-level LoRA mixin for HunyuanVideo 1.5 (the old
``HunyuanVideoLoraLoaderMixin`` targets the ORIGINAL model's key layout) — so
our ai-toolkit keys are the record::

    diffusion_model.transformer_blocks.{i}.<module>.lora_{A,B}.weight

The t2v and i2v checkpoints share ONE transformer layout (hub configs are
identical except ``task_type``), so a LoRA saved from a t2v-config model must
load onto an i2v-config model with zero missing / zero unexpected LoRA keys —
and vice versa. Key count is pinned.
"""

from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from app.engine.models.families.hunyuan_video15.driver import (
    Hv15Driver,
    hv15_lora_target_paths,
)

from .test_hv15_lora_targets import TINY_CFG


def _tiny_peft_model(task_type: str):
    from diffusers import HunyuanVideo15Transformer3DModel
    from peft import LoraConfig, get_peft_model

    cfg = dict(TINY_CFG)
    cfg["task_type"] = task_type
    base = HunyuanVideo15Transformer3DModel(**cfg)
    lora_cfg = LoraConfig(r=2, lora_alpha=2, target_modules=hv15_lora_target_paths(1))
    return get_peft_model(base, lora_cfg)


class _Defn:
    def __init__(self, mode: str):
        self.family = "hunyuan_video15"
        self.id = f"hv15-test-{mode}"
        self.architecture_params = {"mode": mode, "transformer.num_layers": 1}
        self.lora_targetable_modules: list[str] = []


def _save_lora(mode: str, tmp_path, filename: str):
    model = _tiny_peft_model("t2v" if mode == "t2v" else "i2v")
    driver = Hv15Driver(_Defn(mode), torch.device("cpu"))
    saver = driver.get_saver()
    path = tmp_path / filename
    saver.save(components={"unet": model, "config": {}}, path=path, metadata=None)
    assert path.exists(), "saver did not produce a safetensors file"
    return path


def test_saved_keys_are_ai_toolkit_canonical(tmp_path):
    path = _save_lora("t2v", tmp_path, "t2v_lora.safetensors")
    sd = load_file(str(path))
    assert sd, "saved state dict is empty"

    # PINNED key count: 1 block x 12 modules x (lora_A + lora_B) = 24.
    assert len(sd) == 24
    for key in sd:
        assert key.startswith("diffusion_model.transformer_blocks."), key
        assert key.endswith(("lora_A.weight", "lora_B.weight")), key
    # No refiner / embedder keys leaked.
    assert not any("token_refiner" in k or "context_embedder" in k for k in sd)


def test_saver_metadata_records_mode(tmp_path):
    path = _save_lora("i2v", tmp_path, "i2v_lora.safetensors")
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    assert metadata["modelspec.architecture"] == "hunyuanvideo-1.5-i2v"
    assert metadata["ss_network_dim"] == "2"


def _load_onto(path, task_type: str) -> tuple[list[str], list[str]]:
    """Load the saved LoRA onto a fresh PEFT model; return missing/unexpected
    LoRA keys (strict=False load with prefix mapping back to PEFT names)."""
    target = _tiny_peft_model(task_type)
    sd = load_file(str(path))
    mapped = {
        "base_model.model." + k[len("diffusion_model.") :]: v for k, v in sd.items()
    }
    # PEFT default adapter keys carry a ".default." segment in named params.
    peft_named = {
        n: p for n, p in target.named_parameters() if "lora_A" in n or "lora_B" in n
    }
    remapped = {}
    for k, v in mapped.items():
        # saved: ...lora_A.weight → module param: ...lora_A.default.weight
        peft_key = k.replace("lora_A.weight", "lora_A.default.weight").replace(
            "lora_B.weight", "lora_B.default.weight"
        )
        remapped[peft_key] = v

    missing = [k for k in peft_named if k not in remapped]
    unexpected = [k for k in remapped if k not in peft_named]
    # Shape agreement for everything that matched.
    for k, v in remapped.items():
        if k in peft_named:
            assert peft_named[k].shape == v.shape, k
    return missing, unexpected


def test_t2v_lora_round_trips_onto_i2v(tmp_path):
    path = _save_lora("t2v", tmp_path, "t2v_lora.safetensors")
    missing, unexpected = _load_onto(path, "i2v")
    assert missing == [], f"missing LoRA keys on i2v: {missing[:5]}"
    assert unexpected == [], f"unexpected LoRA keys on i2v: {unexpected[:5]}"


def test_i2v_lora_round_trips_onto_t2v(tmp_path):
    path = _save_lora("i2v", tmp_path, "i2v_lora.safetensors")
    missing, unexpected = _load_onto(path, "t2v")
    assert missing == [], f"missing LoRA keys on t2v: {missing[:5]}"
    assert unexpected == [], f"unexpected LoRA keys on t2v: {unexpected[:5]}"


def test_full_size_key_count_projection():
    """54 blocks x 12 modules x 2 tensors = 1296 keys on the real model —
    derived from the same generator the driver uses (pinned here so a target
    change shows up as an explicit diff)."""
    assert len(hv15_lora_target_paths(54)) * 2 == 1296
