"""WAN 2.2 TI2V-5B dense-load manifest contract (no GPU / no weights).

Unlike ``wan22`` A14B (two ~14B experts, deferred second-expert host-RAM
sequencing — see ``test_wan22_load_ram.py``), the TI2V-5B is a SINGLE dense 5B
transformer: the manifest must carry exactly ONE transformer component
(``unet``, from ``transformer/``), NEVER an ``unet_low`` / ``transformer_2``,
and NEVER a CLIP ``image_encoder`` (TI2V-5B has no vision tower in either
video_mode — ``image_dim: null``, unlike wan21's real CLIP-conditioned I2V).
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan22_ti2v_5b.loader import Wan22Ti2v5bLoader


class _Defn:
    architecture_params = {"mode": "both"}
    lora_targetable_modules: list[str] = []


def _manifest_by_key():
    loader = Wan22Ti2v5bLoader(torch.device("cpu"))
    specs = loader.get_component_manifest(_Defn())
    return {s.key: s for s in specs}


def test_manifest_has_exactly_one_transformer_no_second_expert():
    specs = _manifest_by_key()
    assert specs["unet"].subfolder == "transformer"
    assert specs["unet"].hf_class == "diffusers.WanTransformer3DModel"
    assert "unet_low" not in specs, (
        "TI2V-5B is dense — no second-expert component should ever be loaded"
    )


def test_manifest_never_loads_an_image_encoder():
    specs = _manifest_by_key()
    assert "image_encoder" not in specs
    assert "image_processor" not in specs


def test_manifest_carries_te_vae_tokenizer():
    specs = _manifest_by_key()
    assert {"tokenizer", "text_encoder", "vae", "unet"} == set(specs)
    assert specs["text_encoder"].hf_class == "transformers.UMT5EncoderModel"
    assert specs["vae"].hf_class == "diffusers.AutoencoderKLWan"
    assert specs["vae"].dtype_override == torch.float32


def test_manifest_is_identical_regardless_of_mode():
    """Unlike wan21 (I2V definitions add a CLIP branch), TI2V-5B's manifest
    never varies — there is no per-definition mode split (one YAML, mode='both')
    and no image-encoder branch to gate."""
    loader = Wan22Ti2v5bLoader(torch.device("cpu"))

    class _T2V(_Defn):
        architecture_params = {"mode": "both"}

    keys_a = {s.key for s in loader.get_component_manifest(_T2V())}
    keys_b = {s.key for s in loader.get_component_manifest(_Defn())}
    assert keys_a == keys_b == {"tokenizer", "text_encoder", "vae", "unet"}
