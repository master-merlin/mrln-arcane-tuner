"""ideogram4 loader manifest tests (no weights downloaded)."""
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.ideogram4.loader import IdeogramV4Loader


def _defn():
    return ModelDefinition(
        id="ideogram4-fp8", family="ideogram4", name="Ideogram 4",
        defaults={}, components={},
    )


def test_manifest_declares_stock_components():
    loader = IdeogramV4Loader(torch.device("cpu"))
    specs = {s.key: s for s in loader.get_component_manifest(_defn())}

    # Text encoder: Qwen3-VL in a SEPARATE repo, declared in the YAML under the
    # "text_encoder" definition key. MUST load as
    # Qwen3VLForConditionalGeneration (matches the checkpoint architecture +
    # ``model.*`` weight prefix); AutoModel silently random-inits the LLM.
    assert (
        specs["text_encoder"].hf_class
        == "transformers.Qwen3VLForConditionalGeneration"
    )
    assert specs["text_encoder"].separate_repo is True
    assert specs["text_encoder"].definition_key == "text_encoder"

    # Tokenizer: shares the Qwen3-VL repo (separate repo, same definition key).
    assert specs["tokenizer"].hf_class == "transformers.AutoTokenizer"
    assert specs["tokenizer"].is_torch_model is False
    assert specs["tokenizer"].separate_repo is True
    assert specs["tokenizer"].definition_key == "text_encoder"

    # VAE (LANE-150): the checkpoint's ``vae/`` folder is a diffusers-layout
    # AutoencoderKLFlux2 (its config says ``_class_name: AutoencoderKLFlux2``),
    # so it loads through the generic manifest path, in float32.
    assert specs["vae"].hf_class == "diffusers.AutoencoderKLFlux2"
    assert specs["vae"].subfolder == "vae"
    assert specs["vae"].dtype_override is torch.float32

    # The DiT is NOT in the manifest -- loaded by the overridden load().
    assert "unet" not in specs


def test_loader_builds_the_release_transformer_from_a_fused_checkpoint(tmp_path):
    """LANE-150: the fused on-disk layout + the legacy config spelling load into the release class.

    A tiny checkpoint written with the vendored state dict (``attention.qkv`` / ``attention.o``, task 13's
    fixture) and an ``emb_dim``/``num_heads``/``adanln_dim`` config goes through the loader's DiT path;
    no missing/unexpected-key warning, and the forward equals the vendored forward at 1e-6.
    """
    import json
    from pathlib import Path

    import numpy as np
    from diffusers import Ideogram4Transformer2DModel
    from safetensors.torch import save_file

    fixtures = Path(__file__).resolve().parents[4] / "app" / "engine" / "tests" / "fixtures" / "ideogram4"
    with np.load(fixtures / "vendor_parity.npz") as z:
        s = {k: z[k] for k in z.files}
    tdir = tmp_path / "transformer"
    tdir.mkdir()
    save_file({k[3:]: torch.from_numpy(v).contiguous() for k, v in s.items() if k.startswith("sd.")},
              str(tdir / "diffusion_pytorch_model.safetensors"))
    (tdir / "config.json").write_text(json.dumps({
        "_class_name": "Ideogram4Transformer2DModel", "emb_dim": 16, "num_heads": 2, "adanln_dim": 8,
        "in_channels": 8, "num_layers": 2, "intermediate_size": 32, "llm_features_dim": 12,
        "rope_theta": 10000, "mrope_section": [1, 1, 1], "norm_eps": 1e-5,
    }), encoding="utf-8")

    loader = IdeogramV4Loader(torch.device("cpu"))
    model = loader._load_transformer(tdir, torch.float32, torch.device("cpu"))
    assert isinstance(model, Ideogram4Transformer2DModel)
    assert not [w for w in loader.warnings if w.startswith("DiT:")], loader.warnings
    with torch.no_grad():
        out = model(
            hidden_states=torch.from_numpy(s["in.x"]), timestep=torch.from_numpy(s["in.t"]),
            encoder_hidden_states=torch.from_numpy(s["in.llm_features"]),
            position_ids=torch.from_numpy(s["in.position_ids"]), segment_ids=torch.from_numpy(s["in.segment_ids"]),
            indicator=torch.from_numpy(s["in.indicator"]), return_dict=False,
        )[0]
    diff = float((out - torch.from_numpy(s["out_fp32"])).abs().max())
    assert diff <= 1e-6, f"LANE-150 parity ideogram4/loader: max|release - fixture| = {diff:.3e} > 1e-06"
