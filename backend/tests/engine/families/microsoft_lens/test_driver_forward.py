"""microsoft_lens driver tests."""
import torch

from app.engine.models.families.microsoft_lens.driver import MicrosoftLensDriver
from app.engine.core.definitions import ModelDefinition


def _defn():
    return ModelDefinition(
        id="microsoft-lens-base", family="microsoft_lens", name="Lens Base",
        defaults={}, components={},
    )


def test_lora_targets_default_attn_and_mlp():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    targets = drv.get_lora_targets()
    for expected in ["img_qkv", "txt_qkv", "to_out.0", "to_add_out", "w1", "w2", "w3"]:
        assert expected in targets


def test_loading_dtype_is_bf16():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    assert drv.resolve_loading_dtype() == torch.bfloat16


def test_init_scheduler_is_none_for_flow_matching():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    assert drv.init_scheduler() is None


def test_arch_params_override_selected_layers_and_offset():
    defn = ModelDefinition(
        id="x", family="microsoft_lens", name="x", defaults={},
        components={},
        architecture_params={
            "transformer.selected_layer_index": [2, 4, 6, 8],
            "te.txt_offset": 50,
            "te.max_length": 256,
        },
    )
    drv = MicrosoftLensDriver(defn, torch.device("cpu"))
    assert drv.selected_layers == (2, 4, 6, 8)
    assert drv.hf_layer_indices == [3, 5, 7, 9]
    assert drv.txt_offset == 50
    assert drv.te_max_length == 256
