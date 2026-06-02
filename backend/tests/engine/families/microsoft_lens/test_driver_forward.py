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


def _tiny_dit():
    from app.engine.models.families.microsoft_lens.vendor.transformer import (
        LensTransformer2DModel,
    )
    return LensTransformer2DModel(
        patch_size=2, in_channels=128, out_channels=32, num_layers=1,
        attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=2880, axes_dims_rope=(2, 2, 4),
        gate_mlp=True, rms_norm=True, multi_layer_encoder_feature=True,
        selected_layer_index=(5, 11, 17, 23),
    )


class _FakeBN:
    running_mean = torch.full((128,), 2.0)
    running_var = torch.full((128,), 4.0)
    eps = 1e-5


class _FakeVAE:
    bn = _FakeBN()


def test_prepare_latents_and_noise_shapes():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.vae = _FakeVAE()
    vae_latent = torch.randn(2, 32, 4, 4)        # h=w=4 -> S=(2)(2)=4
    seq = drv.prepare_latents(vae_latent)
    assert seq.shape == (2, 4, 128)
    noise_seq = drv.prepare_noise(torch.randn(2, 32, 4, 4))
    assert noise_seq.shape == (2, 4, 128)


def test_forward_pass_runs_and_matches_seq_shape():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.transformer = _tiny_dit().eval()
    b, s = 1, 4  # latent_h = latent_w = 2 -> S = 4
    noisy = torch.randn(b, s, 128)
    text = torch.randn(b, 4, 5, 2880)            # 4 layers, S_txt=5
    mask = torch.ones(b, 5, dtype=torch.bool)
    ts = torch.tensor([0.5])
    batch = {"latent_h": 2, "latent_w": 2}
    out = drv.forward_pass(noisy, ts, (text, mask), batch)
    assert out.shape[0] == b and out.shape[1] == s


def test_forward_pass_converts_timesteps_to_unit_range():
    """The trainer/scheduler convention is [0,1000]; the Lens DiT expects the
    flow-matching value in [0,1]. forward_pass must divide by 1000 so the model
    is told the same noise fraction its input was mixed at."""
    class _CaptureTS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, *, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, timestep, img_shapes):
            self.seen = timestep.detach().clone()
            return torch.zeros_like(hidden_states)

    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    cap = _CaptureTS()
    drv.transformer = cap
    noisy = torch.randn(1, 4, 128)
    text = torch.randn(1, 4, 5, 2880)
    mask = torch.ones(1, 5, dtype=torch.bool)
    drv.forward_pass(noisy, torch.tensor([500.0]), (text, mask),
                     {"latent_h": 2, "latent_w": 2})
    assert torch.allclose(cap.seen, torch.tensor([0.5]))  # 500 / 1000


def test_prepare_latents_bn_normalizes_but_noise_does_not():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.vae = _FakeVAE()
    x = torch.randn(2, 32, 4, 4)
    lat = drv.prepare_latents(x)
    noi = drv.prepare_noise(x)  # same input
    assert lat.shape == noi.shape == (2, 4, 128)
    # latents are BN-normalized (mean=2, std=2); noise is raw -> must differ.
    assert not torch.allclose(lat, noi)
    # And the noise path equals plain patchify (no BN).
    from app.engine.models.families.microsoft_lens import utils
    assert torch.allclose(noi, utils.patchify_to_seq(x))


def test_forward_pass_requires_both_latent_dims():
    import pytest
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.transformer = _tiny_dit().eval()
    noisy = torch.randn(1, 4, 128)
    text = torch.randn(1, 4, 5, 2880)
    mask = torch.ones(1, 5, dtype=torch.bool)
    ts = torch.tensor([0.5])
    with pytest.raises(ValueError):
        drv.forward_pass(noisy, ts, (text, mask), {"latent_h": 2})  # latent_w missing


def test_gradient_checkpointing_enable_and_backward():
    """enable_gradient_checkpointing must not raise and the checkpointed
    forward must run under grad and produce gradients."""
    dit = _tiny_dit().train()
    assert dit.gradient_checkpointing is False
    dit.enable_gradient_checkpointing()  # diffusers ModelMixin -> _set_gradient_checkpointing
    assert dit.gradient_checkpointing is True

    noisy = torch.randn(1, 4, 128, requires_grad=True)  # latent_h=latent_w=2 -> S=4
    text = [torch.randn(1, 5, 2880) for _ in range(4)]
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = dit(
        hidden_states=noisy,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=mask,
        timestep=torch.tensor([0.5]),
        img_shapes=[(1, 2, 2)],
    )
    assert out.shape == (1, 4, 128)
    out.sum().backward()
    assert noisy.grad is not None
    # A LoRA-targetable block weight must receive a gradient through the
    # checkpointed path.
    qkv = dit.transformer_blocks[0].attn.img_qkv.weight
    assert qkv.grad is not None


def test_forward_pass_uses_stashed_nonsquare_grid():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.vae = _FakeVAE()
    drv.transformer = _tiny_dit().eval()
    # Non-square VAE latent [1,32,4,8] -> patchify grid (2,4) -> S=8.
    lat = drv.prepare_latents(torch.randn(1, 32, 4, 8))
    assert lat.shape == (1, 8, 128)
    assert (drv._latent_h, drv._latent_w) == (2, 4)
    text = torch.randn(1, 4, 5, 2880)
    mask = torch.ones(1, 5, dtype=torch.bool)
    # No latent_h/latent_w in batch -> must use the stashed (2,4), NOT isqrt(8) (which would raise).
    out = drv.forward_pass(lat, torch.tensor([0.5]), (text, mask), {})
    assert out.shape[0] == 1 and out.shape[1] == 8
