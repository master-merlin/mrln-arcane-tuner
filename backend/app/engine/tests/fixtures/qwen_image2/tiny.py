"""Tiny, CPU-sized stand-ins for every Qwen-Image 2.1 component (LANE-132 tests).

Loaded by path from ``tests/test_qwen_image2_*.py`` (pytest runs with
``--import-mode=importlib``, so a test module cannot import a sibling).

Every builder returns a REAL class with random weights -- the vendored
``QwenImage21Transformer2DModel`` / ``AutoencoderKLQwenImage21``, a real
``Qwen3VLForConditionalGeneration`` / ``Qwen2_5_VLForConditionalGeneration``,
and a real ``Qwen3VLProcessor`` over a byte-level tokenizer that carries the
checkpoint's special tokens. Nothing here is a mock: the seam under test runs
real code on small tensors.

The processor's chat template reproduces the one line upstream's ``_drop_idx``
reads -- ``<|im_start|>system\\n{text}<|im_end|>\\n`` -- which is what the
checkpoint's own ``processor/chat_template.jinja`` renders for a system message
(measured on Qwen/Qwen-Image-2.1 @ 790c92633540:
``.agent/workdir/lane-132/t3/probe_proc.py`` printed exactly that string).
"""

from __future__ import annotations

import torch

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{{ c['text'] }}{% endfor %}{% endif %}"
    "<|im_end|>\n{% endfor %}"
)

TEXT_HIDDEN = 32
VOCAB = 300  # > 256 byte tokens + the special tokens above


def tiny_processor():
    """A real ``Qwen3VLProcessor`` over a byte-level BPE with no merges."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL_TOKENS)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<|endoftext|>", eos_token="<|im_end|>",
    )
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=fast,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=CHAT_TEMPLATE,
    )


def tiny_qwen3vl_te(seed: int = 0):
    """``Qwen3VLForConditionalGeneration``, 2 text layers of width 32, fp32."""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    torch.manual_seed(seed)
    cfg = Qwen3VLConfig(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 4096,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            "rms_norm_eps": 1e-6,
            "bos_token_id": 256,
            "eos_token_id": 258,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": TEXT_HIDDEN,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
        },
    )
    return Qwen3VLForConditionalGeneration(cfg).eval()


def tiny_qwen25vl_te(seed: int = 0):
    """``Qwen2_5_VLForConditionalGeneration`` of the same width (another encoder identity)."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]},
            "bos_token_id": 256,
            "eos_token_id": 258,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": TEXT_HIDDEN,
            "fullatt_block_indexes": [0],
        },
    )
    return Qwen2_5_VLForConditionalGeneration(cfg).eval()


def tiny_transformer(seed: int = 0, in_channels: int = 8):
    """The vendored single-stream DiT: 2 blocks, 2 heads x 8, context width 32."""
    from app.engine.models.families.qwen_image2.vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    torch.manual_seed(seed)
    return QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=in_channels,
        out_channels=in_channels,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=TEXT_HIDDEN,
        mlp_ratio=2,
        axes_dims_rope=(2, 2, 4),
    )


def tiny_vae(seed: int = 0):
    """The vendored VAE with the checkpoint's geometry (z_dim 64, 4 channels,
    16x spatial) but a base width of 4, so a 512x512 encode runs on a CPU."""
    from app.engine.models.families.qwen_image2.vendor.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
    )

    torch.manual_seed(seed)
    return AutoencoderKLQwenImage21(
        base_dim=4, decoder_base_dim=4, z_dim=64, num_res_blocks=1,
    ).eval()
