"""Qwen-Image 2.1 text encoding -- the text-to-image half of upstream ``encode_prompt``.

Evidence: diffusers ``6256aa7666`` ``pipelines/qwenimage21/pipeline_qwenimage21.py``
lines 274-329 (``_get_qwen_prompt_embeds``), and ``__init__`` for the template
and ``_drop_idx``. Re-implemented here, never delegated to the vendored helper:
``tests/test_qwen_image2_text.py`` uses that helper as the ORACLE, and a family
that called it would make the parity check compare the helper with itself.

Upstream's contract, step by step:

1. an empty prompt becomes ``" "`` (Qwen has no BOS token, so an empty string
   leaves the encoder nothing to read);
2. the prompt is formatted into the raw text-to-image template (not through
   ``apply_chat_template``: the two tokenize differently);
3. the processor tokenizes with ``padding_side="left"``;
4. the text encoder runs with its final RMSNorm BYPASSED -- transformers 5.x
   ties ``hidden_states[-1]`` to the normalized ``last_hidden_state``, and the
   transformer was trained on the pre-norm activations;
5. the padding is dropped per prompt, then the first ``_drop_idx`` tokens (the
   system message, measured by tokenizing it through the processor's own chat
   template);
6. every prompt is zero-padded on the RIGHT to the batch's longest, with a
   matching 0/1 mask.

One addition, not upstream's: ``max_length`` caps the post-template tokens
(the qwen_image sibling's truncation, ``qwen_image/driver.py``). Upstream
truncates nothing; the cap (``te.max_length``, 512) sits far above a real
caption, and it is part of the TE-cache key.
"""

from __future__ import annotations

import hashlib

import torch

SYS_PROMPT = "Comprehend and analyze the provided prompt."

PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

#: Fingerprint of the pre-encode transformation. Part of the TE disk-cache key,
#: so an edit to the template or the system prompt never serves embeddings
#: encoded under the old one (the dreamlite poisoned-cache class).
TEMPLATE_FINGERPRINT = hashlib.sha256(
    f"{SYS_PROMPT}\x00{PROMPT_TEMPLATE_T2I}".encode("utf-8")
).hexdigest()[:16]


def drop_idx(processor) -> int:
    """Number of leading system-message tokens, measured the way upstream does."""
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
    sys_tokens = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
    return len(sys_tokens[0])


def te_cache_identity(text_encoder, max_length: int) -> str:
    """Every input besides the caption that decides a cached embedding.

    The directory already separates definition, dataset and TE quantization
    (``pipeline_caching._resolve_te_cache_dirs``); the filename adds the
    template, the encoder's class and the token cap. Two encoders of the same
    width (Qwen2.5-VL and Qwen3-VL are both 4096 at full size) would otherwise
    read each other's tensors without a shape error.
    """
    return f"qwen_image2/t2i/{TEMPLATE_FINGERPRINT}/{type(text_encoder).__name__}/max{int(max_length)}"


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def encode_prompts(
    text_encoder,
    processor,
    captions: list[str],
    *,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(embeddings [B, L, D], attention_mask [B, L] long)`` for ``captions``."""
    drop = drop_idx(processor)
    prompts = [PROMPT_TEMPLATE_T2I.format(c if c else " ") for c in captions]
    model_inputs = processor(
        text=prompts,
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=int(max_length) + drop,
        return_tensors="pt",
    ).to(device)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
    }
    if "mm_token_type_ids" in model_inputs:
        forward_kwargs["mm_token_type_ids"] = model_inputs["mm_token_type_ids"]

    # A forward hook that returns the module's INPUT replaces its output: the
    # final RMSNorm becomes the identity for this call only (upstream's fix).
    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            outputs = text_encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden_states = outputs.hidden_states[-1]

    split = [e[drop:] for e in _extract_masked_hidden(hidden_states, model_inputs.attention_mask)]
    max_seq_len = max(e.size(0) for e in split)
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split]
    )
    mask = torch.stack(
        [
            torch.cat([
                torch.ones(u.size(0), dtype=torch.long, device=u.device),
                torch.zeros(max_seq_len - u.size(0), dtype=torch.long, device=u.device),
            ])
            for u in split
        ]
    )
    return embeds.to(dtype=dtype), mask
