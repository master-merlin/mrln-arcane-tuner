# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored into MRLN Arcane Tuner from huggingface/diffusers at commit 6256aa7666
# ("Add Qwen-Image 2.1 (#14804)", 2026-09-18),
# src/diffusers/pipelines/qwenimage/pipeline_qwenimage21.py.
# Changes from upstream -- TRIMMED to the encode and pack helpers:
#   - `QwenImage21Pipeline` becomes `QwenImage21PipelineHelpers`, a plain object holding the same
#     `vae` / `text_encoder` / `processor` attributes instead of a `DiffusionPipeline`; `__call__`,
#     `check_inputs`, `retrieve_timesteps`, the LoRA loader mixin and the XLA hooks are not carried.
#   - `__init__` takes every component as optional so a caller may hold only the part it needs; the
#     processor-derived `_drop_idx` / `_img_token_id` are computed only when a processor is given.
#   - `_execution_device` is the text encoder's device (upstream: the pipeline's execution device).
#   - Imports are absolute against the pinned diffusers release.
# Every method body below is upstream's, unchanged (upstream lines: calculate_shift 61-72,
# retrieve_latents 135-146, calculate_dimensions 149-156, __init__ 182-226, _extract_masked_hidden
# 228-232, _get_qwen_prompt_embeds 234-331, encode_prompt 333-387, _pack_latents / _unpack_latents
# 409-420, _encode_vae_image 422-444, prepare_latents 446-485).
# Retire this copy when a diffusers release exports QwenImage21Pipeline
# (backend/app/engine/tests/test_qwen_image2_vendor.py fails that day).

import math

import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image as PILImage


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.flux.pipeline_flux_control_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.calculate_dimensions
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class QwenImage21PipelineHelpers:
    r"""
    The encode and pack helpers of diffusers' `QwenImage21Pipeline` (commit 6256aa7666), without the pipeline.

    Args:
        vae ([`AutoencoderKLQwenImage21`], *optional*):
            Variational auto-encoder mapping images to and from the 64-channel latent space.
        text_encoder ([`Qwen3VLForConditionalGeneration`], *optional*):
            Qwen3-VL model producing the joint text/image embeddings.
        processor ([`Qwen3VLProcessor`], *optional*):
            Processor that builds the chat template and tokenizes prompt and condition images.
    """

    def __init__(self, vae=None, text_encoder=None, processor=None):
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        # The VAE compresses 16x spatially and the transformer consumes latents unpatched, so one token covers a 16x16
        # pixel tile.
        self.vae_scale_factor = 16
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 64
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels
        )
        self.sys_prompt = "Comprehend and analyze the provided prompt."
        # The prompt is built as a raw template string and passed straight to
        # `self.processor(text=..., images=...)`, rather than going through `apply_chat_template`:
        # the two tokenize differently and the checkpoint expects this one. The "Picture 1: ..."
        # vision prefix only appears in the image-conditioned template.
        self.prompt_template_t2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prompt_template_ti2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if self.processor is not None:
            # Number of leading system-role tokens to drop from the hidden states. Derived from the
            # tokenized system message rather than hardcoded, so it tracks the processor's template.
            sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
            sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
            self._drop_idx = len(sys_tokens[0])
            self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

    @property
    def _execution_device(self):
        return self.text_encoder.device

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: list | None = None,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
        prompt = [" " if not p else p for p in prompt]
        is_t2i = image is None

        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "<image1><|vision_start|><|image_pad|><|vision_end|>", replace
                )
                prompts.append(template.format(t))
            # Each prompt's template repeats the `<|image_pad|>` placeholders, so hand the processor one set of
            # images per prompt, in the order the placeholders appear.
            for _ in prompt:
                for img in image:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == "RGBA":
                        # The checkpoint was trained with the alpha composited over white for the vision encoder.
                        # Only this copy is flattened; the VAE still reads all four channels.
                        white = PILImage.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)

        # Left padding, as the checkpoint was trained with. `_extract_masked_hidden` drops the padding either way,
        # but the side decides the positions the encoder sees for a batch of prompts of different lengths.
        processor_kwargs = {
            "text": prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if not is_t2i:
            processor_kwargs["images"] = condition_pil_list

        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # `hidden_states[-1]` has to be the last decoder layer's output, before the text encoder's final RMSNorm:
        # that is what the transformer was trained on. It is what transformers 4.x returns there, but from
        # transformers 5.0 the output capturing ties that entry to `last_hidden_state`, so it comes back normalized
        # instead — a third of the signal the transformer reads, which shows up first in rendered text. A forward hook
        # returning the module's input replaces its output, which neutralizes the norm for this call on either version.
        # TODO: replace this with `tie_last_hidden_states=False` in the text encoder's config, which
        # huggingface/transformers#48087 adds, once that ships in a stable transformers release (5.18).
        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx :] for e in split_hidden_states]

        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[self._drop_idx :] for e in image_pad_mask]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[PipelineImageInput] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ):
        r"""
        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt to be encoded.
            image (`list[PipelineImageInput]`, *optional*):
                Condition images to encode alongside the prompt.
            device (`torch.device`):
                Torch device.
            num_images_per_prompt (`int`):
                Number of images generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Skips encoding when provided.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        elif image_pad_mask is None:
            if image is not None:
                raise ValueError(
                    "Pass `image_pad_mask` alongside `prompt_embeds` when the embeddings cover condition images, so "
                    "the transformer knows which positions hold image tokens."
                )
            # Embeddings supplied without a mask can only be text, so no position holds an image token.
            image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2], dtype=torch.bool)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        # `repeat(1, n)` on the 2D mask, so its rows interleave the same way the 3D embeddings' do. With
        # `repeat(1, n, 1)` the mask picks up a leading axis and the rows come out tiled instead, which pairs each
        # sample with another prompt's padding.
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        # Without padding there is nothing to mask, and a mask that carries no information costs the attention
        # backends that reject one outright.
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        # 2.1 consumes latents unpatched, so packing is a plain spatial flatten.
        return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.transpose(1, 2).reshape(batch_size, channels, 1, height, width)
        return latents

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self, images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image_latents = None
        if images is not None:
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                encoded = self._encode_vae_image(image, generator)
                if batch_size > encoded.shape[0]:
                    if batch_size % encoded.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {encoded.shape[0]} to {batch_size} text prompts."
                        )
                    encoded = torch.cat([encoded] * (batch_size // encoded.shape[0]), dim=0)
                image_latent_height, image_latent_width = encoded.shape[3:]
                all_image_latents.append(
                    self._pack_latents(
                        encoded, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                )
            image_latents = torch.cat(all_image_latents, dim=1)

        if latents is None:
            shape = (batch_size, 1, num_channels_latents, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents
