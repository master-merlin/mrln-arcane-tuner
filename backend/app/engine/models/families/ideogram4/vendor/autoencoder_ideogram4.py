"""Vendored Ideogram 4 custom KL autoencoder (the "Flux2 KL autoencoder").

Ported faithfully from the public upstream source at
``github.com/ideogram-oss/ideogram4`` (``src/ideogram4/autoencoder.py``).
Only the ``nn.Module`` graph + the ``convert_diffusers_state_dict`` key-rewriter
are brought over. Inference-only / pipeline / safety machinery is intentionally
NOT vendored. The single external dependency in upstream (``einops.rearrange``)
is replaced by plain ``torch`` reshapes/permutes so this module has no third-party
deps beyond ``torch``.

The top-level class is exposed as ``Ideogram4AutoEncoder`` (upstream name:
``AutoEncoder``); ``AutoEncoderParams`` and ``convert_diffusers_state_dict`` keep
their upstream names and are importable from this module.

================================ CONTRACT ================================
Recorded directly from the ported code + the upstream pipeline's use of it
(``pipeline_ideogram4.py::_decode`` / ``_load_autoencoder``). Consumed by the
driver / latent-norm task.

CLASS:  Ideogram4AutoEncoder(params: AutoEncoderParams)
        (upstream ``AutoEncoder``). Holds three submodules:
          - ``encoder``  (Encoder)        : image -> raw latent stats
          - ``decoder``  (Decoder)        : latent -> image
          - ``bn``       (BatchNorm2d)    : see LATENT NORM below

LATENT CHANNELS:  z_channels = 32 (default ``AutoEncoderParams.z_channels``).
        The Encoder's ``conv_out`` emits ``2 * z_channels = 64`` channels
        (mean + logvar of the KL posterior); the Decoder consumes ``z_channels``
        (= 32). So the *latent channel count* is 32. NOTE the DiT operates on
        PATCHIFIED latents: patch_size=2 -> 2*2*32 = 128 token channels
        (matches the DiT ``in_channels=128``); patchify/unpatchify is done by the
        PIPELINE, not by this module.

SPATIAL COMPRESSION FACTOR:  8x per side.
        ch_mult has 4 entries -> 3 downsample stages -> 2**3 = 8. (Upstream
        ``Decoder.ffactor = 2 ** (num_resolutions - 1) = 8``; the pipeline's
        ``ae_scale_factor = 8`` agrees.)

ENCODE / DECODE METHOD NAMES & SHAPES:
        Upstream defines NO ``encode``/``decode``/``forward`` on ``AutoEncoder``;
        the pipeline calls ``self.autoencoder.decoder(z)`` (and never the encoder
        at inference). For testability + a clean driver surface we ADD thin
        wrappers that delegate to the ported submodules WITHOUT any patchify or
        latent normalization (the pipeline owns those):
          - ``encode(x: (B, in_channels=3, H, W)) -> (B, 2*z_channels=64, H/8, W/8)``
            returns the raw Encoder output (mean+logvar concatenated; quant_conv
            applied). Sampling/splitting is the caller's job.
          - ``decode(z: (B, z_channels=32, H/8, W/8)) -> (B, out_ch=3, H, W)``
            runs the Decoder (post_quant_conv applied). Image range is the raw
            decoder output (upstream clamps to [-1, 1] downstream).

LATENT NORMALIZATION  *** important for the driver / latent-norm task ***
        The model carries a ``BatchNorm2d`` named ``bn``, BUT it is NOT used to
        normalize/denormalize latents anywhere in the inference path. It is
        constructed with ``affine=False`` over ``prod(ps) * z_channels = 2*2*32
        = 128`` channels (i.e. patchified-latent channels) and only contributes
        ``bn.running_mean`` / ``bn.running_var`` buffers to the checkpoint:

            self.ps = [2, 2]
            self.bn = torch.nn.BatchNorm2d(
              math.prod(self.ps) * params.z_channels,   # 4 * 32 = 128
              eps=self.bn_eps,        # 1e-4
              momentum=self.bn_momentum,
              affine=False,
              track_running_stats=True,
            )

        Actual latent (de)normalization in upstream is done by EXPLICIT per-channel
        affine constants over the 128 patchified channels, defined in
        ``ideogram4/latent_norm.py`` (LATENT_SHIFT / LATENT_SCALE, each length 128).
        The decode path applies (``pipeline_ideogram4.py::_decode``):

            z = z * self.latent_scale + self.latent_shift          # denormalize
            # ... unpatchify 128 -> (32, *2, *2) ...
            decoded = self.autoencoder.decoder(z)

        So: the encode side normalizes a sampled latent as ``(z - shift) / scale``
        and the decode side denormalizes as ``z * scale + shift``, where shift/scale
        are the 128-long LATENT_SHIFT/LATENT_SCALE vectors -- NOT derived from the
        ``bn`` buffers. The ``bn`` head is effectively dead weight at inference and
        is retained only so the checkpoint's ``bn.*`` keys load cleanly. The driver
        must use LATENT_SHIFT/LATENT_SCALE for latent norm, NOT this ``bn``.
==========================================================================
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        # upstream: rearrange(x, "b c h w -> b 1 (h w) c")
        q = q.view(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        k = k.view(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        v = v.view(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # upstream: rearrange(h_, "b 1 (h w) c -> b c h w")
        return h_.squeeze(1).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out)
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out)
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Ideogram4AutoEncoder(nn.Module):
    """Ideogram 4 custom KL autoencoder (upstream ``AutoEncoder``).

    See the module docstring for the full contract. The ``encode`` / ``decode``
    methods are thin local additions (upstream has none) that delegate to the
    ported ``encoder`` / ``decoder`` submodules without patchify or latent
    normalization -- those are the pipeline/driver's responsibility.
    """

    def __init__(self, params: AutoEncoderParams) -> None:
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        # NOTE: dead at inference -- present only so the checkpoint's bn.* running
        # stats load. Real latent (de)norm uses latent_norm.LATENT_SHIFT/SCALE.
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Image -> raw latent stats.

        Args:
            x: (B, in_channels, H, W) image tensor (upstream range [-1, 1]).

        Returns:
            (B, 2 * z_channels, H / 8, W / 8) mean+logvar of the KL posterior
            (quant_conv applied). Posterior sampling / split is the caller's job.
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Latent -> image.

        Args:
            z: (B, z_channels, H / 8, W / 8) latent (NOT patchified, NOT
                latent-normalized -- caller denormalizes + unpatchifies first).

        Returns:
            (B, out_ch, H, W) decoded image (raw; upstream clamps to [-1, 1]).
        """
        return self.decoder(z)


_NUM_RESOLUTIONS = 4


def convert_diffusers_state_dict(src: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    attn_substrings = (".mid.attn_1.",)
    for src_key, tensor in src.items():
        dst_key = _rewrite_diffusers_key(src_key)
        if dst_key is None:
            raise KeyError(f"Unrecognized diffusers VAE state-dict key: {src_key}")
        if (
            any(s in dst_key for s in attn_substrings)
            and dst_key.endswith(".weight")
            and tensor.ndim == 2
        ):
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        out[dst_key] = tensor
    return out


def _rewrite_diffusers_key(key: str) -> str | None:
    if key.startswith("bn."):
        return key

    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "encoder.quant_conv.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

    if key == "encoder.conv_norm_out.weight":
        return "encoder.norm_out.weight"
    if key == "encoder.conv_norm_out.bias":
        return "encoder.norm_out.bias"
    if key == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if key == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"

    m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"{side}.mid.block_{idx + 1}.{rest}"
    m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        side, rest = m.group(1), m.group(2)
        rest = (
            rest.replace("group_norm.", "norm.")
            .replace("to_q.", "q.")
            .replace("to_k.", "k.")
            .replace("to_v.", "v.")
            .replace("to_out.0.", "proj_out.")
        )
        return f"{side}.mid.attn_1.{rest}"

    m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        level, res_idx, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"encoder.down.{level}.block.{res_idx}.{rest}"
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
    if m:
        return f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        res_idx = m.group(2)
        rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
        return (
            f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.block.{res_idx}.{rest}"
        )
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        return (
            f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.upsample.conv.{m.group(2)}"
        )

    if key.startswith(
        ("encoder.conv_in.", "encoder.conv_out.", "decoder.conv_in.", "decoder.conv_out.")
    ):
        return key

    return None


# Upstream alias, for callers/tests that expect the original class name.
AutoEncoder = Ideogram4AutoEncoder
