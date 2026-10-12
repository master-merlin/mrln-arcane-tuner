# Vendored from VectorSpaceLab/OmniGen2 @ 18e6f9d5271b517fcb32e999f10df943ae9b8f20 (2026-07-13)
# Source: omnigen2/models/transformers/block_lumina2.py
# Apache-2.0 - vendored for the omnigen2 family; reduced in LANE-150 to the call-site adapter
# over diffusers 0.41.0 (the norm and feed-forward classes are imported from diffusers).

import torch
from diffusers.models.transformers.transformer_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding as _Lumina2CombinedTimestepCaptionEmbedding,
)


class Lumina2CombinedTimestepCaptionEmbedding(_Lumina2CombinedTimestepCaptionEmbedding):
    """The release's caption embedding under OmniGen2's constructor and forward signature.

    The release names the caption width ``cap_feat_dim`` and has no timestep scale; OmniGen2
    names it ``text_feat_dim``, stores ``timestep_scale`` and the transformer receives a
    [0, 1) timestep, so the adapter multiplies it by that scale before the release's sinusoid
    (the family's ``transformer.timestep_scale`` is 1000; callers never scale). The forward takes
    ``(timestep, text_hidden_states, dtype)``: the release casts the sinusoid with
    ``type_as(hidden_states)``, so an empty tensor of ``dtype`` carries it.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        text_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            cap_feat_dim=text_feat_dim,
            frequency_embedding_size=frequency_embedding_size,
            norm_eps=norm_eps,
        )
        self.timestep_scale = timestep_scale

    def forward(
        self,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype_carrier = torch.empty(0, dtype=dtype, device=timestep.device)
        return super().forward(dtype_carrier, timestep * self.timestep_scale, text_hidden_states)
