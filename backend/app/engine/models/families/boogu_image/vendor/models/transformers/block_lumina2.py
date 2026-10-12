# Vendored from boogu-project/Boogu-Image @ ac9e40c1350fd60c502137a678ad1001d51e2ae7 (2026-07-10)
# Source: boogu/models/transformers/block_lumina2.py
# vendored for boogu_image family; reduced in LANE-150 to the call-site adapter over
# diffusers 0.41.0 (the norm and feed-forward classes are imported from diffusers).

import torch
from diffusers.models.transformers.transformer_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding as _Lumina2CombinedTimestepCaptionEmbedding,
)


class Lumina2CombinedTimestepCaptionEmbedding(_Lumina2CombinedTimestepCaptionEmbedding):
    """The release's caption embedding under Boogu's constructor and forward signature.

    The release names the caption width ``cap_feat_dim`` and has no timestep scale; Boogu
    names it ``instruction_feat_dim``, stores ``timestep_scale`` and scales the sinusoid
    argument with it (``Timesteps.scale``, applied after ``t * freq`` exactly as the
    vendored copy did; scaling the timestep first rounds differently in fp32). The forward takes
    ``(timestep, instruction_hidden_states, dtype)``: the release casts the sinusoid with
    ``type_as(hidden_states)``, so a one-element tensor of ``dtype`` carries it.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            cap_feat_dim=instruction_feat_dim,
            frequency_embedding_size=frequency_embedding_size,
            norm_eps=norm_eps,
        )
        self.timestep_scale = timestep_scale
        self.time_proj.scale = timestep_scale

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype_carrier = torch.empty(0, dtype=dtype, device=timestep.device)
        return super().forward(dtype_carrier, timestep, instruction_hidden_states)
