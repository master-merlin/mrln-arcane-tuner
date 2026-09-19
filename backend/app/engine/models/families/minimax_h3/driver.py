"""MiniMax-H3 model driver — Task 6: non-training surface.

Implements the ``IModelDriver`` methods that do NOT require the real training
forward pass: component wiring, the curated LoRA target list, block topology,
and loading dtype — all sourced from the ``ModelDefinition`` (Task 4), the
single source of truth. Everything that belongs to the packed joint
audio+video forward, the ``t = 1 - sigma`` / ``v = x0 - noise`` INVERTED
flow-match contract (see ``family.py``'s module docstring), and LoRA saving
lands in PR1 and raises ``NotImplementedError`` naming it explicitly — per
the "failure is never silent" invariant, a job that somehow reaches those
methods must fail loudly, not silently train on wrong data or produce a
plausible-looking empty/None default.

``init_scheduler`` — DO NOT CHANGE without reading this
---------------------------------------------------------
``init_scheduler`` returns ``None`` and MUST keep doing exactly that. H3
trains with flow matching (like every sibling family here: ltx2, wan21,
ace_step15, zimage, flux2) — no external scheduler object at train time — so
``return None`` is not a stub, it is the correct permanent answer.

This is also a structural trap: ``app.engine.core.hook_dispatch.
TRIVIAL_BODIES`` recognizes ONLY the exact normalized body ``"return None"``
for ``init_scheduler`` as the trivial/no-op baseline. Any other body —
including one that raises ``NotImplementedError`` — is a *meaningful
override* per ``driver_meaningfully_overrides``, which silently enrolls
``minimax_h3`` into the reviewed auto-delegation allowlist
(``AUTODELEGATED_FAMILY_HOOKS`` in
``tests/engine/test_hook_wiring_meta.py``) and trips
``test_autodelegated_family_hook_set_is_exactly_expected``. The same guard
covers every hook in the derived ``CLOBBER_HOOKS`` set (``add_noise``,
``build_batch_extra``, ``compute_target``, ``get_te_cache``,
``sample_timesteps``, ``set_te_cache``, ``init_scheduler``). PR1 row 1.2
overrides ``add_noise`` / ``compute_target`` / ``sample_timesteps`` here AND
delegates each explicitly from ``MiniMaxH3Trainer`` in the same commit
(plan ordering rule 1), so the family never relies on auto-delegation and
the reviewed allowlist stays unchanged. Any further CLOBBER hook this driver
grows must land with its trainer delegation the same way.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver


def _lands_in_pr1(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"minimax_h3 {what} lands in PR1; PR0 (Task 6) ships the "
        "non-training driver surface only."
    )


class MiniMaxH3Driver(IModelDriver):
    """MiniMax-H3 driver — non-training surface (Task 6).

    Handles:
    - Component wiring (tokenizer/processor, Qwen3-VL text encoder, visual
      VAE, audio VAE, vendored transformer) per the loader manifest (Task 5).
    - LoRA target list and block topology, both read VERBATIM from the
      definition (Task 4) so the YAML stays the single source of truth —
      pinned by ``test_definition_ships_curated_target_list_matching_driver``.
    - bf16 loading dtype (every definition's ``detected_precision`` is bf16
      throughout).

    Text encoding, the joint audio+video forward pass, and LoRA saving raise
    ``NotImplementedError`` naming PR1 (see module docstring).
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device

        # Assigned by assign_components() — component keys per
        # MiniMaxH3Loader.get_component_manifest (Task 5).
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.audio_vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded MiniMax-H3 components into driver state.

        Five components per the loader manifest (Task 5): ``tokenizer``
        (processor, never moved to device), ``text_encoder`` (Qwen3-VL-32B,
        cached then unloaded before the DiT loads), ``vae`` (visual),
        ``audio_vae`` (MONO — run once per stereo channel), ``transformer``
        (vendored; ``transformer`` or ``transformer_ref`` subfolder per
        definition).
        """
        self._components = components
        self.transformer = components.get("transformer")
        self.vae = components.get("vae")
        self.audio_vae = components.get("audio_vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return the single Qwen3-VL text encoder."""
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """Return the definition's curated target list VERBATIM.

        The YAML (Task 4) is the single source of truth — this method must
        never compute, filter, or otherwise derive its own list, or the two
        could drift silently (the nucleus_image contract). Pinned by
        ``test_definition_ships_curated_target_list_matching_driver``.
        """
        return list(self.definition.lora_targetable_modules)

    def init_scheduler(self) -> Any:
        # ``None`` (not a raise) deliberately: H3 trains with flow matching,
        # like every sibling family here — this IS the correct answer, not a
        # placeholder. See the module docstring for why this exact body must
        # never change (the TRIVIAL_BODIES / auto-delegation guard trap).
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """H3 checkpoints are bf16 throughout (every definition's
        ``detected_precision``: text_encoder/vae/unet all ``torch.bfloat16``)."""
        return torch.bfloat16

    # --- Phase 2: Text Encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        raise _lands_in_pr1(
            "text encoding (Qwen3-VL, hidden_state_tap_index=50)"
        )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported — Qwen3-VL-32B stays frozen.

        This is a definitive architectural answer, not a PR1 placeholder:
        ``family.py`` deliberately does NOT override ``supports_train_te``,
        so it inherits the ``latent_diffusion`` archetype's ``False``
        default — the ~66.7 GB TE (66,714,780,128 bytes bf16 on disk, not
        the ~48 GB previously stated here — see the definitions'
        text-encoder comment) must never train.
        """
        return []

    # --- Phase 5: Training Loop Hooks ---
    #
    # The INVERTED flow-match contract (concept §5.1, closed oracle against
    # diffusers 0.40.0 `scheduling_minimax_h3.py`: `:170-171` t = 1 − σ,
    # `:225` x_t = t·x₀ + (1−t)·noise, `:273` x̂₀ = x_t + σ·v):
    #     x₀ = x_t + σ·v   ⇒   v = x₀ − noise   (unique; no scale, no sign freedom)
    # Research §3.1: ai-toolkit `t_v = 1.0 − sigma_v`, `return -noise_pred`;
    # diffusion-pipe `t_v = 1.0 − sigma_v`, `-video_out` — the same contract.
    # These three are CLOBBER hooks: `MiniMaxH3Trainer` delegates each one
    # explicitly (ordering rule 1) so `test_autodelegated_family_hook_set_is_
    # exactly_expected` stays byte-identical.

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Draw ONE ``u`` per item, push it through the VIDEO shift and return
        ``t_v = 1 − σ_v`` in ``[0, 1]`` (the audio clock is derived from σ_v
        in the forward pass through ``H3SigmaSchedule`` — never a second draw).

        Family default draw is ``uniform`` — with the definition's shift on top
        it reproduces the inference grid (diffusion-pipe's recommendation,
        research §4); a user-set ``timestep_sampling`` wins.
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler  # noqa: PLC0415

        from .schedule import H3SigmaSchedule, sigma_to_t
        from .settings import resolve_h3_settings

        mode = config.get("timestep_sampling", "uniform")
        u = TimestepSampler.sample(
            mode, batch_size, device, config, latents=latents, progress=progress,
        )
        schedule = H3SigmaSchedule.from_settings(resolve_h3_settings(self.definition, config))
        sigma_v, _sigma_a = schedule.draw(u)
        return sigma_to_t(sigma_v).clamp(0.0, 1.0)

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """``x_t = t·x₀ + σ·noise`` with ``σ = 1 − t`` — H3's clock (``t = 1``
        clean), identical to ``MiniMaxH3Scheduler.scale_noise``. Written with
        the noise weight from ``schedule.t_to_sigma`` so the family keeps ONE
        conversion site and both endpoints are exact."""
        from .schedule import t_to_sigma

        t = timesteps.to(device=latents.device, dtype=latents.dtype)
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * latents + t_to_sigma(t) * noise

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """The data-ward velocity ``v = x₀ − noise`` — the unique target the
        reference scheduler's ``step`` inverts. The house default
        (``noise − latents``) is the OPPOSITE sign on this family."""
        return latents - noise

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise _lands_in_pr1(
            "the joint audio+video forward pass (packed [text | conditions "
            "| audio | video] sequence, inverted flow-match contract)"
        )

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        raise _lands_in_pr1("the LoRA saver")

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Return the definition's block topology verbatim.

        52 blocks total: 50 main ``transformer_blocks`` + 2 NESTED
        ``token_refiner.refiner_blocks`` (a bare ``refiner_blocks`` attr_path
        does not resolve — see the YAML's comment). Read from the definition
        rather than introspecting a loaded model: PR0 never loads real
        weights, and the definition is authoritative regardless.
        """
        return list(self.definition.block_topology)
