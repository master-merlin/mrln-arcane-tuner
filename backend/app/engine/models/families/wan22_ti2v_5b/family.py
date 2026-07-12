"""WAN 2.2 TI2V-5B model family registration (dense, single transformer).

``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` is the DENSE 5B member of the WAN 2.2 line:
ONE ``WanTransformer3DModel`` (no high/low expert pair, no ``ExpertRouter``, no
deferred second-expert loading — none of the ``wan22`` A14B MoE machinery
applies here) paired with a NEW higher-compression ``AutoencoderKLWan`` (4x
temporal / 16x spatial, ``z_dim=48`` vs the A14B/2.1 VAE's 4x/8x, ``z_dim=16``).

A single checkpoint serves BOTH text-to-video and image-to-video generation
(``architecture_params.mode: both``) — unlike ``wan21``/``wan22`` (a fixed
``t2v``/``i2v`` definition per checkpoint), the actual mode is chosen PER RUN
via the job's ``video_mode`` config field, exactly the ``ltx2`` precedent
(single ``mode: both`` definition, per-step i2v gate in the trainer).

Deliberately lives in its OWN family package (not ``wan22``, whose
``family_name`` a second directory could not safely reuse — the registry
overwrites on a duplicate ``family_name`` — and not ``wan21``, whose driver
has no notion of a runtime-toggled ``mode``). See ``.agent/workdir/a5-report.md``
for the full recon + ownership rationale. This keeps the change a pure
ADDITION: zero lines touched in ``wan21``/``wan22``.

``capability_overrides`` deliberately OMITS ``dual_expert`` (defaults False on
the archetype), which — via ``core/archetypes.py``'s field-visibility table —
automatically hides ``expert_mode``/``expert_swap_mode``/``expert_switch_interval``
for this family with no extra code (the same seam the A14B MoE UI relies on).
"""

from app.engine.core.definitions import ModelFamily


class Wan22Ti2v5bFamily(ModelFamily):
    """WAN 2.2 TI2V-5B (dense, T2V+I2V) logic provider."""

    family_name = "wan22_ti2v_5b"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``core/archetypes.py``.
    #  - is_video:           batches are 5D [B, C, F, H, W] video clips.
    #  - has_image_encoder:  False — no CLIP-vision tower (``image_dim: null``
    #    in the real checkpoint); I2V conditions via the transformer's native
    #    ``expand_timesteps`` first-frame-clean-token scheme instead.
    #  - dual_expert:        NOT set (inherits the archetype default False) —
    #    this is what hides the MoE-only fields for this family.
    #  - native_fps:         24 (Wan2.2 TI2V-5B model-card default; informational
    #    only — ``video_contract`` derives the real fps gate from
    #    ``architecture_params['video.native_fps']``, set in the YAML).
    capability_overrides = {
        "is_video": True,
        "has_image_encoder": False,
        "native_fps": 24,
    }

    def get_trainer_class(self):
        """Single dense trainer — no expert-mode branching (unlike wan22)."""
        from .trainer import Wan22Ti2v5bTrainer

        return Wan22Ti2v5bTrainer
