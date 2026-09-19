"""MiniMax-H3 Trainer — PR0 SCAFFOLD STUB.

Real training (loader, data path, packed-sequence forward, the inverted
flow-match convention, joint audio+video loss) lands in PR1. This stub
exists so ``MiniMaxH3Family.get_trainer_class()`` returns a real class
instead of raising — the registry-wide guard
(``tests/engine/test_hook_wiring_meta.py::test_every_family_resolves_a_trainer_and_driver``)
requires every registered family to resolve BOTH a trainer and a driver, and
a family whose trainer resolution raises is indistinguishable from a broken
one.

``_setup_family`` is the only method the base ``GenericTrainingPipeline``
requires a subclass to implement, and it is the first family hook the real
pipeline calls (via ``setup()``). Wiring ``self.driver = MiniMaxH3Driver(...)``
here — even though the next line always raises — keeps the trainer→driver
seam real and source-greppable (see
``tests/engine/test_hook_wiring_meta.py::_driver_for_trainer``, which
resolves a family's driver by regex-searching the trainer's MRO source for
exactly this assignment shape) instead of a fabricated shortcut. A job that
somehow reaches this trainer must fail loudly here, not limp through with a
missing loader/data path.
"""

from __future__ import annotations

import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import MiniMaxH3Driver


class MiniMaxH3Trainer(GenericTrainingPipeline):
    """MiniMax-H3 LoRA trainer — PR0 STUB. Real training lands in PR1."""

    # ── Explicit delegations of the driver's CLOBBER hooks (plan row 1.2,
    # ordering rule 1). The base pipeline WOULD auto-delegate these, but an
    # explicit method keeps the family out of the reviewed auto-delegation
    # allowlist (`test_autodelegated_family_hook_set_is_exactly_expected`)
    # and makes the seam greppable.

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return self.driver.add_noise(latents, noise, timesteps)

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return self.driver.compute_target(latents, noise, timesteps)

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        return self.driver.sample_timesteps(
            batch_size, self.device, self.config, latents=latents, progress=progress
        )

    def _setup_family(self) -> None:
        # Real assignment, so the trainer→driver seam is genuine (see the
        # module docstring): the resolution guard's regex finds this line,
        # not a decoy. No loader exists yet (PR1 scope), so setup stops here.
        self.driver = MiniMaxH3Driver(self.definition, self.device)
        raise NotImplementedError(
            "minimax_h3 training lands in PR1; PR0 ships the scaffold only."
        )
