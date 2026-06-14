"""WAN 2.1 driver — subclasses the shared :class:`WanDriverBase`.

WAN 2.1 reuses all of the shared flow-match / forward logic. The only family
specifics are the saver (ComfyUI ``diffusion_model.blocks.*`` keys) and a small
block-topology hook for the VRAM-management UI. The I2V conditioning path lives
in the base ``forward_pass`` and triggers when the definition's ``mode`` is
``i2v``.
"""

from __future__ import annotations

from typing import Any

from app.engine.models.families.wan_shared.driver_base import WanDriverBase


class Wan21Driver(WanDriverBase):
    """WAN 2.1 family driver (T2V 1.3B/14B, I2V 14B)."""

    def get_saver(self) -> Any:
        from app.engine.models.families.wan21.saver import Wan21Saver

        return Wan21Saver(mode=self.mode)

    def get_block_topology(self) -> list[dict[str, Any]]:
        """WAN 2.1 block topology: a single stack of ``blocks``."""
        topology: list[dict[str, Any]] = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "blocks", None)
            if blocks is not None:
                topology.append(
                    {
                        "name": "blocks",
                        "attr_path": "blocks",
                        "count": len(blocks),
                        "approx_vram_mb": 320,
                    }
                )
        return topology
