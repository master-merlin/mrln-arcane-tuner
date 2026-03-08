"""Block swapping — CPU↔GPU migration of transformer blocks via forward hooks.

Reduces peak VRAM by keeping only the actively-executing transformer block
on GPU.  All other blocks reside on CPU until their forward pass fires.

Usage::

    from app.engine.core.optimization.block_swapping import BlockSwappingManager

    manager = BlockSwappingManager(blocks, device=torch.device("cuda"))
    manager.apply()          # Register hooks
    # ... training loop ...
    manager.remove()         # Clean up hooks
"""

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)


class BlockSwappingManager:
    """Swap transformer blocks between CPU and GPU via forward hooks.

    Maintains a dictionary of pinned CPU tensors for zero-allocation
    swapping. Before each block's forward pass, its parameters are
    copied to GPU. After, they are copied back to the pinned CPU buffer.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
    ) -> None:
        self.blocks = blocks
        self.execution_device = device
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._cpu = torch.device("cpu")
        self._cpu_shadow_map: dict[torch.Tensor, torch.Tensor] = {}

    def _get_or_create_pinned_shadow(self, tensor: torch.Tensor) -> torch.Tensor:
        """Get the pinned CPU shadow buffer for a tensor, creating it if needed."""
        if tensor not in self._cpu_shadow_map:
            if tensor.device.type == "cpu":
                pinned = tensor.pin_memory()
            else:
                pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
                pinned.copy_(tensor, non_blocking=True)
            self._cpu_shadow_map[tensor] = pinned
        return self._cpu_shadow_map[tensor]

    def _move_block_to_cpu(self, module: nn.Module) -> None:
        """Copy block state from GPU to pinned CPU shadow, and re-link ``.data``."""
        for param in getattr(module, "_swappable_params", []):
            cpu_tensor = self._get_or_create_pinned_shadow(param)
            if param.data.device.type != "cpu":
                cpu_tensor.copy_(param.data, non_blocking=True)
                param.data = cpu_tensor

            if param.grad is not None:
                # We dynamically track .grad because it appears mid-training
                if not hasattr(param, "_cpu_grad_shadow"):
                    param._cpu_grad_shadow = torch.empty_like(param.grad.data, device="cpu", pin_memory=True)
                cpu_grad = param._cpu_grad_shadow
                if param.grad.data.device.type != "cpu":
                    cpu_grad.copy_(param.grad.data, non_blocking=True)
                    param.grad.data = cpu_grad

        for buf in getattr(module, "_swappable_buffers", []):
            cpu_tensor = self._get_or_create_pinned_shadow(buf)
            if buf.data.device.type != "cpu":
                cpu_tensor.copy_(buf.data, non_blocking=True)
                buf.data = cpu_tensor

    def _move_block_to_gpu(self, module: nn.Module) -> None:
        """Copy block state from pinned CPU shadow to GPU, and re-link ``.data``."""
        for param in getattr(module, "_swappable_params", []):
            if param.data.device != self.execution_device:
                # Let PyTorch's native CUDA allocator handle the GPU buffer
                # Since we copy from pinned memory, non_blocking=True triggers DMA async transfer
                param.data = param.data.to(self.execution_device, non_blocking=True)
                
            if param.grad is not None and param.grad.data.device != self.execution_device:
                param.grad.data = param.grad.data.to(self.execution_device, non_blocking=True)

        for buf in getattr(module, "_swappable_buffers", []):
            if buf.data.device != self.execution_device:
                buf.data = buf.data.to(self.execution_device, non_blocking=True)

    # ── Public API ───────────────────────────────────────────────────────

    def apply(self) -> None:
        """Initialize shadow buffers and register pre/post forward hooks."""
        logger.info("block_swapping_apply", num_blocks=len(self.blocks))

        for block in self.blocks:
            # Cache the list of params and buffers to avoid iterating over generator constantly
            block._swappable_params = list(block.parameters())
            block._swappable_buffers = list(block.buffers())

            # Seed the shadow map and physically move the block to CPU (pinned)
            self._move_block_to_cpu(block)

            self._hooks.extend([
                block.register_forward_pre_hook(self._pre_forward_hook),
                block.register_forward_hook(self._post_forward_hook),
            ])

        # Force the CUDA allocator to release the base memory to the OS
        # since we've now permanently staged everything to pinned CPU RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "block_swapping_offloaded",
            vram_allocated_mb=round(torch.cuda.memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0,
        )

    def remove(self) -> None:
        """Remove all hooks and clean up shadow maps."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
        # Clean up shadow references so memory can be GC'd
        self._cpu_shadow_map.clear()
        
        for block in self.blocks:
            if hasattr(block, "_swappable_params"):
                del block._swappable_params
            if hasattr(block, "_swappable_buffers"):
                del block._swappable_buffers

        logger.info("block_swapping_removed")

    # ── Hook Callbacks ───────────────────────────────────────────────────

    def _pre_forward_hook(self, module: nn.Module, args: tuple) -> None:
        self._move_block_to_gpu(module)

    def _post_forward_hook(self, module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        self._move_block_to_cpu(module)
