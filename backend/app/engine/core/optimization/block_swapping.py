"""Block swapping: a swapped block's frozen weights hold card memory only while the block works.

Every param keeps its device; the manager moves memory, not tensors. Moving a param's ``.data``
to the host broke training three ways: a LoRA param went to the host while its grad stayed on the
card, so the optimizer step mixed devices; a frozen weight autograd saved for backward (a
LayerNorm's, a multiply's) was read from the host; and a ``Linear``'s saved ``weight.t()`` view kept
the card storage alive, so training saved almost nothing.

- **Trainable params** are never touched, so a param's grad and optimizer state stay beside it.
- **Frozen params on the execution device** are *seated*: each storage under them (tensor
  subclasses are walked through ``__tensor_flatten__``) gets one pinned host copy of its bytes,
  and its card memory is then freed in place (``untyped_storage().resize_(0)``). The tensor, and
  every view autograd saved of it, keeps its device and shape; only the bytes come and go.
- **Before the block runs** the storages are refilled from the pinned copies. Frozen weights
  never change, so nothing is copied back.
- **After the block's forward** the storages are freed. With grad, a hook on the block's outputs
  refills the block just before its own backward and frees the block refilled before it; an
  end-of-backward callback frees the last one.
- **A gradient-checkpoint recompute** runs inside backward. Its pre-hook frees the blocks backward
  has passed and keeps this one, as a non-reentrant recompute stops early and skips its post-hook.
- **Buffers** stay resident: they are small, and a forward may change one.
- **``.to()``** on a swapped block refills it, applies, and seats it again; a block moved off the
  card is whole there.
- **On a CPU execution device** nothing is freed.

Usage::

    manager = BlockSwappingManager(blocks, device=torch.device("cuda"))
    manager.apply()          # seat the blocks and register the hooks
    # ... training loop ...
    manager.remove()         # refill, and remove the hooks
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._pytree import tree_leaves

logger = structlog.get_logger(__name__)


def _leaves(tensor: torch.Tensor) -> Iterator[torch.Tensor]:
    """The plain tensors under ``tensor``: itself, or a subclass's inner tensors, recursively."""
    if is_traceable_wrapper_subclass(tensor):
        names, _ = tensor.__tensor_flatten__()
        for name in names:
            yield from _leaves(getattr(tensor, name))
    else:
        yield tensor


def _bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """A uint8 tensor over ``storage``'s bytes."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def _in_backward() -> bool:
    return torch._C._current_graph_task_id() != -1


def _normalized(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


class BlockSwappingManager:
    """Frees the card memory of swapped blocks' frozen weights between the times each block runs."""

    def __init__(self, blocks: list[nn.Module], device: torch.device) -> None:
        self.blocks = list(blocks)
        self.execution_device = _normalized(device)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # block -> [(storage, pinned host copy of its bytes)]
        self._seated: dict[nn.Module, list[tuple[torch.UntypedStorage, torch.Tensor]]] = {}
        self._signature: dict[nn.Module, tuple] = {}
        self._shared: set[int] = set()
        self._pending: list[nn.Module] = []
        self._end_queued_for: int | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def apply(self) -> None:
        """Seat every block and register its hooks."""
        logger.info("block_swapping_apply", num_blocks=len(self.blocks))
        self._shared = self._storages_in_more_than_one_block()
        for block in self.blocks:
            self._seat(block)
            self._wrap_apply(block)
            self._hooks.extend([
                block.register_forward_pre_hook(self._pre_forward_hook),
                block.register_forward_hook(self._post_forward_hook),
            ])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "block_swapping_offloaded",
            seated_mb=round(self.seated_bytes() / 2**20),
            vram_allocated_mb=round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
        )

    def remove(self) -> None:
        """Refill every block, restore its ``_apply``, and remove the hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for block in self.blocks:
            self._fill(block)
            if "_apply" in block.__dict__:
                del block.__dict__["_apply"]
        self._seated.clear()
        self._signature.clear()
        self._pending.clear()
        logger.info("block_swapping_removed")

    def seated_bytes(self) -> int:
        """Bytes of frozen weights the manager frees between runs, over all its blocks."""
        return sum(pinned.numel() for entries in self._seated.values() for _, pinned in entries)

    def filled_blocks(self) -> list[nn.Module]:
        """The swapped blocks whose seated weights hold card memory now."""
        return [b for b, entries in self._seated.items() if any(st.nbytes() for st, _ in entries)]

    # ── Seating, filling, freeing ────────────────────────────────────────

    @property
    def _frees(self) -> bool:
        return self.execution_device.type != "cpu"

    def _frozen_on_device(self, block: nn.Module) -> Iterator[nn.Parameter]:
        for param in block.parameters():
            if not param.requires_grad and param.device == self.execution_device:
                yield param

    def _storages_in_more_than_one_block(self) -> set[int]:
        """Storages two swapped blocks share: freeing one block's would break the other's."""
        owner: dict[int, nn.Module] = {}
        shared: set[int] = set()
        for block in self.blocks:
            for param in block.parameters():
                for leaf in _leaves(param.data):
                    key = leaf.untyped_storage()._cdata
                    if owner.setdefault(key, block) is not block:
                        shared.add(key)
        return shared

    def _block_signature(self, block: nn.Module) -> tuple:
        """What seating depends on: which storages each param has, where, and whether it trains."""
        return tuple(
            (id(p), p.requires_grad, p.device, tuple(leaf.untyped_storage()._cdata for leaf in _leaves(p.data)))
            for p in block.parameters()
        )

    def _seat(self, block: nn.Module) -> None:
        """Pin a host copy of each frozen storage on the device, then free it. The block must be filled."""
        entries: list[tuple[torch.UntypedStorage, torch.Tensor]] = []
        previous = {st._cdata: pinned for st, pinned in self._seated.get(block, [])}
        if self._frees:
            seen: set[int] = set()
            for param in self._frozen_on_device(block):
                for leaf in _leaves(param.data):
                    storage = leaf.untyped_storage()
                    key = storage._cdata
                    if key in seen or key in self._shared or storage.nbytes() == 0:
                        continue
                    seen.add(key)
                    if not storage.resizable():
                        continue
                    pinned = previous.get(key)
                    if pinned is None or pinned.numel() != storage.nbytes():
                        pinned = torch.empty(storage.nbytes(), dtype=torch.uint8, pin_memory=True)
                        pinned.copy_(_bytes(storage))
                    entries.append((storage, pinned))
        self._seated[block] = entries
        self._signature[block] = self._block_signature(block)
        self._free(block)

    def _free(self, block: nn.Module) -> None:
        for storage, _ in self._seated.get(block, ()):
            storage.resize_(0)

    def _fill(self, block: nn.Module) -> None:
        for storage, pinned in self._seated.get(block, ()):
            if storage.nbytes() == 0:
                storage.resize_(pinned.numel())
                _bytes(storage).copy_(pinned, non_blocking=True)

    def _reseat_if_changed(self, block: nn.Module) -> None:
        """A param frozen or unfrozen, or given new data, since the block was seated: seat it again."""
        if self._block_signature(block) != self._signature.get(block):
            self._fill(block)
            self._seat(block)

    def _wrap_apply(self, block: nn.Module) -> None:
        """``.to()``, ``.float()`` and friends see the block whole, and seat it again afterwards."""
        original = type(block)._apply.__get__(block)

        def _apply(fn: Any, recurse: bool = True) -> nn.Module:
            self._fill(block)
            try:
                return original(fn, recurse)
            finally:
                self._seat(block)

        block.__dict__["_apply"] = _apply

    # ── Backward bookkeeping ─────────────────────────────────────────────

    def _queue_end(self) -> None:
        task = torch._C._current_graph_task_id()
        if self._end_queued_for != task:
            torch.autograd.Variable._execution_engine.queue_callback(self._end_of_backward)
            self._end_queued_for = task

    def _end_of_backward(self) -> None:
        self._end_queued_for = None
        while self._pending:
            self._free(self._pending.pop())

    def _retain(self, block: nn.Module) -> None:
        """Inside backward: free the blocks backward is done with, keep ``block`` until the next one."""
        while self._pending:
            previous = self._pending.pop()
            if previous is not block:
                self._free(previous)
        self._pending.append(block)
        self._queue_end()

    def _reload(self, block: nn.Module) -> None:
        """The block's outputs got their grads: its own backward is next."""
        self._retain(block)
        self._fill(block)

    # ── Hook Callbacks ───────────────────────────────────────────────────

    def _pre_forward_hook(self, module: nn.Module, args: tuple) -> None:
        self._reseat_if_changed(module)
        if _in_backward():
            self._retain(module)  # a checkpoint recompute
        self._fill(module)

    def _post_forward_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        if _in_backward():
            # A recompute: the block's own backward runs next, so it stays filled until then.
            if module not in self._pending:
                self._pending.append(module)
            self._queue_end()
            return
        if self._seated.get(module):
            outputs = [t for t in tree_leaves(output) if torch.is_tensor(t) and t.requires_grad]
            if outputs:
                torch.autograd.graph.register_multi_grad_hook(
                    outputs, lambda _grads, block=module: self._reload(block), mode="any"
                )
        self._free(module)
