"""Block swapping: a swapped block's frozen weights hold card memory only while the block works.

Every param keeps its device; the manager moves memory, not tensors. Moving a param's ``.data``
to the host broke training three ways: a LoRA param went to the host while its grad stayed on the
card, so the optimizer step mixed devices; a frozen weight autograd saved for backward (a
LayerNorm's, a multiply's) was read from the host; and a ``Linear``'s saved ``weight.t()`` view kept
the card storage alive, so training saved almost nothing.

- **Trainable params** are never touched, so a param's grad and optimizer state stay beside it.
  A param that trains at any point, even one frozen again later (an adapter that targeted training
  or adaptive targeting turns off), stays resident: the LoRA save and EMA read it.
- **Params frozen when the swap is applied** are *seated*: each storage under them (tensor
  subclasses are walked through ``__tensor_flatten__``) gets one pinned host copy of its bytes,
  and its card memory is then freed in place (``untyped_storage().resize_(0)``). The tensor, and
  every view autograd saved of it, keeps its device and shape; only the bytes come and go.
  A block still on the host when the swap is applied is first moved to the card.
- **A storage something else owns is never seated**: one a trainable param or a buffer reaches,
  one two swapped blocks share, or one a param outside the swapped blocks reaches (``root``).
- **Before the block runs** the storages are refilled from the pinned copies. Frozen weights
  never change, so nothing is copied back.
- **After the block's forward** the storages are freed, also when the forward raised.
- **Backward**, the way FSDP does it: a hook on the block's outputs refills it before its own
  backward, and an identity function on its inputs frees it once its inputs' grads are made, so
  its backward is done. A block with no input that needs a grad is freed by an end-of-backward
  callback, as is anything a failed backward left.
- **A gradient-checkpoint recompute** runs inside backward. Its pre-hook refills the block and its
  post-hook, which runs even when a non-reentrant recompute stops early by raising, marks it
  pending. The reentrant kind's inputs get the identity function as well; the non-reentrant
  kind's block is freed by the function from the original forward.
- **Buffers** stay resident: they are small, and a forward may change one.
- **``.to()``** on a swapped block refills it, applies, and seats it again; a block moved off the
  card is whole there.
- **On a CPU execution device** nothing is freed.

Usage::

    manager = BlockSwappingManager(blocks, device=torch.device("cuda"), root=model)
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
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

logger = structlog.get_logger(__name__)


def _leaves(tensor: torch.Tensor) -> Iterator[torch.Tensor]:
    """The plain tensors under ``tensor``: itself, or a subclass's inner tensors, recursively."""
    if is_traceable_wrapper_subclass(tensor):
        names, _ = tensor.__tensor_flatten__()
        for name in names:
            yield from _leaves(getattr(tensor, name))
    else:
        yield tensor


def _keys(tensor: torch.Tensor) -> Iterator[int]:
    """Identities of the storages under ``tensor``."""
    for leaf in _leaves(tensor):
        yield leaf.untyped_storage()._cdata


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


class _FreeAfterBackward(torch.autograd.Function):
    """Identity on a block's inputs; its backward runs once their grads are made: the block is done."""

    @staticmethod
    def forward(ctx: Any, manager: BlockSwappingManager, block: nn.Module, *inputs: torch.Tensor) -> Any:
        ctx.manager, ctx.block = manager, block
        return inputs

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> Any:
        ctx.manager._after_backward(ctx.block)
        return (None, None, *grads)


class BlockSwappingManager:
    """Frees the card memory of swapped blocks' frozen weights between the times each block runs."""

    def __init__(self, blocks: list[nn.Module], device: torch.device, root: nn.Module | None = None) -> None:
        self.blocks = list(blocks)
        self.execution_device = _normalized(device)
        self.root = root
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # block -> [(storage, pinned host copy of its bytes)]
        self._seated: dict[nn.Module, list[tuple[torch.UntypedStorage, torch.Tensor]]] = {}
        self._signature: dict[nn.Module, tuple] = {}
        # Params frozen when the swap was applied, and never trainable since: the only ones seated.
        self._never_trained: set[int] = set()
        self._pending: list[nn.Module] = []
        self._end_queued_for: int | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def apply(self) -> None:
        """Seat every block and register its hooks."""
        logger.info("block_swapping_apply", num_blocks=len(self.blocks))
        self._never_trained = {id(p) for block in self.blocks for p in block.parameters() if not p.requires_grad}
        for block in self.blocks:
            # A block still on the host is moved and seated before the next one moves, so the card
            # holds one extra block at most while the swap is applied.
            if self._frees and any(t.device != self.execution_device for t in [*block.parameters(), *block.buffers()]):
                block.to(self.execution_device)
            self._seat(block)
            self._wrap_apply(block)
            self._hooks.extend([
                block.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=True),
                block.register_forward_hook(self._post_forward_hook, always_call=True),
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

    def _seatable(self, block: nn.Module) -> Iterator[nn.Parameter]:
        """Params frozen when the swap was applied, never trainable since, on the execution device."""
        for param in block.parameters():
            if param.requires_grad:
                self._never_trained.discard(id(param))
            elif id(param) in self._never_trained and param.device == self.execution_device:
                yield param

    def _owned_elsewhere(self, block: nn.Module) -> set[int]:
        """Storages ``block`` must leave whole: another owner reads them outside its forward.

        Any storage a trainable param or a buffer reaches, anywhere; any another swapped block
        reaches; and any a param of ``root`` outside the swapped blocks reaches.
        """
        # Who holds each param, by every path it is reached on: the swapped block the path runs
        # through, or "outside" for a path of root through none. A child module shared by two
        # blocks, or by a swapped and an unswapped one, is held on two paths.
        outside = -1
        holders: dict[int, set[int]] = {}
        params: dict[int, nn.Parameter] = {}
        owned: set[int] = set()

        def walk(top: nn.Module, default: int) -> None:
            swapped = {id(b) for b in self.blocks}
            enclosing: dict[str, int] = {}  # path of a swapped block -> its id
            for path, module in top.named_modules(remove_duplicate=False):
                if id(module) in swapped:
                    enclosing[path] = id(module)
                holder = default
                for prefix, block_id in enclosing.items():
                    if path == prefix or path.startswith(prefix + ".") or prefix == "":
                        holder = block_id
                        break
                for param in module._parameters.values():
                    if param is not None:
                        holders.setdefault(id(param), set()).add(holder)
                        params[id(param)] = param
                for buffer in module._buffers.values():
                    if buffer is not None:
                        owned.update(_keys(buffer))

        for b in self.blocks:
            walk(b, id(b))
        if self.root is not None:
            walk(self.root, outside)
        for key, param in params.items():
            # Held by anything but this block alone, or trainable: someone reads it while it is freed.
            if param.requires_grad or holders[key] != {id(block)}:
                owned.update(_keys(param.data))
        return owned

    def _block_signature(self, block: nn.Module) -> tuple:
        """What seating depends on: which storages each param has, where, and whether it trains."""
        for param in block.parameters():
            if param.requires_grad:
                self._never_trained.discard(id(param))
        return tuple((id(p), p.requires_grad, p.device, tuple(_keys(p.data))) for p in block.parameters())

    def _seat(self, block: nn.Module) -> None:
        """Pin a host copy of each frozen storage on the device, then free it. The block must be filled."""
        entries: list[tuple[torch.UntypedStorage, torch.Tensor]] = []
        previous = {st._cdata: pinned for st, pinned in self._seated.get(block, [])}
        if self._frees:
            owned = self._owned_elsewhere(block)
            seen: set[int] = set()
            for param in self._seatable(block):
                for leaf in _leaves(param.data):
                    storage = leaf.untyped_storage()
                    key = storage._cdata
                    if key in seen or key in owned or storage.nbytes() == 0 or not storage.resizable():
                        continue
                    seen.add(key)
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

    def _keep_until_done(self, block: nn.Module) -> None:
        """Inside backward: ``block`` stays filled until its backward is done or backward ends."""
        if block not in self._pending:
            self._pending.append(block)
        task = torch._C._current_graph_task_id()
        if self._end_queued_for != task:
            torch.autograd.Variable._execution_engine.queue_callback(self._end_of_backward)
            self._end_queued_for = task

    def _end_of_backward(self) -> None:
        self._end_queued_for = None
        while self._pending:
            self._free(self._pending.pop())

    def _reload(self, block: nn.Module) -> None:
        """One of the block's outputs got its grad: its own backward may come next."""
        self._keep_until_done(block)
        self._fill(block)

    def _after_backward(self, block: nn.Module) -> None:
        """The grads of the block's inputs are made: nothing in its backward reads its weights again."""
        if block in self._pending:
            self._pending.remove(block)
        self._free(block)

    def _free_after_backward(self, block: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        flat, spec = tree_flatten((args, kwargs))
        where = [i for i, t in enumerate(flat) if torch.is_tensor(t) and t.requires_grad]
        if not where or not self._seated.get(block):
            return args, kwargs
        wrapped = _FreeAfterBackward.apply(self, block, *(flat[i] for i in where))
        for i, tensor in zip(where, wrapped, strict=True):
            flat[i] = tensor
        return tree_unflatten(flat, spec)

    # ── Hook Callbacks ───────────────────────────────────────────────────

    def _pre_forward_hook(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
        self._reseat_if_changed(module)
        self._fill(module)
        if torch.is_grad_enabled():
            return self._free_after_backward(module, args, kwargs)
        return None

    def _post_forward_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        if _in_backward():
            # A recompute (a non-reentrant one stops early by raising): its backward is next.
            self._keep_until_done(module)
            return
        if output is not None and self._seated.get(module):
            # Made in this forward: a hook on a leaf (a tensor the block hands back without making
            # it) would outlive the graph and pile up, one more at every step.
            outputs = [
                t for t in tree_leaves(output) if torch.is_tensor(t) and t.requires_grad and t.grad_fn is not None
            ]
            if outputs:
                torch.autograd.graph.register_multi_grad_hook(
                    outputs, lambda _grads, block=module: self._reload(block), mode="any"
                )
        self._free(module)
