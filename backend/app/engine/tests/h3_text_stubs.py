"""Offline stand-ins for the MiniMax-H3 text path (plan row 2.1).

No weights, no network. ``StubQwen3VL`` mimics the ONE thing the driver reads
off ``transformers.Qwen3VLForConditionalGeneration``: ``.model(...)`` returning
an object whose ``hidden_states`` is a tuple of ``num_hidden_layers + 1``
tensors, where entry ``i`` is filled with the value ``i`` — so the tap index
the driver used is readable straight off the returned embedding (an
observable, not a recorded kwarg). ``StubTokenizer`` tokenizes by whitespace
so token counts depend on the caption and the 512-token trim is testable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


class StubTokenizer:
    """Whitespace tokenizer with the two surfaces the driver touches."""

    pad_token_id = 0

    def __init__(self, vocab_tag: str = "v1") -> None:
        self._vocab_tag = vocab_tag

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any):
        assert add_special_tokens is False, "H3 presents raw tokens, no specials"
        ids = [len(w) + 1 for w in text.split()]
        return {"input_ids": ids}

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "a": 1, self._vocab_tag: 2}


class StubProcessor:
    """``AutoProcessor``-shaped: the tokenizer hangs off ``.tokenizer``."""

    def __init__(self, tokenizer: StubTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or StubTokenizer()


class _Inner:
    """The ``.model`` attribute: callable, counts its calls."""

    def __init__(self, owner: StubQwen3VL) -> None:
        self._owner = owner

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_: Any,
    ):
        assert output_hidden_states, "the tap needs every hidden state"
        self._owner.calls += 1
        self._owner.seen_input_ids.append(input_ids.detach().cpu().clone())
        B, L = input_ids.shape
        n = self._owner.num_hidden_layers + 1
        hs = tuple(
            torch.full((B, L, self._owner.hidden_size), float(i), dtype=torch.float32)
            for i in range(n)
        )
        return SimpleNamespace(hidden_states=hs, last_hidden_state=hs[-1])


class StubQwen3VL(nn.Module):
    """``Qwen3VLForConditionalGeneration``-shaped stub with real parameters
    (so weight-byte accounting and ``.to()`` have something to measure)."""

    def __init__(self, num_hidden_layers: int = 64, hidden_size: int = 8) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.calls = 0
        self.seen_input_ids: list[torch.Tensor] = []
        self.weight = nn.Parameter(torch.zeros(1024, dtype=torch.float32))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
        )
        self.model = _Inner(self)

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - never used
        raise AssertionError("the driver must call .model, not the LM-head wrapper")
