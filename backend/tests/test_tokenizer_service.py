# backend/tests/test_tokenizer_service.py
"""Unit tests for TokenizerService — cutoff math + heuristic, no network."""

from app.engine.core.caption_target import CaptionTarget
from app.core.captioning.tokenizer_service import TokenizerService


class _FakeEncoding(dict):
    pass


class _FakeTokenizer:
    """Mimics a HF fast tokenizer: one token per whitespace-separated word,
    with char offsets, when called with return_offsets_mapping=True."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = []
        offsets = []
        pos = 0
        for word in text.split(" "):
            if word == "":
                pos += 1
                continue
            start = text.index(word, pos)
            end = start + len(word)
            ids.append(1)
            offsets.append((start, end))
            pos = end
        enc = _FakeEncoding()
        enc["input_ids"] = ids
        if return_offsets_mapping:
            enc["offset_mapping"] = offsets
        return enc


def _target(kind, limit):
    return CaptionTarget(
        family="test",
        tokenizer_kind=kind,
        tokenizer_id=("fake/tok" if kind != "heuristic" else None),
        raw_max_length=limit,
        usable_limit=limit,
    )


def test_count_under_limit_no_cutoff():
    svc = TokenizerService()
    svc._cache["fake/tok"] = _FakeTokenizer()
    tokens, cutoff = svc.count_with_cutoff("one two three", _target("t5", 5))
    assert tokens == 3
    assert cutoff is None


def test_count_over_limit_returns_cutoff_char_index():
    svc = TokenizerService()
    svc._cache["fake/tok"] = _FakeTokenizer()
    # "aa bb cc dd" -> 4 tokens, limit 2 -> cutoff at end of 2nd token ("bb").
    tokens, cutoff = svc.count_with_cutoff("aa bb cc dd", _target("t5", 2))
    assert tokens == 4
    assert cutoff == 5  # index after "aa bb"


def test_empty_text_is_zero_tokens():
    svc = TokenizerService()
    svc._cache["fake/tok"] = _FakeTokenizer()
    tokens, cutoff = svc.count_with_cutoff("", _target("clip", 3))
    assert tokens == 0
    assert cutoff is None


def test_heuristic_branch_needs_no_tokenizer():
    svc = TokenizerService()
    # ~4 chars/token; "abcdefgh" (8 chars) -> 2 tokens, limit 1 -> cutoff at char 4.
    tokens, cutoff = svc.count_with_cutoff("abcdefgh", _target("heuristic", 1))
    assert tokens == 2
    assert cutoff == 4
