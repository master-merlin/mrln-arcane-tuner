"""DreamLite trainer text-cache tests — disk-cache key template identity (W3.T8).

Pins the fix: ``TextEmbeddingCache.caption_to_filename`` hashes ONLY the
string it is given. Before this task, that string was the (optionally
``"[Generate]: "``-prefixed) caption alone — the pinned chat template +
``drop_idx`` that ``DreamLiteDriver.encode_text`` wraps every caption in
never reached the hashed key. A future template/drop-idx edit would have
silently served embeddings encoded under the OLD template (the same
poisoned-cache incident class the qwen_image/boogu_image families already
guard against). ``DreamLiteTrainer._disk_cache_key`` now bakes
``te_template_fingerprint()`` (driver.py) into the hashed string.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

_L, _D = 6, 8  # tiny embedding dims


class _FakeDriver:
    """Driver stand-in: counts calls, returns deterministic tiny tensors."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode_text(self, texts, dtype):
        self.encoded.extend(texts)
        b = len(texts)
        emb = torch.stack(
            [torch.full((_L, _D), float(len(t))) for t in texts],
        ).to(dtype)
        mask = torch.ones(b, _L, dtype=torch.int64)
        return SimpleNamespace(embeddings=emb, attention_mask=mask)


def _trainer(
    tmp_path=None,
    captions: dict[str, str] | None = None,
) -> DreamLiteTrainer:
    t = object.__new__(DreamLiteTrainer)
    t.device = torch.device("cpu")
    t.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "mixed_precision": "bf16",
        "sample_prompts": [],
        "sample_negative_prompt": "",
    }
    t.text_cache = {}
    t.text_encoder = object()  # "resident" marker (not None)
    t.driver = _FakeDriver()
    t.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    t._log_writer = None
    t._resolve_te_cache_dirs = lambda: [str(tmp_path)] if tmp_path else []
    t._build_caption_hints = lambda: dict(captions or {})
    t._resolve_loading_dtype = lambda: torch.float32
    return t


# ── Unit: the disk-cache-key template fingerprint ──────────────────────────


def test_disk_cache_key_changes_when_chat_template_changes(monkeypatch):
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    original = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")
    monkeypatch.setattr(
        dreamlite_driver,
        "DREAMLITE_PROMPT_TEMPLATE",
        "<|im_start|>system\nCHANGED<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    )
    changed = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_changes_when_drop_idx_changes(monkeypatch):
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    original = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")
    monkeypatch.setattr(dreamlite_driver, "_DEFAULT_DROP_IDX", 99)
    changed = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_changes_when_max_sequence_length_changes(monkeypatch):
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    original = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")
    monkeypatch.setattr(dreamlite_driver, "_DEFAULT_MAX_SEQUENCE_LENGTH", 99)
    changed = DreamLiteTrainer._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_stable_for_the_same_template():
    assert DreamLiteTrainer._disk_cache_key(
        "[Generate]: a cat",
    ) == DreamLiteTrainer._disk_cache_key("[Generate]: a cat")


# ── Integration: a template edit invalidates the on-disk cache ─────────────


def test_template_change_invalidates_disk_cache(tmp_path, monkeypatch):
    """End-to-end: a template edit must force a re-encode, not a stale hit."""
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    cold = _trainer(tmp_path, captions={"a cat": ""})
    cold._pre_cache_text_embeddings()
    assert cold.driver.encoded == ["[Generate]: a cat"]

    # Simulate a future template edit.
    monkeypatch.setattr(
        dreamlite_driver,
        "DREAMLITE_PROMPT_TEMPLATE",
        "<|im_start|>system\nCHANGED<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    )

    warm = _trainer(tmp_path, captions={"a cat": ""})
    warm._pre_cache_text_embeddings()

    # Re-encoded — NOT silently served from the disk entry written under
    # the old template.
    assert warm.driver.encoded == ["[Generate]: a cat"]
