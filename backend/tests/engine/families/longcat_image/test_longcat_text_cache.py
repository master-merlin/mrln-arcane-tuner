"""LongCat-Image trainer text-cache tests — (embedding, mask) tuple cache.

Pins:
- ``_pre_cache_text_embeddings`` warms the configured (non-empty)
  ``sample_negative_prompt`` — and the empty-string default — whenever
  sample prompts are configured, mirroring every other CFG sibling
  (ovis_image, prx, prx_pixel, krea2, chroma, dreamlite, nucleus_image,
  lumina2, omnigen2, boogu_image). Before this fix, longcat was the ONLY
  CFG family whose pre-cache never warmed the negative: default-guidance
  (4.5) previews hit an uncached negative mid-denoise — a VRAM-spiking
  reload in offload mode, a hard ``RuntimeError`` in unload mode.
"""

from types import SimpleNamespace

import torch

from app.engine.models.families.longcat_image.trainer import LongCatImageTrainer

_L, _D = 8, 12  # tiny embedding dims


class _FakeDriver:
    """Driver stand-in: counts calls, returns deterministic tiny tensors."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode_text(self, captions, dtype):
        self.encoded.extend(captions)
        b = len(captions)
        emb = torch.stack([torch.full((_L, _D), float(len(c))) for c in captions]).to(
            dtype
        )
        mask = torch.ones(b, _L, dtype=torch.int64)
        return SimpleNamespace(embeddings=emb, attention_mask=mask)


def _trainer(
    tmp_path=None,
    captions: dict[str, str] | None = None,
    sample_prompts: list | None = None,
    sample_negative_prompt: str = "",
    driver: "_FakeDriver | None" = None,
) -> LongCatImageTrainer:
    t = object.__new__(LongCatImageTrainer)
    t.device = torch.device("cpu")
    t.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "mixed_precision": "bf16",
        "sample_prompts": sample_prompts or [],
        "sample_negative_prompt": sample_negative_prompt,
    }
    t.text_cache = {}
    t.text_encoder = object()  # "resident" marker (not None)
    t.driver = driver if driver is not None else _FakeDriver()
    t.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    t._log_writer = None
    t._resolve_te_cache_dirs = lambda: [str(tmp_path)] if tmp_path else []
    t._build_caption_hints = lambda: dict(captions or {})
    return t


# ── The bug: CFG negative was never warmed ─────────────────────────────────


def test_configured_negative_is_warmed_when_sample_prompts_present(tmp_path):
    t = _trainer(
        tmp_path,
        captions={"train cap": ""},
        sample_prompts=[{"prompt": "a preview prompt"}],
        sample_negative_prompt="ugly",
    )
    t._pre_cache_text_embeddings()

    assert "ugly" in t.text_cache
    assert "a preview prompt" in t.text_cache


def test_default_empty_negative_is_warmed_when_sample_prompts_present(tmp_path):
    t = _trainer(
        tmp_path,
        captions={"train cap": ""},
        sample_prompts=[{"prompt": "a preview prompt"}],
        # sample_negative_prompt left at its default ("")
    )
    t._pre_cache_text_embeddings()

    assert "" in t.text_cache
    assert "a preview prompt" in t.text_cache


def test_no_negative_warm_when_no_sample_prompts_configured(tmp_path):
    """No sampling means no CFG preview, so nothing to warm for."""
    t = _trainer(
        tmp_path,
        captions={"train cap": ""},
        sample_prompts=[],
        sample_negative_prompt="ugly",
    )
    t._pre_cache_text_embeddings()

    assert "ugly" not in t.text_cache


def test_negative_is_idempotent_within_the_same_trainer_instance(tmp_path):
    """A second pre-cache pass on the SAME instance must not re-encode the
    negative — it's already resident in ``self.text_cache`` (the ``neg not
    in self.text_cache`` guard), matching every sibling's warm block."""
    t = _trainer(
        tmp_path,
        captions={"train cap": ""},
        sample_prompts=[{"prompt": "a preview prompt"}],
        sample_negative_prompt="ugly",
    )
    t._pre_cache_text_embeddings()
    assert "ugly" in t.text_cache

    t.driver.encoded.clear()
    t._pre_cache_text_embeddings()
    assert "ugly" not in t.driver.encoded  # already cached — not re-encoded


# ── W3.T8: disk-cache key carries the TE template fingerprint ──────────────
#
# ``TextEmbeddingCache.caption_to_filename`` hashes ONLY the string it is
# given. Before this fix, that string was the raw caption alone — a future
# edit to the driver's prefix/suffix chat template or its quotation-aware
# tokenization quote pairs would silently reuse embeddings encoded under the
# OLD template (the same poisoned-cache incident class the qwen_image /
# boogu_image families already guard against). ``_disk_cache_key`` now bakes
# ``te_template_fingerprint()`` (driver.py) into the hashed string.


def test_disk_cache_key_changes_when_chat_template_changes(monkeypatch):
    from app.engine.models.families.longcat_image import driver as longcat_driver
    from app.engine.models.families.longcat_image.trainer import _disk_cache_key

    original = _disk_cache_key("a cat")
    monkeypatch.setattr(
        longcat_driver,
        "PROMPT_TEMPLATE_SUFFIX",
        "<|im_end|>\n<|im_start|>assistant\nCHANGED\n",
    )
    changed = _disk_cache_key("a cat")

    assert original != changed


def test_disk_cache_key_changes_when_quote_pairs_change(monkeypatch):
    from app.engine.models.families.longcat_image import driver as longcat_driver
    from app.engine.models.families.longcat_image.trainer import _disk_cache_key

    original = _disk_cache_key("a cat")
    monkeypatch.setattr(longcat_driver, "QUOTE_PAIRS", [("'", "'")])
    changed = _disk_cache_key("a cat")

    assert original != changed


def test_disk_cache_key_stable_for_the_same_template():
    from app.engine.models.families.longcat_image.trainer import _disk_cache_key

    assert _disk_cache_key("a cat") == _disk_cache_key("a cat")


def test_template_change_invalidates_disk_cache(tmp_path, monkeypatch):
    """End-to-end: a template edit must force a re-encode, not a stale hit."""
    from app.engine.models.families.longcat_image import driver as longcat_driver

    cold = _trainer(tmp_path, captions={"a cat": ""})
    cold._pre_cache_text_embeddings()
    assert cold.driver.encoded == ["a cat"]

    # Simulate a future template edit.
    monkeypatch.setattr(
        longcat_driver,
        "PROMPT_TEMPLATE_SUFFIX",
        "<|im_end|>\n<|im_start|>assistant\nCHANGED\n",
    )

    warm = _trainer(tmp_path, captions={"a cat": ""})
    warm._pre_cache_text_embeddings()

    # Re-encoded — NOT silently served from the disk entry written under
    # the old template.
    assert warm.driver.encoded == ["a cat"]
