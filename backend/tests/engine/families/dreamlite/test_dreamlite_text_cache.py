"""DreamLite trainer text-cache tests — disk-cache key template identity (W3.T8 + fix-wave).

Pins the fix: ``TextEmbeddingCache.caption_to_filename`` hashes ONLY the
string it is given. Before this task, that string was the (optionally
``"[Generate]: "``-prefixed) caption alone — the pinned chat template +
``drop_idx`` that ``DreamLiteDriver.encode_text`` wraps every caption in
never reached the hashed key. A future template/drop-idx edit would have
silently served embeddings encoded under the OLD template (the same
poisoned-cache incident class the qwen_image/boogu_image families already
guard against). ``DreamLiteTrainer._disk_cache_key`` now bakes
``te_template_fingerprint()`` (driver.py) into the hashed string.

Fix-wave addendum: the first pass only hashed the module-level DEFAULT
``drop_idx``/``max_sequence_length``, not the per-definition
``te.drop_idx``/``te.max_sequence_length`` OVERRIDE — the reachable case,
since ``DreamLiteDriver.__init__`` already resolves both from
``architecture_params``. ``_disk_cache_key`` moved off ``@staticmethod`` to
a plain instance method so it can read ``self.driver``'s resolved values.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.dreamlite.driver import (
    _DEFAULT_DROP_IDX,
    _DEFAULT_MAX_SEQUENCE_LENGTH,
    DreamLiteDriver,
)
from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

_L, _D = 6, 8  # tiny embedding dims


class _FakeDriver:
    """Driver stand-in: counts calls, returns deterministic tiny tensors.

    Also carries ``max_sequence_length``/``drop_idx`` — the resolved,
    possibly per-definition-overridden values ``DreamLiteDriver.__init__``
    would compute — since ``DreamLiteTrainer._disk_cache_key`` reads them
    off ``self.driver``.
    """

    def __init__(
        self,
        max_sequence_length: int = _DEFAULT_MAX_SEQUENCE_LENGTH,
        drop_idx: int = _DEFAULT_DROP_IDX,
    ) -> None:
        self.encoded: list[str] = []
        self.max_sequence_length = max_sequence_length
        self.drop_idx = drop_idx

    def encode_text(self, texts, dtype):
        self.encoded.extend(texts)
        b = len(texts)
        emb = torch.stack(
            [torch.full((_L, _D), float(len(t))) for t in texts],
        ).to(dtype)
        mask = torch.ones(b, _L, dtype=torch.int64)
        return SimpleNamespace(embeddings=emb, attention_mask=mask)


def _make_definition(architecture_params: dict) -> MagicMock:
    """Mock ModelDefinition carrying a per-definition ``te.max_sequence_length``
    / ``te.drop_idx`` override — mirrors ``_make_dreamlite_definition`` in
    test_dreamlite_family.py."""
    definition = MagicMock(spec=ModelDefinition)
    definition.architecture_params = architecture_params
    return definition


def _trainer(
    tmp_path=None,
    captions: dict[str, str] | None = None,
    driver: "_FakeDriver | None" = None,
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
    t.driver = driver if driver is not None else _FakeDriver()
    t.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    t._log_writer = None
    t._resolve_te_cache_dirs = lambda: [str(tmp_path)] if tmp_path else []
    t._build_caption_hints = lambda: dict(captions or {})
    t._resolve_loading_dtype = lambda: torch.float32
    return t


# ── Unit: the disk-cache-key template fingerprint ──────────────────────────
#
# ``_disk_cache_key`` is a ``DreamLiteTrainer`` PLAIN INSTANCE METHOD (moved
# off ``@staticmethod`` in the W3 fix-wave) so it can fold in
# ``self.driver.max_sequence_length`` / ``self.driver.drop_idx`` — the
# EFFECTIVE, possibly per-definition-overridden values for this run.


def test_disk_cache_key_changes_when_chat_template_changes(monkeypatch):
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    t = _trainer()
    original = t._disk_cache_key("[Generate]: a cat")
    monkeypatch.setattr(
        dreamlite_driver,
        "DREAMLITE_PROMPT_TEMPLATE",
        "<|im_start|>system\nCHANGED<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    )
    changed = t._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_changes_when_drop_idx_default_changes(monkeypatch):
    """Module-DEFAULT axis (no per-definition override present): a driver
    constructed AFTER a ``_DEFAULT_DROP_IDX`` edit must resolve the NEW
    default and produce a different key — mirrors a training run started
    after a module-constant bump. Uses the REAL ``DreamLiteDriver`` (not the
    fake) because the default is resolved once in ``__init__`` via
    ``arch.get("te.drop_idx", _DEFAULT_DROP_IDX)`` — an already-constructed
    instance's resolved ``self.drop_idx`` correctly does NOT float with a
    later monkeypatch (that would mean an in-flight run's truncation length
    silently drifted from its own disk key mid-training)."""
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    original_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    original = _trainer(driver=original_driver)._disk_cache_key("[Generate]: a cat")

    monkeypatch.setattr(dreamlite_driver, "_DEFAULT_DROP_IDX", 99)

    changed_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    changed = _trainer(driver=changed_driver)._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_changes_when_max_sequence_length_default_changes(monkeypatch):
    """Same as above for ``_DEFAULT_MAX_SEQUENCE_LENGTH``."""
    from app.engine.models.families.dreamlite import driver as dreamlite_driver

    original_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    original = _trainer(driver=original_driver)._disk_cache_key("[Generate]: a cat")

    monkeypatch.setattr(dreamlite_driver, "_DEFAULT_MAX_SEQUENCE_LENGTH", 99)

    changed_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    changed = _trainer(driver=changed_driver)._disk_cache_key("[Generate]: a cat")

    assert original != changed


def test_disk_cache_key_stable_for_the_same_template():
    """Two INDEPENDENT trainer instances/computations must agree — no
    accidental nondeterminism (e.g. from set/dict iteration order) leaking
    into the hashed string."""
    t1 = _trainer()
    t2 = _trainer()

    assert t1._disk_cache_key("[Generate]: a cat") == t2._disk_cache_key(
        "[Generate]: a cat",
    )


# ── W3 fix-wave: per-definition override axis (not just module defaults) ──
#
# The finding this closes: the fingerprint hashed only the module-level
# DEFAULTS, not a per-definition ``te.drop_idx`` / ``te.max_sequence_length``
# override — even though both are already exercised as explicit overrides in
# base.yaml/mobile.yaml. Driven through the REAL override-resolution path
# (``DreamLiteDriver.__init__`` reading ``architecture_params``), NOT a
# monkeypatched module constant — the override is the reachable production
# case.


def test_disk_cache_key_changes_when_definition_overrides_max_sequence_length():
    default_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    assert default_driver.max_sequence_length == _DEFAULT_MAX_SEQUENCE_LENGTH

    overridden_driver = DreamLiteDriver(
        _make_definition({"te.max_sequence_length": 99}),
        torch.device("cpu"),
    )
    assert overridden_driver.max_sequence_length == 99

    default_trainer = _trainer(driver=default_driver)
    overridden_trainer = _trainer(driver=overridden_driver)

    assert default_trainer._disk_cache_key(
        "[Generate]: a cat",
    ) != overridden_trainer._disk_cache_key("[Generate]: a cat")


def test_disk_cache_key_changes_when_definition_overrides_drop_idx():
    default_driver = DreamLiteDriver(_make_definition({}), torch.device("cpu"))
    assert default_driver.drop_idx == _DEFAULT_DROP_IDX

    overridden_driver = DreamLiteDriver(
        _make_definition({"te.drop_idx": 5}),
        torch.device("cpu"),
    )
    assert overridden_driver.drop_idx == 5

    default_trainer = _trainer(driver=default_driver)
    overridden_trainer = _trainer(driver=overridden_driver)

    assert default_trainer._disk_cache_key(
        "[Generate]: a cat",
    ) != overridden_trainer._disk_cache_key("[Generate]: a cat")


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


def test_drop_idx_override_invalidates_disk_cache(tmp_path):
    """End-to-end: a run with a per-definition ``te.drop_idx`` override must
    not silently reuse a disk entry written by a run at the module default —
    proving the template-prefix-drop axis is covered, not just the chat
    template text."""
    cold = _trainer(
        tmp_path,
        captions={"a cat": ""},
        driver=_FakeDriver(drop_idx=_DEFAULT_DROP_IDX),
    )
    cold._pre_cache_text_embeddings()
    assert cold.driver.encoded == ["[Generate]: a cat"]

    # Simulate a run against a definition overriding te.drop_idx.
    warm = _trainer(
        tmp_path,
        captions={"a cat": ""},
        driver=_FakeDriver(drop_idx=5),
    )
    warm._pre_cache_text_embeddings()

    # Re-encoded — the differing effective drop_idx produced a fresh key,
    # not a stale hit against the entry written at drop_idx=34.
    assert warm.driver.encoded == ["[Generate]: a cat"]
