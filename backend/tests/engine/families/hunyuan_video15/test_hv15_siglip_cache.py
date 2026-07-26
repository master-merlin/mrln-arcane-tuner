"""hv15 I2V Siglip image-embedding aux cache (LTX-2 audio-cache pattern).

``_pre_cache_aux`` encodes the first frame of each item through the Siglip
image encoder while it is resident (then offloads it); ``build_batch_extra``
serves the cached ``[B, 729, 1152]`` ``image_embeds`` to the training batch.
T2V runs are no-ops on both paths.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import Hv15Driver
from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer

_TOKENS, _DIM = 5, 4  # tiny stand-ins for 729 / 1152


class _FakeFeatureExtractor:
    def preprocess(self, images=None, do_resize=True, return_tensors="pt",
                   do_convert_rgb=True):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


class _FakeImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)  # gives parameters() a dtype

    def forward(self, pixel_values=None):
        return SimpleNamespace(
            last_hidden_state=torch.full((1, _TOKENS, _DIM), 3.0)
        )


class _FailingImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)  # gives parameters() a dtype

    def forward(self, pixel_values=None):
        raise RuntimeError("siglip encode blew up (e.g. CUDA OOM)")


class _PartialFailingImageEncoder(torch.nn.Module):
    """Fails the FIRST encode, then succeeds — a partial-failure run."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)  # gives parameters() a dtype
        self.calls = 0

    def forward(self, pixel_values=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first item siglip encode blew up")
        return SimpleNamespace(
            last_hidden_state=torch.full((1, _TOKENS, _DIM), 3.0)
        )


class _RecLogger:
    """Records structured log events so tests can assert visibility."""

    def __init__(self):
        self.infos: list[tuple] = []
        self.warnings: list[tuple] = []

    def info(self, event, **kw):
        self.infos.append((event, kw))

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def debug(self, *a, **k):
        pass


class _FakeLatentManager:
    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}.safetensors"


def _make_trainer(mode: str, tmp_path, items) -> Hv15Trainer:
    t = object.__new__(Hv15Trainer)
    t.device = torch.device("cpu")
    t.config = {"cache_latents": True}
    t.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    driver = SimpleNamespace(
        is_i2v=(mode == "i2v"),
        image_encoder=_FakeImageEncoder() if mode == "i2v" else None,
        feature_extractor=_FakeFeatureExtractor() if mode == "i2v" else None,
    )
    t.driver = driver
    t.inventory = items
    t.latent_manager = _FakeLatentManager()
    return t


def _still_item(tmp_path, name="img1") -> dict:
    from PIL import Image

    img_path = os.path.join(str(tmp_path), f"{name}.png")
    Image.new("RGB", (16, 16), (128, 64, 32)).save(img_path)
    cache_dir = os.path.join(str(tmp_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return {"id": name, "path": img_path, "cache_dir": cache_dir, "is_video": False}


def test_precache_aux_encodes_and_offloads(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()

    path = os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    assert os.path.exists(path)
    from safetensors.torch import load_file

    emb = load_file(path)["image_embeds"]
    assert emb.shape == (_TOKENS, _DIM)
    assert torch.all(emb == 3.0)


def test_precache_aux_skips_existing(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()
    mtime = os.path.getmtime(
        os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    )
    t2 = _make_trainer("i2v", tmp_path, [item])
    t2._pre_cache_aux()
    assert os.path.getmtime(
        os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    ) == mtime


def test_precache_aux_noop_for_t2v(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("t2v", tmp_path, [item])
    t._pre_cache_aux()
    assert not os.path.exists(os.path.join(item["cache_dir"], "siglip"))


def test_build_batch_extra_serves_cached_embeds(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()

    extra = t.build_batch_extra([item])
    emb = extra[Hv15Driver.BATCH_IMAGE_EMBED]
    assert emb.shape == (1, _TOKENS, _DIM)
    assert torch.all(emb == 3.0)


def test_build_batch_extra_zero_fills_missing_items(tmp_path):
    cached = _still_item(tmp_path, "have")
    missing = _still_item(tmp_path, "missing")
    t = _make_trainer("i2v", tmp_path, [cached])
    t._pre_cache_aux()  # only "have" is cached

    extra = t.build_batch_extra([cached, missing])
    emb = extra[Hv15Driver.BATCH_IMAGE_EMBED]
    assert emb.shape == (2, _TOKENS, _DIM)
    assert torch.all(emb[0] == 3.0)
    assert torch.all(emb[1] == 0.0)


def test_build_batch_extra_empty_without_any_cache(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    assert t.build_batch_extra([item]) == {}


def test_build_batch_extra_noop_for_t2v(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("t2v", tmp_path, [item])
    assert t.build_batch_extra([item]) == {}


# ── silent-failure policy: encoder failures counted + surfaced visibly ─────


def test_precache_aux_partial_failure_warns_and_continues(tmp_path):
    """A per-item Siglip encode failure must NOT kill the run when OTHER items
    encode — it is counted and the item is left uncached (→ zero-filled
    downstream), surfaced by a visible warning, never silent."""
    bad = _still_item(tmp_path, "boom")
    good = _still_item(tmp_path, "good")
    # bad fails first, good succeeds → partial failure.
    t = _make_trainer("i2v", tmp_path, [bad, good])
    rec = _RecLogger()
    t.logger = rec
    t.driver.image_encoder = _PartialFailingImageEncoder()

    t._pre_cache_aux()  # must not raise — partial failure degrades gracefully

    # Good item cached; failing item left uncached.
    assert os.path.exists(
        os.path.join(good["cache_dir"], "siglip", "good.safetensors")
    )
    assert not os.path.exists(
        os.path.join(bad["cache_dir"], "siglip", "boom.safetensors")
    )
    # Summary carries both counts.
    done = [kw for ev, kw in rec.infos if ev == "hv15_siglip_precache_done"]
    assert done and done[0]["failed"] == 1 and done[0]["encoded"] == 1
    # A VISIBLE warning surfaces the zero-fill exposure.
    warn_events = [ev for ev, _ in rec.warnings]
    assert "hv15_siglip_precache_incomplete" in warn_events


def test_precache_aux_raises_when_all_items_fail_to_encode(tmp_path):
    """TOTAL Siglip encode failure must ESCALATE — proceeding would train a
    100%% zero-image_embeds run, i.e. a silently mislabeled t2v run."""
    item = _still_item(tmp_path, "boom")
    t = _make_trainer("i2v", tmp_path, [item])
    rec = _RecLogger()
    t.logger = rec
    t.driver.image_encoder = _FailingImageEncoder()

    with pytest.raises(RuntimeError, match="hv15_siglip_precache_incomplete"):
        t._pre_cache_aux()

    # Nothing cached for the failing item.
    assert not os.path.exists(
        os.path.join(item["cache_dir"], "siglip", "boom.safetensors")
    )
    # The visible incomplete warning still fired before escalating.
    warn_events = [ev for ev, _ in rec.warnings]
    assert "hv15_siglip_precache_incomplete" in warn_events


def test_precache_aux_resume_with_cached_items_does_not_escalate(tmp_path):
    """Resume nuance: when prior items are already cached (skipped>0) and only
    the NEW item(s) fail, the run still has real image_embeds for the cached
    majority — that is the partial-degrade case (warn + zero-fill), NOT the
    total-failure case. Escalating here would hard-abort a healthy resume."""
    cached = _still_item(tmp_path, "cached")
    t1 = _make_trainer("i2v", tmp_path, [cached])
    t1._pre_cache_aux()  # working encoder → "cached" lands on disk

    new_bad = _still_item(tmp_path, "newboom")
    t2 = _make_trainer("i2v", tmp_path, [cached, new_bad])
    rec = _RecLogger()
    t2.logger = rec
    t2.driver.image_encoder = _FailingImageEncoder()

    t2._pre_cache_aux()  # must NOT raise: skipped=1, failed=1, encoded=0

    done = [kw for ev, kw in rec.infos if ev == "hv15_siglip_precache_done"]
    assert done and done[0]["skipped"] == 1 and done[0]["failed"] == 1
    warn_events = [ev for ev, _ in rec.warnings]
    assert "hv15_siglip_precache_incomplete" in warn_events


def test_precache_aux_no_incomplete_warning_when_all_ok(tmp_path):
    item = _still_item(tmp_path, "ok")
    t = _make_trainer("i2v", tmp_path, [item])
    rec = _RecLogger()
    t.logger = rec

    t._pre_cache_aux()

    warn_events = [ev for ev, _ in rec.warnings]
    assert "hv15_siglip_precache_incomplete" not in warn_events


# ── corrupt cache file (poison-pill) handling ──────────────────────────────
#
# Task W2.T2 widened build_batch_extra's load-site catch from (OSError,
# KeyError) to Exception, because safetensors 0.8.0's SafetensorError
# subclasses Exception DIRECTLY (not OSError) — so a truncated/corrupt cache
# file used to crash every subsequent run at the same step (os.path.exists
# still counts the truncated file as "cached"). These tests write a REAL
# corrupt .safetensors file (not a mock) at the exact lookup path and drive
# the real load path.


def _write_corrupt_cache_file(item: dict, name: str) -> str:
    """A genuinely corrupt (truncated) .safetensors file at the exact path
    ``build_batch_extra`` looks up — matches the real crash mode: a bad
    header that raises ``safetensors.SafetensorError`` (NOT OSError/KeyError)."""
    sdir = os.path.join(item["cache_dir"], "siglip")
    os.makedirs(sdir, exist_ok=True)
    path = os.path.join(sdir, f"{name}.safetensors")
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)
    return path


def test_build_batch_extra_degrades_to_miss_on_corrupt_cache_file(tmp_path):
    """A pre-existing corrupt cache file must degrade to a MISS, never raise
    — the exact poison-pill regression this task guards against."""
    item = _still_item(tmp_path, "img1")
    _write_corrupt_cache_file(item, "img1")
    t = _make_trainer("i2v", tmp_path, [item])

    extra = t.build_batch_extra([item])  # must not raise

    assert extra == {}  # nothing usable cached → driver falls back to zeros


def test_build_batch_extra_zero_fills_item_with_corrupt_cache_alongside_good(tmp_path):
    """Mixed batch: "good" has a genuinely valid cached embed, "corrupt"'s
    cache file is present-but-corrupt. The corrupt one must degrade exactly
    like "missing" (zero-filled), "good" must be unaffected, and a visible
    warning must name the corrupt path — never an unhandled raise."""
    good = _still_item(tmp_path, "good")
    corrupt = _still_item(tmp_path, "corrupt")
    t = _make_trainer("i2v", tmp_path, [good, corrupt])
    t._pre_cache_aux()  # only "good" gets a real cached embed
    _write_corrupt_cache_file(corrupt, "corrupt")  # poison pill for "corrupt"

    rec = _RecLogger()
    t.logger = rec
    extra = t.build_batch_extra([good, corrupt])  # must not raise

    emb = extra[Hv15Driver.BATCH_IMAGE_EMBED]
    assert emb.shape == (2, _TOKENS, _DIM)
    assert torch.all(emb[0] == 3.0)  # good item keeps its real embedding
    assert torch.all(emb[1] == 0.0)  # corrupt item degrades to zero-fill
    warn = [kw for ev, kw in rec.warnings if ev == "hv15_siglip_cache_load_failed"]
    assert warn and "corrupt.safetensors" in warn[0]["path"]


def test_precache_aux_does_not_repair_a_pre_existing_corrupt_cache_file(tmp_path):
    """Documents the ACTUAL contract (not something this task fixes):
    ``_pre_cache_aux``'s skip check is a plain ``os.path.exists``, content-
    blind. A pre-existing corrupt file is therefore counted as "skipped" and
    is NEVER re-encoded/overwritten by a later precache pass — only
    ``build_batch_extra``'s load-time catch prevents the crash. The poison
    pill persists on disk and the item permanently zero-fills rather than
    being repaired. Same content-blind exists-check pattern as
    ``LatentManager`` (out of scope here); not a regression from db26b618."""
    item = _still_item(tmp_path, "img1")
    path = _write_corrupt_cache_file(item, "img1")
    with open(path, "rb") as f:
        corrupt_bytes = f.read()
    t = _make_trainer("i2v", tmp_path, [item])

    t._pre_cache_aux()  # must not raise

    with open(path, "rb") as f:
        assert f.read() == corrupt_bytes  # left untouched, never re-encoded
