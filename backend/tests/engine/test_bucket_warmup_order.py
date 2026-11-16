"""VRAM-safe largest-bucket-first warmup ordering (B-TEST-4).

Pins ``PipelineTrainMixin._iter_training_batches`` — the anti-fragmentation
fix that forces the FIRST batch of each epoch (and the first batch after each
sampling pause, signalled via ``_vram_rewarm_pending``) to come from the
largest-pixel bucket while leaving every other batch fully random.

Exercises the REAL generator (extracted verbatim from ``train()``); the only
stubs are the ``inventory``/``config`` owner state it reads — no mock touches
the ordering logic under test. Determinism comes from ``random.seed`` (the
generator uses the module-level ``random``).
"""

from __future__ import annotations

import random
import threading
from types import SimpleNamespace

import pytest

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin, WarmUpBatch


# ── Stub inventory ────────────────────────────────────────────────────────
# Three buckets of distinct pixel area. "large" (1024²) is the peak the warmup
# must reserve first; each item carries a unique id + its bucket label so the
# tests can assert identity/coverage.

def _item(idx: int, w: int, h: int, bucket: str) -> dict:
    return {"id": idx, "target_w": w, "target_h": h, "bucket": bucket}


def _make_inventory() -> list[dict]:
    # (count, w, h, label) — three buckets of distinct pixel area.
    layout = [(3, 1024, 1024, "large"), (4, 512, 512, "medium"), (5, 256, 256, "small")]
    inv: list[dict] = []
    for count, w, h, label in layout:
        for _ in range(count):
            inv.append(_item(len(inv), w, h, label))
    return inv


def _owner(inventory: list[dict], *, vram_safe: bool = True) -> SimpleNamespace:
    """Minimal stand-in exposing only what the generator reads/writes."""
    return SimpleNamespace(
        inventory=inventory,
        config={"vram_safe_bucket_order": vram_safe},
    )


def _iter(owner: SimpleNamespace, batch_size: int):
    # Unbound method call → runs the REAL production generator against the stub.
    return PipelineTrainMixin._iter_training_batches(owner, batch_size)


# ── First batch of an epoch comes from the max-pixel bucket ───────────────

def test_first_batch_of_epoch_is_largest_bucket():
    inv = _make_inventory()
    for seed in range(15):  # hold across many shuffles, not one lucky seed
        random.seed(seed)
        it = _iter(_owner(inv), batch_size=1)
        first = next(it)
        assert [x["bucket"] for x in first] == ["large"], (
            f"seed={seed}: first batch not from largest bucket: {first}"
        )


def test_first_batch_largest_bucket_batch_size_gt_1():
    inv = _make_inventory()
    random.seed(3)
    it = _iter(_owner(inv), batch_size=2)
    first = next(it)
    assert len(first) == 2 and all(x["bucket"] == "large" for x in first)


# ── First batch after a sampling interruption re-warms the largest bucket ──

def test_rewarm_after_sampling_pause_is_largest_bucket():
    inv = _make_inventory()
    random.seed(7)
    owner = _owner(inv)
    it = _iter(owner, batch_size=1)

    next(it)          # initial epoch-start warm
    next(it)          # a regular (random) batch
    # Sampling ran empty_cache → the train loop sets this flag (pipeline:775).
    owner._vram_rewarm_pending = True
    rewarm = next(it)

    assert [x["bucket"] for x in rewarm] == ["large"], (
        f"batch after sampling pause not re-warmed from largest: {rewarm}"
    )


# ── All other batches: random but complete (each item once per epoch) ──────

def test_epoch_after_warm_covers_every_item_exactly_once():
    inv = _make_inventory()
    n = len(inv)
    random.seed(11)
    it = _iter(_owner(inv), batch_size=1)

    next(it)                                    # discard the extra warm batch
    epoch = [next(it) for _ in range(n)]        # the epoch's own N batches
    ids = sorted(b[0]["id"] for b in epoch)
    assert ids == list(range(n)), (
        "epoch must cover every item exactly once (complete, no duplicates)"
    )


def test_epoch_after_warm_complete_batch_size_gt_1():
    inv = _make_inventory()
    n = len(inv)
    random.seed(13)
    it = _iter(_owner(inv), batch_size=2)

    next(it)  # discard warm batch
    # bs=2 groups by bucket: large→2 batches(3), medium→2, small→3 = 7 batches.
    collected: list[int] = []
    while len(collected) < n:
        collected.extend(x["id"] for x in next(it))
    assert sorted(collected) == list(range(n))


# ── Flag off → pure random (no forced first batch, no warm duplicate) ──────

def test_flag_off_epoch_has_no_warm_duplicate():
    inv = _make_inventory()
    n = len(inv)
    random.seed(2)
    it = _iter(_owner(inv, vram_safe=False), batch_size=1)

    # With the flag OFF the first N single-item batches ARE the whole epoch:
    # no extra largest-bucket warm batch is injected, so N items cover all
    # ids exactly once (with the flag ON the first item is a duplicate).
    epoch = [next(it) for _ in range(n)]
    ids = sorted(b[0]["id"] for b in epoch)
    assert ids == list(range(n))


# ── A family that routes per optimizer step: one bucket per accumulation window ──

ACCUM = 4


def _windowed_owner(inventory, *, opt_in: bool = True, vram_safe: bool = True, accum: int = ACCUM):
    owner = _owner(inventory, vram_safe=vram_safe)
    owner.config["gradient_accumulation_steps"] = accum
    owner.one_bucket_per_accumulation_window = opt_in
    return owner


def _within(check, seconds: float = 30.0) -> None:
    """Run ``check`` with a time limit: a windowed iterator that holds every batch back forever
    loops without yielding, and that must fail here, not hang the run."""
    outcome: dict = {}

    def run():
        try:
            check()
        except BaseException as exc:  # noqa: BLE001 -- re-raised below, in the test's thread
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"the iterator yielded no batch for {seconds}s"
    if "error" in outcome:
        raise outcome["error"]


def _training_batch(it, warm_ups: list | None = None):
    """The next batch that trains. A warm-up batch (a throwaway forward and backward on the
    largest bucket, discarded) is set aside in ``warm_ups``: it is never a training batch."""
    batch = next(it)
    while isinstance(batch, WarmUpBatch):
        if warm_ups is not None:
            warm_ups.append(batch)
        batch = next(it)
    return batch


def test_a_dataset_smaller_than_one_window_still_trains_every_item():
    """Buckets of 1, 2 and 3 items, windows of 4: no bucket fills a window in one epoch. Its
    batches wait across epochs until one does, so every item trains, at once per epoch."""
    inventory = [_item(i, w, w, f"b{w}") for i, w in enumerate([256] + [512] * 2 + [768] * 3)]

    def check():
        random.seed(0)
        it = _iter(_windowed_owner(inventory, vram_safe=False), 1)
        counts = dict.fromkeys(range(6), 0)
        for _ in range(240 // ACCUM):
            window = [_training_batch(it) for _ in range(ACCUM)]
            assert len({x["bucket"] for batch in window for x in batch}) == 1
            for batch in window:
                counts[batch[0]["id"]] += 1
        # 240 batches = 40 epochs of 6 items; each item within one window of 40.
        assert all(abs(c - 40) <= ACCUM for c in counts.values()), counts

    _within(check)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("vram_safe", [True, False], ids=["warmup", "no-warmup"])
def test_an_opted_in_family_gets_one_bucket_per_accumulation_window(batch_size, vram_safe):
    """Every ``gradient_accumulation_steps`` consecutive training batches (one optimizer step)
    come from one bucket, across epoch boundaries and the re-warm after a sample; a warm-up
    batch only ever comes between two windows; every item still trains."""
    _within(lambda: _one_bucket_per_window(batch_size, vram_safe))


def _one_bucket_per_window(batch_size, vram_safe):
    inv = _make_inventory()
    for seed in range(10):
        random.seed(seed)
        owner = _windowed_owner(inv, vram_safe=vram_safe)
        it = _iter(owner, batch_size)
        seen: set[int] = set()
        for step in range(40):  # several epochs
            if step % 7 == 3:
                owner._vram_rewarm_pending = True  # a sample ran between two optimizer steps
            window = []
            for slot in range(ACCUM):
                warm_ups: list = []
                window.append(_training_batch(it, warm_ups))
                assert not warm_ups or slot == 0, f"seed={seed} step={step}: a warm-up inside a window"
            buckets = {x["bucket"] for batch in window for x in batch}
            assert len(buckets) == 1, f"seed={seed} step={step}: one window spans {sorted(buckets)}"
            seen.update(x["id"] for batch in window for x in batch)
        assert seen == set(range(len(inv))), f"seed={seed}: items never trained: {set(range(len(inv))) - seen}"


@pytest.mark.parametrize("accum", [1, 4, 16])
def test_accumulation_never_changes_how_often_an_item_trains(accum):
    """Buckets of 64 items and 1 item, batch size 1, no warm-up. However many batches a window
    holds, every item trains once per epoch in the long run: a bucket's leftover batches wait
    for the next epoch, never padded with repeats (that would train the single item 16 times
    per epoch at accumulation 16, a 20 % share instead of 1.5 %).

    Tolerance: after the batches of ``EPOCHS`` epochs, an item is behind by at most the batches
    its bucket holds back (fewer than one window) and ahead by at most the part of a later
    epoch already drawn (under one window per bucket), so every count lies within
    ``EPOCHS +- 2 * accum``."""
    inventory = [_item(i, 512, 512, "big") for i in range(64)] + [_item(64, 256, 256, "single")]

    def check():
        owner = _windowed_owner(inventory, vram_safe=False, accum=accum)
        random.seed(0)
        it = _iter(owner, 1)
        counts = dict.fromkeys(range(65), 0)
        for _ in range(EPOCHS * 65):
            counts[_training_batch(it)[0]["id"]] += 1
        off = {i: c for i, c in counts.items() if abs(c - EPOCHS) > 2 * accum}
        assert not off, f"accumulation {accum}: items trained off once per epoch ({EPOCHS} epochs): {off}"

    _within(check)


EPOCHS = 160
REWARM_EVERY = 5  # optimizer steps between two samples (each sample asks for a re-warm)
SPARSE = [_item(i, 512, 512, "small") for i in range(64)] + [_item(64, 1024, 1024, "large")]


@pytest.mark.parametrize("accum", [1, 4, 16])
def test_the_warm_up_never_changes_how_often_an_item_trains(accum):
    """The default warm-up, a largest bucket of one item, 64 smaller items, and a sample every
    ``REWARM_EVERY`` optimizer steps. A warm-up batch is a throwaway, never a training batch,
    so it changes no item's training count (an extra warm window would give the single largest
    item a 21 % share at accumulation 16 instead of 1.5 %). Tolerance as above: every count
    within ``EPOCHS +- 2 * accum``."""

    def check():
        owner = _windowed_owner(SPARSE, vram_safe=True, accum=accum)
        random.seed(0)
        it = _iter(owner, 1)
        counts = dict.fromkeys(range(65), 0)
        warm_ups: list = []
        for step in range(EPOCHS * 65 // accum):
            if step % REWARM_EVERY == REWARM_EVERY - 1:
                owner._vram_rewarm_pending = True  # a sample ran after this optimizer step
            for _ in range(accum):
                counts[_training_batch(it, warm_ups)[0]["id"]] += 1
        off = {i: c for i, c in counts.items() if abs(c - EPOCHS) > 2 * accum}
        assert not off, f"accumulation {accum}: items trained off once per epoch ({EPOCHS} epochs): {off}"
        assert warm_ups, "no warm-up batch at all"

    _within(check)


@pytest.mark.parametrize("accum", [1, 16])
def test_the_largest_shape_comes_first_at_the_start_and_after_each_sample(accum):
    """A largest bucket of one item that fills a window only every ``accum`` epochs: the first
    batch through the model at the start, and the first after each sample, is still that
    largest shape (a warm-up batch drawn from its inventory, consumed by no epoch)."""

    def check():
        for seed in range(5):
            random.seed(seed)
            owner = _windowed_owner(SPARSE, vram_safe=True, accum=accum)
            it = _iter(owner, 1)
            first = next(it)
            assert isinstance(first, WarmUpBatch) and first[0]["bucket"] == "large", first
            for step in range(30):
                for _ in range(accum):
                    _training_batch(it)
                owner._vram_rewarm_pending = True  # a sample after this optimizer step
                after = next(it)
                assert isinstance(after, WarmUpBatch) and after[0]["bucket"] == "large", (seed, step, after)

    _within(check)


def test_a_warm_up_draw_leaves_the_random_stream_alone():
    """The warm-up batch is drawn without moving ``random``: the training batches are the same
    as with warm-up off (where no warm-up batch is drawn at all)."""
    def batches(vram_safe):
        random.seed(3)
        owner = _windowed_owner([dict(x) for x in SPARSE], vram_safe=vram_safe, accum=4)
        it = _iter(owner, 1)
        return [_training_batch(it)[0]["id"] for _ in range(400)]

    def check():
        assert batches(True) == batches(False)

    _within(check)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("vram_safe", [True, False], ids=["warmup", "no-warmup"])
def test_a_family_that_does_not_opt_in_keeps_its_batch_order(batch_size, vram_safe):
    """Gradient accumulation alone changes nothing: the same seed gives the same batches as an
    owner with no accumulation setting at all."""
    for seed in range(5):
        orders = []
        # Each owner its own list: the iterator shuffles ``inventory`` in place.
        for owner in (_windowed_owner(_make_inventory(), opt_in=False, vram_safe=vram_safe),
                      _owner(_make_inventory(), vram_safe=vram_safe)):
            random.seed(seed)
            it = _iter(owner, batch_size)
            orders.append([[x["id"] for x in next(it)] for _ in range(40)])
        assert orders[0] == orders[1], f"seed={seed}"


def test_only_the_dual_expert_families_opt_in():
    """Single-expert families keep their batching: the base default is off."""
    assert getattr(PipelineTrainMixin, "one_bucket_per_accumulation_window", False) is False


def test_flag_off_does_not_force_largest_first():
    inv = _make_inventory()
    first_buckets = []
    for seed in range(25):
        random.seed(seed)
        it = _iter(_owner(inv, vram_safe=False), batch_size=1)
        first_buckets.append(next(it)[0]["bucket"])
    # A forced-largest-first would make every entry "large"; pure random must not.
    assert any(b != "large" for b in first_buckets), (
        "flag-off first batch appears forced to the largest bucket"
    )
