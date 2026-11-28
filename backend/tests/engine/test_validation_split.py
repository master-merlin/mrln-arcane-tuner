"""The held-out split for training experiments: by source image, the same for every arm."""
import json
import random

import numpy as np
import pytest
from PIL import Image

from app.engine.core.pipeline import validation_split
from app.engine.core.pipeline.validation_split import (
    ValidationSplit,
    difference_hash,
    measurement_difference,
    same_measurement,
    split_file_content,
    split_holdout,
    split_separate_dataset,
)
from app.engine.core.validation_compare import (
    COMPARED,
    DEFAULT_TIMESTEP_BANDS,
    LEFT_OUT,
    REPRESENTATIVE_LEFT_OUT,
)
from app.engine.models.base import BaseTrainingConfig

# Distinct hashes far apart, one bit pattern per source, so tests choose the links.
FAR = [int("".join(str((i >> b) & 1) * 8 for b in range(8)), 2) for i in range(1, 200)]


def _item(name, dataset="ds", width=512, height=512, caption="a", variant="original"):
    return {
        "dataset_name": dataset,
        "id": name,
        "path": f"/data/{dataset}/{name}",
        "caption": caption,
        "target_w": width,
        "target_h": height,
        "variant": variant,
    }


def _inventory(count, dataset="ds"):
    return [_item(f"{i:03d}.png", dataset) for i in range(count)]


def _hashes(items, overrides=None):
    """Well-separated hashes by source, with chosen paths given near-identical ones."""
    paths = sorted({it["path"] for it in items})
    table = {path: FAR[i] for i, path in enumerate(paths)}
    table.update(overrides or {})
    return table.__getitem__


def _held_out(result):
    return sorted(result.holdout_keys)


def test_the_same_files_and_seed_give_the_same_split_in_any_order():
    items = _inventory(40)
    first = split_holdout(items, 0.15, 3, _hashes(items))

    shuffled = list(items)
    random.Random(1).shuffle(shuffled)
    second = split_holdout(shuffled, 0.15, 3, _hashes(items))

    assert _held_out(first) == _held_out(second)


def test_a_different_split_seed_gives_a_different_split():
    items = _inventory(40)
    assert _held_out(split_holdout(items, 0.15, 3, _hashes(items))) != _held_out(
        split_holdout(items, 0.15, 4, _hashes(items))
    )


def test_the_training_seed_cannot_reach_the_split():
    items = _inventory(40)
    random.seed(1)
    np.random.seed(1)
    first = split_holdout(items, 0.15, 3, _hashes(items))
    random.seed(2)
    np.random.seed(2)
    assert _held_out(first) == _held_out(split_holdout(items, 0.15, 3, _hashes(items)))


def test_every_bucket_and_repeat_of_a_source_stays_on_one_side():
    items = []
    for i in range(20):
        items += [
            _item(f"{i:03d}.png", width=512, height=512),
            _item(f"{i:03d}.png", width=1024, height=1024),
            _item(f"{i:03d}.png", width=1024, height=1024),
        ]
    result = split_holdout(items, 0.2, 0, _hashes(items))

    trained = {f"ds/{it['id']}" for it in result.training_items}
    assert trained.isdisjoint(result.holdout_keys)
    assert trained | set(result.holdout_keys) == {f"ds/{i:03d}.png" for i in range(20)}
    assert len(result.training_items) == 3 * (20 - len(result.holdout_keys))


def test_unequal_repeats_do_not_change_the_split():
    items = _inventory(40)
    repeated = list(items)
    for i in range(0, 40, 3):
        repeated += [dict(items[i])] * (i % 7 + 1)
    assert _held_out(split_holdout(items, 0.15, 3, _hashes(items))) == _held_out(
        split_holdout(repeated, 0.15, 3, _hashes(items))
    )


def test_the_holdout_is_the_first_groups_reaching_the_fraction_of_sources():
    items = _inventory(40)
    assert len(split_holdout(items, 0.15, 3, _hashes(items)).holdout_keys) == 6


def test_near_identical_images_land_together_and_different_ones_do_not():
    items = _inventory(30)
    base = FAR[0]
    links = {items[0]["path"]: base, items[1]["path"]: base ^ 0b111}
    hashes = _hashes(items, links)
    groups = split_holdout(items, 0.15, 0, hashes).groups
    together = [g for g in groups if "ds/000.png" in g]
    assert together == [["ds/000.png", "ds/001.png"]]
    assert all(len(g) == 1 for g in groups if g is not together[0])


def test_a_chain_of_near_identical_images_is_one_group():
    """A and C are far apart, but B is near both: photos of one shoot drift like this."""
    items = _inventory(30)
    a = FAR[0]
    b = a ^ 0b111111
    c = b ^ (0b111111 << 8)
    assert bin(a ^ c).count("1") > 6
    hashes = _hashes(items, {items[0]["path"]: a, items[1]["path"]: b, items[2]["path"]: c})
    groups = split_holdout(items, 0.15, 0, hashes).groups
    assert ["ds/000.png", "ds/001.png", "ds/002.png"] in groups


def test_a_group_is_never_cut_by_the_fraction():
    items = _inventory(10)
    same = FAR[0]
    hashes = _hashes(items, {it["path"]: same for it in items[:4]})
    for seed in range(20):
        held = set(split_holdout(items, 0.1, seed, hashes).holdout_keys)
        group = {f"ds/{i:03d}.png" for i in range(4)}
        assert group <= held or group.isdisjoint(held)


def test_two_datasets_sharing_a_filename_keep_two_sources():
    items = [_item(f"{i:03d}.png", "first") for i in range(10)] + [
        _item(f"{i:03d}.png", "second") for i in range(10)
    ]
    random.Random(2).shuffle(items)
    result = split_holdout(items, 0.2, 0, _hashes(items))
    keys = {key for group in result.groups for key in group}
    assert len(keys) == 20
    assert "first/000.png" in keys and "second/000.png" in keys


@pytest.mark.parametrize("count", [1, 2])
def test_a_side_left_empty_fails_naming_the_dataset(count):
    items = _inventory(count, "Tiny Set")
    hashes = _hashes(items, {it["path"]: FAR[0] for it in items})
    with pytest.raises(ValueError, match="Tiny Set"):
        split_holdout(items, 0.5, 0, hashes)


def test_each_source_has_one_fixed_representative():
    items = [
        _item("a.png", width=512, height=512, caption="zebra"),
        _item("a.png", width=1024, height=1024, caption="zebra"),
        _item("a.png", width=1024, height=1024, caption="apple"),
        _item("a.png", width=2048, height=512, caption="zebra"),
        _item("a.png", width=512, height=2048, caption="zebra"),
    ] + [_item(f"{i}.png") for i in range(30)]
    hashes = _hashes(items)
    for seed in range(40):
        result = split_holdout(items, 0.3, seed, hashes)
        if "ds/a.png" in result.holdout_keys:
            break
    chosen = [r for r in result.representatives if r["key"] == "ds/a.png"]
    assert len(chosen) == 1
    # Largest by pixels; among the two 1024 x 1024 entries the sorted first caption.
    # 2048 x 512 and 512 x 2048 have the same pixels as 1024 x 1024, and "1024x1024"
    # is the lexically smallest name of the three.
    assert (chosen[0]["target_w"], chosen[0]["target_h"]) == (1024, 1024)
    assert chosen[0]["caption"] == "apple"
    assert len(result.representatives) == len(result.holdout_keys)


def test_the_representatives_do_not_depend_on_repeats_or_order():
    items = _inventory(40)
    shuffled = list(items) + [dict(items[0])] * 3
    random.Random(3).shuffle(shuffled)
    one = split_holdout(items, 0.15, 3, _hashes(items)).representatives
    two = split_holdout(shuffled, 0.15, 3, _hashes(items)).representatives
    assert [r["key"] for r in one] == [r["key"] for r in two]


def test_a_separate_dataset_leaves_training_whole():
    train = _inventory(20, "train")
    held = _inventory(5, "held")
    result = split_separate_dataset(train, held)

    assert result.training_items == train
    assert sorted(result.holdout_keys) == [f"held/{i:03d}.png" for i in range(5)]
    assert len(result.representatives) == 5


def test_a_file_in_both_the_training_and_the_separate_dataset_fails_naming_it():
    train = _inventory(5, "train")
    held = [dict(_item("x.png", "held"), path=train[2]["path"])]
    with pytest.raises(ValueError, match="002.png"):
        split_separate_dataset(train, held)


def test_an_empty_separate_dataset_fails():
    with pytest.raises(ValueError, match="validation"):
        split_separate_dataset(_inventory(5), [])


def _split_text(**changes):
    content = {"holdout": ["ds/a.png"], "representatives": [{"key": "ds/a.png", "sha256": "0"}],
               "split_seed": 0, "global_triggerword": "", "noise_draws": 1, **changes}
    return json.dumps(content)


def test_another_draw_count_is_the_same_measurement_and_another_seed_is_not():
    assert same_measurement(_split_text(), _split_text(noise_draws=8))
    assert not same_measurement(_split_text(), _split_text(split_seed=1))
    # The reason it is left out is written where the rule is.
    assert "noise_draws" in same_measurement.__doc__


def test_another_band_count_is_another_measurement():
    assert not same_measurement(_split_text(timestep_bands=8), _split_text(timestep_bands=2))
    assert measurement_difference(_split_text(timestep_bands=8), _split_text(timestep_bands=2)) == "timestep_bands"
    assert same_measurement(_split_text(timestep_bands=8), _split_text(timestep_bands=8))
    assert measurement_difference(_split_text(timestep_bands=8), _split_text(timestep_bands=8)) is None


def test_a_split_file_without_a_band_count_reads_as_eight_bands():
    assert same_measurement(_split_text(), _split_text(timestep_bands=8))
    assert not same_measurement(_split_text(), _split_text(timestep_bands=2))
    assert not same_measurement(_split_text(timestep_bands=None), _split_text(timestep_bands=8))


def _written_split(tmp_path):
    """The real text ``split_file_content`` writes for a split of one held-out image."""
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "white").save(path)
    representative = dict(_item("a.png"), path=str(path), prefix="")
    split = ValidationSplit(
        training_items=[], holdout_keys=["ds/a.png"], representatives=[representative], groups=[["ds/a.png"]]
    )
    return split_file_content(split, {}, "m")


def test_every_recorded_key_is_compared_or_left_out_with_a_reason(tmp_path):
    written = json.loads(_written_split(tmp_path))

    assert set(written) == set(COMPARED) | set(LEFT_OUT)
    assert set(COMPARED).isdisjoint(LEFT_OUT)
    for reason in (*LEFT_OUT.values(), *REPRESENTATIVE_LEFT_OUT.values()):
        assert reason.strip().endswith(".") and len(reason.strip()) > 20

    fields = set(written["representatives"][0])
    assert "path" in REPRESENTATIVE_LEFT_OUT
    assert set(REPRESENTATIVE_LEFT_OUT) <= fields
    assert fields - set(REPRESENTATIVE_LEFT_OUT) == {"key", "caption", "prefix", "target_w", "target_h", "sha256"}


# Another value for every key the split file records; a key missing here fails the test.
_OTHER_VALUE = {
    "timestep_bands": 2,
    "split_seed": 1,
    "noise_draws": 4,
    "every_n_steps": 50,
    "holdout_fraction": 0.3,
    "hash_threshold_bits": 7,
    "model": "n",
    "global_triggerword": "ohwx",
    "validation_dataset": "held",
    "holdout": ["ds/a.png", "ds/b.png"],
    "groups": [["ds/a.png", "ds/b.png"]],
    "representatives": lambda items: [dict(items[0], key="ds/b.png")],
}


@pytest.mark.parametrize("key", sorted(set(COMPARED) | set(LEFT_OUT)))
def test_a_changed_key_changes_the_verdict_only_when_it_is_compared(tmp_path, key):
    before = _written_split(tmp_path)
    written = json.loads(before)
    assert key in written
    other = _OTHER_VALUE[key]
    changed = other(written[key]) if callable(other) else other
    assert changed != written[key]

    after = json.dumps({**written, key: changed})
    assert same_measurement(before, after) == (key in LEFT_OUT)


def test_every_written_key_has_another_value_in_the_change_table(tmp_path):
    assert set(json.loads(_written_split(tmp_path))) == set(_OTHER_VALUE)


def test_the_band_default_is_the_schema_default():
    bounds = {name: default for name, default, _low, _high in validation_split._BOUNDS}
    assert DEFAULT_TIMESTEP_BANDS == BaseTrainingConfig.model_fields["validation_timestep_bands"].default == 8
    assert bounds["validation_timestep_bands"] == DEFAULT_TIMESTEP_BANDS


def test_the_difference_hash_is_close_for_near_copies_and_far_for_others(tmp_path):
    rng = np.random.default_rng(0)
    gradient = np.tile(np.linspace(0, 255, 256), (256, 1)).astype("uint8")
    other = rng.integers(0, 255, (256, 256), dtype="uint8")
    noisy = np.clip(gradient.astype("int16") + rng.integers(-3, 4, gradient.shape), 0, 255)

    paths = {}
    for name, data in (("a", gradient), ("b", noisy.astype("uint8")), ("c", other)):
        paths[name] = tmp_path / f"{name}.png"
        Image.fromarray(data).convert("RGB").save(paths[name])

    a, b, c = (difference_hash(str(paths[n])) for n in "abc")
    assert bin(a ^ b).count("1") <= 6
    assert bin(a ^ c).count("1") > 6
