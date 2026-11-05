"""The held-out split on the real prepare_data: the dataset API faked at the HTTP client."""
import asyncio
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from app.engine.components.checkpoints import (
    MANIFEST_NAME,
    CheckpointManager,
    checkpoint_resumable,
    select_resumable_checkpoint,
)
from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.pipeline import validation_split
from app.engine.models.registry import ModelRegistry

DEFINITION_ID = "flux3-image-klein-base"


class _Pipeline(GenericTrainingPipeline):
    def _setup_family(self) -> None:  # pragma: no cover - never called by the shell
        pass


class _Unavailable(_Pipeline):
    family_name = "unavailable_family"
    validation_timestep_domain = None


class _RecordingLatents:
    def __init__(self):
        self.encoded = []

    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}.safetensors"

    def encode_and_cache_batch(self, tensor, ids, cache_dirs=None, source_paths=None, extra_keys=None):
        self.encoded.extend(ids)
        return tensor


def _image(path, seed):
    pattern = np.random.default_rng(seed).integers(0, 255, (9, 9, 3), dtype=np.uint8)
    Image.fromarray(pattern).resize((64, 64), Image.Resampling.NEAREST).save(path)


def _dataset(root, name, count, first_seed=0, kind="standard", meta=None):
    folder = root / name
    folder.mkdir()
    pairs = []
    for i in range(count):
        media = f"img{i:02d}.png"
        _image(folder / media, first_seed + i)
        pairs.append(
            {
                "media_file": media,
                "caption_content": f"{name} caption {i}",
                "metadata": {"width": 64, "height": 64, "enabled": True, **(meta or {})},
            }
        )
    return {"info": {"path": str(folder), "version": "1.0.0", "kind": kind}, "pairs": pairs}


def _fake_api(monkeypatch, datasets):
    async def _get(_self, url, *args, **kwargs):
        name = url.split("/datasets/")[1].split("/")[0]
        entry = datasets.get(name)
        if entry is None:
            return SimpleNamespace(status_code=404, json=lambda: {})
        body = entry["pairs"] if url.endswith("/pairs") else entry["info"]
        return SimpleNamespace(status_code=200, json=lambda: json.loads(json.dumps(body)))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


def _definition():
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[DEFINITION_ID]


def _pipeline(monkeypatch, datasets, config, cls=_Pipeline):
    t = object.__new__(cls)
    t.definition = _definition()
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "cache_latents": True, **config}
    t._assign_components()
    _fake_api(monkeypatch, datasets)
    return t


def _prepare(t):
    asyncio.run(t.prepare_data())
    return t


def _with_output_dir(t, tmp_path):
    t.checkpoint_manager = SimpleNamespace(output_dir=str(tmp_path))
    return t


def _ids(items):
    return sorted({item["id"] for item in items})


# ── Off changes nothing ──────────────────────────────────────────────────────


def test_off_keeps_the_inventory_and_never_hashes_an_image(monkeypatch, tmp_path):
    def no_hashing(_path):
        raise AssertionError("no image may be hashed with validation off")

    monkeypatch.setattr(validation_split, "difference_hash", no_hashing)
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    t = _prepare(_pipeline(monkeypatch, datasets, {"datasets": [{"dataset_name": "ds"}]}))
    full = list(t.inventory)
    output = tmp_path / "out"
    output.mkdir()
    _with_output_dir(t, output)

    t._enter_validation_split()

    assert len(full) == 20
    assert t.inventory == full
    assert t._validation_split is None
    assert list(output.iterdir()) == []


def test_off_ignores_a_named_validation_dataset(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 3), "val": _dataset(tmp_path, "val", 3, first_seed=50)}
    config = {"datasets": [{"dataset_name": "ds"}], "validation_dataset": "val"}
    t = _prepare(_pipeline(monkeypatch, datasets, config))
    assert {item["dataset_name"] for item in t.inventory} == {"ds"}


# ── The holdout ──────────────────────────────────────────────────────────────


def _holdout_config(**extra):
    return {
        "datasets": [{"dataset_name": "ds", "num_repeats": 3}],
        "validation_enabled": True,
        **extra,
    }


def test_caching_sees_every_item_and_training_only_the_training_items(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    t = _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))
    held = [key.split("/", 1)[1] for key in t._validation_split.holdout_keys]
    assert len(held) == 3  # ceil(0.15 * 20)
    assert _ids(t.inventory) == sorted(f"img{i:02d}.png" for i in range(20))

    _with_output_dir(t, tmp_path)
    t._enter_validation_split()

    assert set(_ids(t.inventory)).isdisjoint(held)
    assert len(t.inventory) == 17 * 3


def test_training_batches_never_hold_a_held_out_image(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    t = _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))
    held = {key.split("/", 1)[1] for key in t._validation_split.holdout_keys}
    _with_output_dir(t, tmp_path)
    t._enter_validation_split()

    batches = t._iter_training_batches(2)
    seen = {item["id"] for _ in range(200) for item in next(batches)}

    assert seen.isdisjoint(held)
    assert len(seen) == 17


def test_a_cold_cache_covers_every_held_out_image(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    t = _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))
    held = [key.split("/", 1)[1] for key in t._validation_split.holdout_keys]

    hints = t._build_caption_hints()
    t.latent_manager = _RecordingLatents()
    t._latent_cache_missing = 20
    t._loop = None
    asyncio.run(t._pre_cache_latents())

    assert set(held) <= set(t.latent_manager.encoded)
    for key in held:
        caption = next(item["caption"] for item in t.inventory if item["id"] == key)
        assert caption in hints


def test_the_split_file_records_the_split(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    config = _holdout_config(validation_split_seed=7, validation_every_n_steps=100, validation_timestep_bands=4)
    t = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, config)), tmp_path)
    t._enter_validation_split()

    written = json.loads((tmp_path / "validation_split.json").read_text(encoding="utf-8"))

    assert written["holdout"] == t._validation_split.holdout_keys
    assert written["split_seed"] == 7
    assert written["hash_threshold_bits"] == validation_split.NEAR_IDENTICAL_BITS
    assert written["every_n_steps"] == 100
    assert written["timestep_bands"] == 4
    assert written["groups"] == t._validation_split.groups
    assert [r["key"] for r in written["representatives"]] == written["holdout"]
    assert {"path", "caption", "target_w", "target_h"} <= set(written["representatives"][0])


def test_a_resume_gets_the_same_split(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    first = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    first._enter_validation_split()
    written = (tmp_path / "validation_split.json").read_text(encoding="utf-8")

    second = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    second._enter_validation_split()
    assert (tmp_path / "validation_split.json").read_text(encoding="utf-8") == written


# ── A separate validation dataset ────────────────────────────────────────────


def test_a_separate_dataset_replaces_the_holdout(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 5), "val": _dataset(tmp_path, "val", 2, first_seed=50)}
    config = {"datasets": [{"dataset_name": "ds"}], "validation_enabled": True, "validation_dataset": "val"}
    t = _prepare(_pipeline(monkeypatch, datasets, config))

    assert {item["dataset_name"] for item in t.inventory} == {"ds", "val"}  # all are cached
    assert t._validation_split.holdout_keys == ["val/img00.png", "val/img01.png"]

    _with_output_dir(t, tmp_path)
    t._enter_validation_split()

    assert {item["dataset_name"] for item in t.inventory} == {"ds"}
    assert len(t.inventory) == 5


def test_the_validation_dataset_also_training_fails_naming_the_file(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 5)}
    config = {"datasets": [{"dataset_name": "ds"}], "validation_enabled": True, "validation_dataset": "ds"}
    with pytest.raises(ValueError, match="img00.png"):
        _prepare(_pipeline(monkeypatch, datasets, config))


def test_a_missing_validation_dataset_fails_naming_it(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 5)}
    config = {"datasets": [{"dataset_name": "ds"}], "validation_enabled": True, "validation_dataset": "gone"}
    with pytest.raises(ValueError, match="gone"):
        _prepare(_pipeline(monkeypatch, datasets, config))


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_a_family_that_declares_none_is_refused_naming_it(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    with pytest.raises(ValueError, match="unavailable.*unavailable_family"):
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config(), cls=_Unavailable))


def test_a_paired_dataset_is_refused_naming_it(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20, kind="edit")}
    with pytest.raises(ValueError, match="ds.*edit"):
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))


def test_a_video_dataset_is_refused(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20, meta={"is_video": True, "fps": 24, "duration_s": 2.0})}
    with pytest.raises(ValueError, match="video"):
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))


def test_an_audio_dataset_is_refused(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20, meta={"is_audio": True, "duration_s": 5.0})}
    with pytest.raises(ValueError, match="audio"):
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config()))


def test_an_adaptive_rebuild_is_refused(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    config = _holdout_config(adaptive_targeting=True, adaptive_targeting_config={"action": "rebuild"})
    with pytest.raises(ValueError, match="rebuild"):
        _prepare(_pipeline(monkeypatch, datasets, config))


def test_adaptive_freeze_is_allowed(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    config = _holdout_config(adaptive_targeting=True, adaptive_targeting_config={"action": "freeze"})
    assert _prepare(_pipeline(monkeypatch, datasets, config))._validation_split is not None


def test_an_adaptive_rebuild_with_validation_off_is_not_refused(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    config = {
        "datasets": [{"dataset_name": "ds"}],
        "adaptive_targeting": True,
        "adaptive_targeting_config": {"action": "rebuild"},
    }
    assert _prepare(_pipeline(monkeypatch, datasets, config))._validation_split is None


# ── The settings ─────────────────────────────────────────────────────────────


def test_the_settings_and_their_bounds():
    from app.engine.models.base import BaseTrainingConfig

    fields = BaseTrainingConfig.model_fields
    assert fields["validation_enabled"].default is False
    assert fields["validation_holdout_fraction"].default == 0.15
    assert fields["validation_split_seed"].default == 0
    assert fields["validation_dataset"].default == ""
    assert fields["validation_every_n_steps"].default == 500
    assert fields["validation_timestep_bands"].default == 8

    extras = {name: fields[name].json_schema_extra for name in fields if name.startswith("validation_")}
    assert (extras["validation_holdout_fraction"]["min"], extras["validation_holdout_fraction"]["max"]) == (0.05, 0.5)
    assert extras["validation_split_seed"]["min"] == 0
    assert extras["validation_every_n_steps"]["min"] == 1
    assert (extras["validation_timestep_bands"]["min"], extras["validation_timestep_bands"]["max"]) == (2, 32)


# ── Round 1 of the code review ───────────────────────────────────────────────


def _written_split(tmp_path):
    return json.loads((tmp_path / "validation_split.json").read_text(encoding="utf-8"))


def test_a_changed_split_stops_the_run_and_keeps_the_record(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    first = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    first._enter_validation_split()
    before = _written_split(tmp_path)

    changed = _with_output_dir(
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config(validation_split_seed=9))), tmp_path
    )
    with pytest.raises(ValueError, match="validation_split.json"):
        changed._enter_validation_split()

    assert _written_split(tmp_path) == before
    assert len(changed.inventory) == 20 * 3  # the run never narrowed


def test_a_changed_interval_is_not_a_changed_split(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    first = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    first._enter_validation_split()
    later = _with_output_dir(
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config(validation_every_n_steps=50))), tmp_path
    )
    later._enter_validation_split()
    assert _written_split(tmp_path)["every_n_steps"] == 50


def test_a_new_training_image_with_the_same_holdout_is_not_a_changed_split(monkeypatch, tmp_path):
    from copy import deepcopy

    from app.engine.core.pipeline.validation_split import ValidationSplit

    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    first = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    first._enter_validation_split()
    held = _written_split(tmp_path)["holdout"]

    later = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    split = later._validation_split
    later._validation_split = ValidationSplit(
        training_items=split.training_items,
        holdout_keys=split.holdout_keys,
        representatives=deepcopy(split.representatives),
        groups=[*split.groups, ["ds/extra.png"]],
    )
    later._enter_validation_split()

    assert _written_split(tmp_path)["holdout"] == held
    assert ["ds/extra.png"] in _written_split(tmp_path)["groups"]


def test_a_valid_validation_dataset_with_no_training_images_fails_naming_it(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 0), "val": _dataset(tmp_path, "val", 2, first_seed=50)}
    config = {"datasets": [{"dataset_name": "ds"}], "validation_enabled": True, "validation_dataset": "val"}
    with pytest.raises(ValueError, match="training"):
        _prepare(_pipeline(monkeypatch, datasets, config))


_OUT_OF_BOUNDS = [
    ("validation_every_n_steps", 0),
    ("validation_every_n_steps", -1),
    ("validation_timestep_bands", 1),
    ("validation_timestep_bands", 33),
    ("validation_holdout_fraction", 0.01),
    ("validation_holdout_fraction", 0.9),
    ("validation_split_seed", -1),
]


@pytest.mark.parametrize("setting, value", _OUT_OF_BOUNDS)
def test_the_schema_rejects_a_value_out_of_bounds(setting, value):
    from pydantic import TypeAdapter, ValidationError

    from app.engine.models.base import BaseTrainingConfig

    field = BaseTrainingConfig.model_fields[setting]
    with pytest.raises(ValidationError):
        TypeAdapter(Annotated[field.annotation, field]).validate_python(value)


@pytest.mark.parametrize("setting, value", _OUT_OF_BOUNDS)
def test_a_raw_config_out_of_bounds_stops_the_job_naming_the_setting(monkeypatch, tmp_path, setting, value):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    with pytest.raises(ValueError, match=setting):
        _prepare(_pipeline(monkeypatch, datasets, _holdout_config(**{setting: value})))


# ── The caption a pass measures with ─────────────────────────────────────────


def _variants(monkeypatch, text):
    """A model-aware caption sidecar for every image, reading ``text[0]`` when asked."""
    from app.core.captioning import caption_variants

    monkeypatch.setattr(caption_variants, "read_variant", lambda *_args, **_kwargs: text[0])


def test_a_representative_keeps_the_model_aware_caption_of_the_start(monkeypatch, tmp_path):
    from app.engine.core.pipeline.caption_selection import select_training_caption

    text = ["first sidecar"]
    _variants(monkeypatch, text)
    t = _prepare(_pipeline(monkeypatch, {"ds": _dataset(tmp_path, "ds", 20)}, _holdout_config()))
    representatives = t._validation_split.representatives
    assert {r["caption"] for r in representatives} == {"first sidecar"}

    text[0] = "edited during the run"

    assert {select_training_caption(r, DEFINITION_ID) for r in representatives} == {"first sidecar"}


def test_freezing_the_caption_leaves_the_training_items_on_their_own_captions(monkeypatch, tmp_path):
    from app.engine.core.pipeline.caption_selection import select_training_caption

    text = ["first sidecar"]
    _variants(monkeypatch, text)
    t = _prepare(_pipeline(monkeypatch, {"ds": _dataset(tmp_path, "ds", 20)}, _holdout_config()))
    text[0] = "edited during the run"

    training = t._validation_split.training_items[0]
    assert training.get("use_model_aware_captions", True) is True
    assert select_training_caption(training, DEFINITION_ID) == "edited during the run"


def test_a_resume_after_a_sidecar_edit_stops_naming_the_split_file(monkeypatch, tmp_path):
    text = ["first sidecar"]
    _variants(monkeypatch, text)
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    first = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)
    first._enter_validation_split()

    text[0] = "edited before the resume"
    resumed = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path)

    with pytest.raises(ValueError, match="validation_split.json"):
        resumed._enter_validation_split()


def test_a_dataset_that_trains_without_captions_measures_without_them(monkeypatch, tmp_path):
    from app.engine.core.pipeline.caption_selection import select_training_caption

    _variants(monkeypatch, ["a sidecar"])
    t = _prepare(_pipeline(monkeypatch, {"ds": _dataset(tmp_path, "ds", 20)}, _holdout_config()))
    representative = dict(t._validation_split.representatives[0], use_captions=False)

    assert select_training_caption(representative, DEFINITION_ID) == ""


@pytest.mark.parametrize(
    "first, second",
    [
        ({"global_triggerword": "one"}, {"global_triggerword": "two"}),
        ({"datasets": [{"dataset_name": "ds", "caption_prefix": "one"}]},
         {"datasets": [{"dataset_name": "ds", "caption_prefix": "two"}]}),
    ],
    ids=["trigger word", "caption prefix"],
)
def test_a_resume_with_another_caption_setting_stops_naming_the_split_file(monkeypatch, tmp_path, first, second):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    earlier = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config(**first))), tmp_path)
    earlier._enter_validation_split()
    before = _written_split(tmp_path)

    resumed = _with_output_dir(_prepare(_pipeline(monkeypatch, datasets, _holdout_config(**second))), tmp_path)

    with pytest.raises(ValueError, match="validation_split.json"):
        resumed._enter_validation_split()
    assert _written_split(tmp_path) == before


# ── The split travels with the checkpoint ────────────────────────────────────


def _checkpoint_of(t, run_dir, **save):
    """A real checkpoint folder of ``t``, saved after the run entered its split."""
    t.checkpoint_manager = CheckpointManager(str(run_dir))
    t._enter_validation_split()
    return t.checkpoint_manager.save_checkpoint(step=100, components={}, config=t.config, **save)


def _resuming(monkeypatch, datasets, checkpoint, output, **config):
    """A run resuming ``checkpoint`` into ``output``, which holds nothing yet."""
    t = _prepare(_pipeline(monkeypatch, datasets, _holdout_config(resume_from_checkpoint=checkpoint, **config)))
    t.checkpoint_manager = CheckpointManager(str(output))
    return t


def _imported(monkeypatch, checkpoint, root):
    """``checkpoint`` taken through the real ZIP export and unpacked elsewhere."""
    from app.api.training import job_routes

    source = Path(checkpoint)
    monkeypatch.setattr(job_routes, "job_manager", SimpleNamespace(get_job=lambda _job_id: object()))
    monkeypatch.setattr(job_routes, "_resolve_run_dir", lambda _job: source.parent)
    response = asyncio.run(job_routes.download_checkpoint_zip("job", source.name))
    target = root / "imported" / source.name
    with zipfile.ZipFile(response.path) as archive:
        archive.extractall(target)
    os.remove(response.path)
    return str(target)


def _assert_refused(resumed, output):
    with pytest.raises(ValueError, match="validation_split.json"):
        resumed._enter_validation_split()
    assert not (output / "validation_split.json").exists()
    assert resumed.checkpoint_manager.validation_split_content is None
    assert len(resumed.inventory) > len(resumed._validation_split.training_items)  # never narrowed


def _changed_seed(_root):
    return lambda datasets: datasets, {"validation_split_seed": 9}


def _changed_dataset(root):
    (root / "later").mkdir()
    grown = _dataset(root / "later", "ds", 40)
    return lambda _datasets: {"ds": grown}, {}


_CHANGES = pytest.mark.parametrize("change", [_changed_seed, _changed_dataset], ids=["split seed", "dataset"])


@_CHANGES
def test_a_resume_under_another_name_with_a_changed_split_stops(monkeypatch, tmp_path, change):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    later, config = change(tmp_path)
    renamed = tmp_path / "renamed"

    _assert_refused(_resuming(monkeypatch, later(datasets), checkpoint, renamed, **config), renamed)


@_CHANGES
def test_an_imported_checkpoint_with_a_changed_split_stops(monkeypatch, tmp_path, change):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    imported = _imported(monkeypatch, checkpoint, tmp_path)
    later, config = change(tmp_path)
    fresh = tmp_path / "fresh"

    _assert_refused(_resuming(monkeypatch, later(datasets), imported, fresh, **config), fresh)


def test_a_resume_under_another_name_with_the_same_split_goes_on_and_stays_bound(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    renamed = tmp_path / "renamed"
    resumed = _resuming(monkeypatch, datasets, checkpoint, renamed)

    next_checkpoint = _checkpoint_of(resumed, renamed)

    assert len(resumed.inventory) == 17 * 3
    held = json.loads((Path(checkpoint) / "validation_split.json").read_text(encoding="utf-8"))["holdout"]
    assert _written_split(renamed)["holdout"] == held
    assert _written_split(Path(next_checkpoint))["holdout"] == held


def test_an_imported_checkpoint_with_the_same_split_goes_on(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    imported = _imported(monkeypatch, checkpoint, tmp_path)
    assert checkpoint_resumable(imported) == (True, "")
    fresh = tmp_path / "fresh"
    resumed = _resuming(monkeypatch, datasets, imported, fresh)

    resumed._enter_validation_split()

    assert len(resumed.inventory) == 17 * 3


def test_the_same_images_in_another_folder_are_the_same_split(monkeypatch, tmp_path):
    """A checkpoint moves to another machine, where the dataset lies elsewhere."""
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    imported = _imported(monkeypatch, checkpoint, tmp_path)
    (tmp_path / "there").mkdir()
    moved = {"ds": _dataset(tmp_path / "there", "ds", 20)}
    fresh = tmp_path / "fresh"
    resumed = _resuming(monkeypatch, moved, imported, fresh)

    resumed._enter_validation_split()

    assert len(resumed.inventory) == 17 * 3


def test_a_moved_dataset_with_another_picture_under_a_held_out_name_is_another_split(monkeypatch, tmp_path):
    """Same names and captions, but a picture the checkpoint trained on is now measured."""
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    split = _written_split(Path(checkpoint))
    imported = _imported(monkeypatch, checkpoint, tmp_path)
    (tmp_path / "there").mkdir()
    moved = {"ds": _dataset(tmp_path / "there", "ds", 20)}
    folder = Path(moved["ds"]["info"]["path"])
    held = split["representatives"][0]["key"].split("/", 1)[1]
    trained = next(p["media_file"] for p in moved["ds"]["pairs"] if f"ds/{p['media_file']}" not in split["holdout"])
    held_bytes, trained_bytes = (folder / held).read_bytes(), (folder / trained).read_bytes()
    (folder / held).write_bytes(trained_bytes)
    (folder / trained).write_bytes(held_bytes)
    fresh = tmp_path / "fresh"
    resumed = _resuming(monkeypatch, moved, imported, fresh)
    assert sorted(resumed._validation_split.holdout_keys) == sorted(split["holdout"])  # only the pictures differ

    _assert_refused(resumed, fresh)


def test_another_caption_on_a_held_out_image_is_still_another_split(monkeypatch, tmp_path):
    """The folder doesn't count; which image is measured, and with what caption, does."""
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    held = _written_split(Path(checkpoint))["representatives"][0]["key"].split("/", 1)[1]
    for pair in datasets["ds"]["pairs"]:
        if pair["media_file"] == held:
            pair["caption_content"] = "another caption"
    fresh = tmp_path / "fresh"

    _assert_refused(_resuming(monkeypatch, datasets, checkpoint, fresh), fresh)


def test_a_checkpoint_from_before_validation_resumes_as_before(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    off = {"datasets": [{"dataset_name": "ds"}]}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, off)), tmp_path / "first")
    assert not (Path(checkpoint) / "validation_split.json").exists()
    fresh = tmp_path / "fresh"
    resumed = _resuming(monkeypatch, datasets, checkpoint, fresh, validation_split_seed=9)

    resumed._enter_validation_split()

    assert len(resumed.inventory) == 17 * 3
    assert _written_split(fresh)["split_seed"] == 9


def test_validation_off_writes_no_split_into_a_checkpoint(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    off = {"datasets": [{"dataset_name": "ds"}]}
    run = tmp_path / "first"

    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, off)), run, is_final=True)

    assert checkpoint_resumable(checkpoint) == (True, "")
    assert list(run.rglob("validation_split.json")) == []


def test_validation_off_resumes_a_bound_checkpoint_and_writes_nothing(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")
    off = {"datasets": [{"dataset_name": "ds"}], "resume_from_checkpoint": checkpoint, "validation_split_seed": 9}
    renamed = tmp_path / "renamed"

    next_checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, off)), renamed)

    assert checkpoint_resumable(next_checkpoint) == (True, "")
    assert list(renamed.rglob("validation_split.json")) == []


def test_every_folder_of_a_final_save_carries_the_split_and_lists_it(monkeypatch, tmp_path):
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    run = tmp_path / "first"
    _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), run, is_final=True)

    for folder in (run / "final", run / "checkpoint-000100"):
        files = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
        assert files["validation_split.json"] == (folder / "validation_split.json").stat().st_size
        assert checkpoint_resumable(folder) == (True, "")
    best, skipped = select_resumable_checkpoint(run)
    assert best[0] == "final" and skipped == []


def test_a_checkpoint_stripped_of_its_split_isnt_resumable(monkeypatch, tmp_path):
    """Deleting the file can't be the way past the guard."""
    datasets = {"ds": _dataset(tmp_path, "ds", 20)}
    checkpoint = _checkpoint_of(_prepare(_pipeline(monkeypatch, datasets, _holdout_config())), tmp_path / "first")

    (Path(checkpoint) / "validation_split.json").unlink()

    assert checkpoint_resumable(checkpoint) == (False, "validation_split.json is missing")
