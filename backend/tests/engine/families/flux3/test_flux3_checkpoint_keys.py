"""FLUX.3 Image Klein (LANE-144): the REAL checkpoint's keys against the vendored class, the YAML and the saver.

The contract tests (``test_flux3_contracts.py``) hold the family to the PR defaults on a tiny random model;
nothing there reads the released weights. This module reads the locally cached snapshot of
``black-forest-labs/flux-3-image-klein-base`` -- header and index only, never a tensor, never a download --
and asserts:

1. the transformer's tensor names equal the vendored ``Flux3Transformer2DModel``'s ``state_dict()`` keys
   (built from the checkpoint's own ``transformer/config.json`` on the meta device), every shape equal;
2. ``transformer/config.json`` equals the committed ``klein_base.yaml`` ``architecture_params``
   ``transformer.*`` entries, and the checkpoint is not a flow map;
3. each of the YAML's 228 LoRA targets (BFL-native names) is the saver's native name of a module whose
   weight the checkpoint stores (the checkpoint stores diffusers names).

The snapshot is absent on CI and on any host that has not accepted the gated licence: the module SKIPS
there with the reason. Memory stays small: the model is built on ``torch.device("meta")``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

REPO_ID = "black-forest-labs/flux-3-image-klein-base"
# The snapshot the definition YAML's comment names as the source of its architecture_params.
REVISION = "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4"
MARK = "[flux3-checkpoint-keys]"

FAMILY_DIR = Path(__file__).resolve().parents[4] / "app" / "engine" / "models" / "families" / "flux3"
DEFINITION = FAMILY_DIR / "definitions" / "klein_base.yaml"
# ``__init__`` bookkeeping keys diffusers writes into every config.json; they are not constructor args.
_CONFIG_META = ("_class_name", "_diffusers_version", "_name_or_path")


def _snapshot_root() -> Path:
    """The cached snapshot directory, resolved by huggingface_hub's own cache lookup -- never a download."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(REPO_ID, "transformer/config.json", revision=REVISION)
    if not isinstance(hit, str) or not Path(hit).is_file():
        pytest.skip(f"{MARK}: {REPO_ID}@{REVISION} is not in the local HF cache (gated weights; absent on CI)")
    return Path(hit).parent.parent


@pytest.fixture(scope="module")
def snapshot() -> Path:
    return _snapshot_root()


@pytest.fixture(scope="module")
def config(snapshot: Path) -> dict:
    return json.loads((snapshot / "transformer" / "config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checkpoint_shapes(snapshot: Path) -> dict[str, tuple[int, ...]]:
    """Every transformer tensor name -> shape, read from the safetensors HEADERS only."""
    from safetensors import safe_open

    tdir = snapshot / "transformer"
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = sorted(p.name for p in tdir.glob("*.safetensors"))
    assert shards, f"{MARK}: no safetensors files under {tdir}"
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(str(tdir / shard), framework="pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def _committed_definition() -> dict:
    """``klein_base.yaml`` as committed at HEAD (the working copy may carry an uncommitted enrichment)."""
    rel = DEFINITION.relative_to(FAMILY_DIR.parents[4]).as_posix()
    try:
        text = subprocess.run(
            ["git", "show", f"HEAD:backend/{rel}"], cwd=FAMILY_DIR, capture_output=True, text=True,
            encoding="utf-8", check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        text = DEFINITION.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def test_checkpoint_keys_and_shapes_equal_the_vendored_class(config, checkpoint_shapes):
    from app.engine.models.families.flux3.vendor import Flux3Transformer2DModel

    kwargs = {k: v for k, v in config.items() if k not in _CONFIG_META}
    with torch.device("meta"):
        model = Flux3Transformer2DModel(**kwargs)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    missing = sorted(set(expected) - set(checkpoint_shapes))
    unexpected = sorted(set(checkpoint_shapes) - set(expected))
    assert not missing and not unexpected, (
        f"{MARK}: checkpoint has {len(checkpoint_shapes)} tensors, the vendored class {len(expected)}; "
        f"missing from the checkpoint {missing[:6]}, unexpected in it {unexpected[:6]}")
    wrong = {k: (checkpoint_shapes[k], s) for k, s in expected.items() if checkpoint_shapes[k] != s}
    assert not wrong, f"{MARK}: {len(wrong)} of {len(expected)} shapes differ (checkpoint, class): {list(wrong.items())[:4]}"


def test_config_equals_the_committed_definition_and_is_not_a_flow_map(config):
    assert not config.get("flow_map", False), f"{MARK}: transformer/config.json sets flow_map: true"
    params = _committed_definition().get("architecture_params") or {}
    declared = {k.removeprefix("transformer."): v for k, v in params.items() if k.startswith("transformer.")}
    assert declared, f"{MARK}: the committed definition declares no transformer.* architecture_params"

    # Not stored in config.json: the inner width is heads x head_dim; flow_map defaults to False.
    derived = dict(config)
    derived["hidden_size"] = config["num_attention_heads"] * config["attention_head_dim"]
    derived.setdefault("flow_map", False)
    diff = {k: (v, derived.get(k, "<absent>")) for k, v in declared.items() if derived.get(k, "<absent>") != v}
    assert not diff, f"{MARK}: {len(diff)} of {len(declared)} transformer.* entries differ (yaml, config): {diff}"


def test_every_lora_target_names_a_checkpoint_module(checkpoint_shapes):
    from app.engine.models.families.flux3.saver import diffusers_to_native_module

    targets = list(_committed_definition().get("lora_targetable_modules") or [])
    assert len(targets) == len(set(targets)) == 228, f"{MARK}: the committed YAML lists {len(targets)} targets"

    native_of_checkpoint: dict[str, str] = {}
    for key in checkpoint_shapes:
        if not key.endswith(".weight"):
            continue
        module = key.removesuffix(".weight")
        try:
            native_of_checkpoint[diffusers_to_native_module(module)] = module
        except ValueError:
            continue  # a module the saver never writes (norms, embedders, ...): not a LoRA target
    absent = sorted(t for t in targets if t not in native_of_checkpoint)
    assert not absent, (
        f"{MARK}: {len(absent)} of {len(targets)} LoRA targets name no tensor among the checkpoint's "
        f"{len(checkpoint_shapes)} keys ({len(native_of_checkpoint)} saver-mappable modules): {absent[:6]}")
