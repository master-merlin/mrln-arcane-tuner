"""Cross-family trainer→driver seam contract (B-TEST-6).

The historical bug class (only ever regression-tested for krea2, see
``backend/app/engine/tests/test_krea2_family.py``): a family trainer's
overrides — ``_update_primary_model`` (must sync the DRIVER's primary-model
reference, which the base method does NOT do), the ``encode_text`` return-shape
contract, and the primary-model alias — silently drift and unit mocks hide it
until GPU UAT.

This pins those contracts for every remaining override-carrying family with a
PARAMETRIZED real-seam test: the REAL trainer method runs against the REAL
driver (real classes, real bound methods); the only stubs are leaf models /
pre-seeded caches — nothing mocks the trainer→driver boundary under test.

Per-family capability table below (``FAMILIES``): each entry declares which
driver attribute holds the primary model, the trainer alias name, and the
documented ``encode_text`` return contract (or ``None`` where the family does
not override encode — hidream_o1 uses the base encode path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from app.engine.core.text_encoding import TextEncoderOutput


# ── Leaf stub model (a real nn.Module, never a mock) ──────────────────────
class _Stub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tag = nn.Parameter(torch.zeros(1))


_DT = torch.float32
_D = 8   # feature dim
_L = 4   # sequence length
_P = 6   # pooled dim
_CAPS = ["a caption", "another caption"]  # B = 2


# ── encode_text cache seeders (one per documented contract) ───────────────
def _seed_tuple_emb_mask(caps: list[str]) -> dict[str, Any]:
    """(emb[L,D], mask[L]) — ernie / ideogram."""
    return {c: (torch.randn(_L, _D), torch.ones(_L, dtype=torch.long)) for c in caps}


def _seed_tuple_layered(caps: list[str]) -> dict[str, Any]:
    """(feat[4,L,D], mask[L]) — microsoft_lens (extra leading layer dim)."""
    return {c: (torch.randn(4, _L, _D), torch.ones(_L, dtype=torch.long)) for c in caps}


def _seed_tensor(caps: list[str]) -> dict[str, Any]:
    """[1,L,D] raw tensor — flux2 / wan21 / wan22."""
    return {c: torch.randn(1, _L, _D) for c in caps}


def _seed_teo_triple(caps: list[str]) -> dict[str, Any]:
    """(emb[1,L,D], pooled[1,P], mask[1,L]) — ltx2 TextEncoderOutput."""
    return {
        c: (torch.randn(1, _L, _D), torch.randn(1, _P), torch.ones(1, _L, dtype=torch.long))
        for c in caps
    }


# ── encode_text return-contract checks ────────────────────────────────────
def _check_tuple2_3d(out: Any) -> None:
    assert isinstance(out, tuple) and len(out) == 2, f"expected 2-tuple, got {type(out)}"
    emb, mask = out
    assert emb.ndim == 3 and emb.shape[0] == 2, f"emb must be [B,S,D], got {tuple(emb.shape)}"
    assert mask.ndim == 2 and mask.shape[0] == 2


def _check_tuple2_4d(out: Any) -> None:
    assert isinstance(out, tuple) and len(out) == 2, f"expected 2-tuple, got {type(out)}"
    emb, mask = out
    assert emb.ndim == 4 and emb.shape[0] == 2, f"emb must be [B,4,S,D], got {tuple(emb.shape)}"
    assert mask.ndim == 2 and mask.shape[0] == 2


def _check_tensor_3d(out: Any) -> None:
    assert isinstance(out, torch.Tensor), f"expected raw Tensor, got {type(out)}"
    assert out.ndim == 3 and out.shape[0] == 2, f"expected [B,L,D], got {tuple(out.shape)}"


def _check_teo(out: Any) -> None:
    assert isinstance(out, TextEncoderOutput), f"expected TextEncoderOutput, got {type(out)}"
    assert out.embeddings.shape[0] == 2
    assert out.attention_mask is not None and out.pooled is not None


@dataclass
class FamilySpec:
    id: str
    trainer_path: str          # "module:ClassName"
    driver_path: str
    driver_primary_attr: str   # attr get_primary_model reads / _update writes on driver
    trainer_alias: str         # trainer-side alias attr
    expert_slots: bool = False  # wan22: also mirrors onto the active-expert slot
    encode_kind: str | None = None
    encode_seed: Callable[[list[str]], dict[str, Any]] | None = None
    encode_check: Callable[[Any], None] | None = None
    encode_extra: dict[str, Any] = field(default_factory=dict)


FAMILIES: list[FamilySpec] = [
    FamilySpec(
        "ernie_image",
        "app.engine.models.families.ernie_image.trainer:ErnieImageTrainer",
        "app.engine.models.families.ernie_image.driver:ErnieImageDriver",
        "transformer", "transformer",
        encode_kind="tuple2", encode_seed=_seed_tuple_emb_mask, encode_check=_check_tuple2_3d,
    ),
    FamilySpec(
        "flux2",
        "app.engine.models.families.flux2.trainer:Flux2Trainer",
        "app.engine.models.families.flux2.driver:Flux2Driver",
        "transformer", "transformer",
        encode_kind="tensor", encode_seed=_seed_tensor, encode_check=_check_tensor_3d,
    ),
    FamilySpec(
        "ideogram4",
        "app.engine.models.families.ideogram4.trainer:IdeogramV4Trainer",
        "app.engine.models.families.ideogram4.driver:IdeogramV4Driver",
        "transformer", "transformer",
        encode_kind="tuple2", encode_seed=_seed_tuple_emb_mask, encode_check=_check_tuple2_3d,
    ),
    FamilySpec(
        "ltx2",
        "app.engine.models.families.ltx2.trainer:Ltx2Trainer",
        "app.engine.models.families.ltx2.driver:Ltx2Driver",
        "transformer", "transformer",
        encode_kind="teo", encode_seed=_seed_teo_triple, encode_check=_check_teo,
    ),
    FamilySpec(
        "microsoft_lens",
        "app.engine.models.families.microsoft_lens.trainer:MicrosoftLensTrainer",
        "app.engine.models.families.microsoft_lens.driver:MicrosoftLensDriver",
        "transformer", "transformer",
        encode_kind="tuple2_layered", encode_seed=_seed_tuple_layered, encode_check=_check_tuple2_4d,
    ),
    FamilySpec(
        "hidream_o1",
        "app.engine.models.families.hidream_o1.trainer:HiDreamO1Trainer",
        "app.engine.models.families.hidream_o1.driver:HiDreamO1Driver",
        "model", "model",
        # No encode_text override — uses the base encode path (nothing to pin here).
        encode_kind=None,
    ),
    FamilySpec(
        "wan21",
        "app.engine.models.families.wan21.trainer:Wan21Trainer",
        "app.engine.models.families.wan21.driver:Wan21Driver",
        "transformer", "transformer",
        encode_kind="tensor", encode_seed=_seed_tensor, encode_check=_check_tensor_3d,
    ),
    FamilySpec(
        "wan22",
        "app.engine.models.families.wan22.trainer:Wan22Trainer",
        "app.engine.models.families.wan22.driver:Wan22Driver",
        "transformer", "transformer", expert_slots=True,
        encode_kind="tensor", encode_seed=_seed_tensor, encode_check=_check_tensor_3d,
    ),
]

_IDS = [f.id for f in FAMILIES]


def _load(path: str) -> type:
    mod_name, cls_name = path.split(":")
    import importlib
    return getattr(importlib.import_module(mod_name), cls_name)


def _make_driver(spec: FamilySpec, primary: nn.Module):
    """Real driver instance (methods are real) pre-loaded with ``primary`` in
    its primary-model slot — the state after ``assign_components``."""
    DriverCls = _load(spec.driver_path)
    drv = object.__new__(DriverCls)  # real class, skip heavy __init__
    setattr(drv, spec.driver_primary_attr, primary)
    if spec.expert_slots:  # wan22 active-expert bookkeeping
        drv.transformer_high = primary
        drv.transformer_low = None
        drv._active_expert = "high"
    return drv


def _make_trainer(spec: FamilySpec, driver, primary: nn.Module):
    TrainerCls = _load(spec.trainer_path)
    t = object.__new__(TrainerCls)
    t.device = torch.device("cpu")
    t.driver = driver
    t.components = {"unet": primary}
    setattr(t, spec.trainer_alias, primary)  # alias set live by _assign_components
    return t


# ── Aspect 1 + 3: _update_primary_model syncs the driver + the alias ──────
@pytest.mark.parametrize("spec", FAMILIES, ids=_IDS)
def test_update_primary_model_syncs_driver_and_alias(spec: FamilySpec):
    TrainerCls = _load(spec.trainer_path)
    loaded, wrapped = _Stub(), _Stub()
    drv = _make_driver(spec, loaded)
    t = _make_trainer(spec, drv, loaded)

    # Pre-state: the alias the pipeline assigned resolves to the loaded model.
    assert drv.get_primary_model() is loaded

    # Run the REAL override (PEFT/quant wrapping hands over a new module).
    TrainerCls._update_primary_model(t, wrapped)

    # Driver's primary reference must follow — the base method does NOT do this,
    # so a family that forgets to override strands the driver on the old graph.
    assert drv.get_primary_model() is wrapped, (
        f"{spec.id}: driver primary not synced after _update_primary_model"
    )
    # Alias + components dict resolve to the SAME wrapped object.
    assert getattr(t, spec.trainer_alias) is wrapped
    assert t.components["unet"] is wrapped
    if spec.expert_slots:  # wan22: active-expert slot must also flip
        assert drv.transformer_high is wrapped


# ── Aspect 2: encode_text returns the family's documented contract shape ───
@pytest.mark.parametrize(
    "spec", [f for f in FAMILIES if f.encode_kind], ids=[f.id for f in FAMILIES if f.encode_kind]
)
def test_encode_returns_documented_contract(spec: FamilySpec):
    TrainerCls = _load(spec.trainer_path)
    t = object.__new__(TrainerCls)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.text_cache = spec.encode_seed(_CAPS)
    t.text_encoder = None                       # untouched: every caption pre-cached
    t.driver = SimpleNamespace(text_encoder=None)

    out = TrainerCls.encode_text(t, list(_CAPS), _DT)
    spec.encode_check(out)


def test_all_override_families_covered():
    """Guard: the six families the brief names plus wan21/wan22 are all here."""
    required = {
        "ernie_image", "hidream_o1", "ideogram4", "flux2", "ltx2",
        "microsoft_lens", "wan21", "wan22",
    }
    assert required <= set(_IDS)
