"""``unload_text_encoder=True`` must actually release the text encoder (W2.T5).

THE BUG
=======
``unload_text_encoder=True`` is advertised as the max-VRAM-savings mode. The
pre-fix unload branch of ``PipelineLoadingMixin._offload_text_encoders``
(``pipeline_loading.py`` ~310-324) only:

* popped TE entries from ``self.components``, and
* nulled the TRAINER's own attributes (``setattr(self, name, None)``)

but never touched the DRIVER's references. ``_assign_components`` wires the
driver as the single owner of component state
(``self.driver.assign_components(self.components)``), and
``get_text_encoders()`` reads DIRECTLY off driver attributes -- so the driver
kept reporting the "unloaded" encoder as present.  Worse, the unload branch
never called ``.to("cpu")`` on the encoder(s) first (unlike the offload
branch), so at the moment this runs the TE is GPU-resident
(``run_trainer.py`` moves TEs to GPU before caching, then calls
``_offload_text_encoders()`` right after) -- "unload" dropped Python
references to a still-CUDA-resident module while the driver kept it alive,
so nothing was ever freed.

THE FIX
=======
``IModelDriver.release_text_encoders()`` (new hook; base = warn-if-
non-empty safety net so a family that forgets to override is VISIBLE in
logs, not silently leaking). The unload branch now moves every TE returned
by ``get_text_encoders()`` to CPU FIRST, then calls
``driver.release_text_encoders()``. Each family driver overrides the hook
to null EXACTLY the attribute(s) its own ``get_text_encoders()`` reads --
no more, no less (verified per-family below; a wrong-attribute override
would pass a naive "still returns something" check but leak in production).
"""

from __future__ import annotations

import importlib

import pytest
import structlog
import torch

from app.engine.core.interfaces import IModelDriver
from app.engine.models.families.qwen_image.driver import QwenImageDriver
from app.engine.models.families.qwen_image.trainer import QwenImageTrainer
from app.engine.models.families.sdxl.driver import SDXLDriver


# ---------------------------------------------------------------------------
# Test double: records every ``.to()`` call AND whether the driver still
# held the reference at call-time -- proves the CPU move happens BEFORE the
# reference is dropped (dropping first is exactly the bug: a GPU-resident
# module with no remaining Python ref is inert to a later CPU move).
# ---------------------------------------------------------------------------


class _RecordingTE:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.attached_at_move: bool | None = None
        self._owner_driver = None
        self._owner_attr = ""

    def bind_owner(self, driver, attr: str) -> None:
        self._owner_driver = driver
        self._owner_attr = attr

    def to(self, device):
        self.to_calls.append(str(device))
        if self._owner_driver is not None:
            self.attached_at_move = (
                getattr(self._owner_driver, self._owner_attr, None) is self
            )
        return self


# ---------------------------------------------------------------------------
# Step 1/2 (brief): the end-to-end unload-branch test, through the REAL
# pipeline mixin method (not a re-implementation of it).
# ---------------------------------------------------------------------------


def _wired_qwen_trainer(te: _RecordingTE) -> QwenImageTrainer:
    driver = object.__new__(QwenImageDriver)
    driver.text_encoder = te
    te.bind_owner(driver, "text_encoder")

    trainer = object.__new__(QwenImageTrainer)
    trainer.driver = driver
    trainer.components = {"text_encoder": te}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": True}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.logger = structlog.get_logger("test")
    return trainer


def test_unload_branch_releases_driver_refs():
    te = _RecordingTE()
    trainer = _wired_qwen_trainer(te)

    trainer._offload_text_encoders()

    # (a) the driver no longer reports the TE.
    assert trainer.driver.get_text_encoders() == {}
    assert trainer.driver.text_encoder is None
    # components dict cleanup (pre-existing behavior) still holds.
    assert trainer.components == {}
    assert trainer._te_unloaded is True


def test_unload_branch_moves_to_cpu_before_dropping_reference():
    te = _RecordingTE()
    trainer = _wired_qwen_trainer(te)

    trainer._offload_text_encoders()

    assert te.to_calls == ["cpu"], "TE must be moved to CPU exactly once"
    assert te.attached_at_move is True, (
        "the driver must still hold the reference at the moment .to('cpu') "
        "is called -- moving AFTER the reference is dropped frees nothing "
        "for a GPU-resident module with no other strong ref"
    )


def test_offload_branch_unaffected_te_stays_assigned():
    """Regression guard: the (non-buggy) offload path must keep behaving —
    TE moved to CPU, kept assigned on both trainer and driver (needed for
    later re-encode / phased sampling), never released."""
    te = _RecordingTE()
    driver = object.__new__(QwenImageDriver)
    driver.text_encoder = te
    te.bind_owner(driver, "text_encoder")

    trainer = object.__new__(QwenImageTrainer)
    trainer.driver = driver
    trainer.components = {"text_encoder": te}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": False}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.logger = structlog.get_logger("test")

    trainer._offload_text_encoders()

    assert te.to_calls == ["cpu"]
    assert driver.text_encoder is te, "offload must NOT release the driver ref"
    assert driver.get_text_encoders() == {"text_encoder": te}


# ---------------------------------------------------------------------------
# The SDXL None-filter fix: get_text_encoders() must filter None entries.
# Without this fix, releasing both text_encoder_1/2 leaves
# ``{"text_encoder_1": None, "text_encoder_2": None}`` -- NOT an empty dict
# -- which both breaks the "driver no longer reports the TE" contract and
# would crash any consumer doing ``next(te.parameters())`` over the values.
# ---------------------------------------------------------------------------


def test_sdxl_get_text_encoders_filters_none():
    driver = object.__new__(SDXLDriver)
    driver.text_encoder_1 = None
    driver.text_encoder_2 = None
    assert driver.get_text_encoders() == {}


def test_sdxl_release_leaves_get_text_encoders_empty():
    driver = object.__new__(SDXLDriver)
    te1, te2 = _RecordingTE(), _RecordingTE()
    driver.text_encoder_1 = te1
    driver.text_encoder_2 = te2

    driver.release_text_encoders()

    assert driver.text_encoder_1 is None
    assert driver.text_encoder_2 is None
    assert driver.get_text_encoders() == {}


# ---------------------------------------------------------------------------
# Base-hook safety net: a driver that forgets to override
# release_text_encoders() must be VISIBLE in logs, not silently leaking.
# ---------------------------------------------------------------------------


class _StubDriverNoOverride(IModelDriver):
    """Minimal concrete driver that does NOT override release_text_encoders."""

    def __init__(self) -> None:
        self.text_encoder = None

    def assign_components(self, components):  # pragma: no cover - unused
        raise NotImplementedError

    def get_components(self):  # pragma: no cover - unused
        return {}

    def get_primary_model(self):  # pragma: no cover - unused
        return None

    def get_text_encoders(self):
        return (
            {"text_encoder": self.text_encoder} if self.text_encoder is not None else {}
        )

    def get_lora_targets(self):  # pragma: no cover - unused
        return []

    def init_scheduler(self):  # pragma: no cover - unused
        return None

    def resolve_loading_dtype(self):  # pragma: no cover - unused
        return torch.float32

    def encode_text(self, captions, dtype):  # pragma: no cover - unused
        return None

    def get_te_lora_targets(self):  # pragma: no cover - unused
        return []

    def forward_pass(
        self, noisy_input, timesteps, text_embeddings, batch
    ):  # pragma: no cover
        return None

    def get_saver(self):  # pragma: no cover - unused
        return None


def test_base_release_warns_when_family_forgot_override():
    from structlog.testing import capture_logs

    driver = _StubDriverNoOverride()
    driver.text_encoder = object()

    with capture_logs() as logs:
        driver.release_text_encoders()

    # The base is a NO-OP by design (it doesn't know the family's attrs) --
    # so it must NOT silently claim success.
    assert driver.get_text_encoders() != {}
    assert any(
        e.get("event") == "release_text_encoders_not_overridden" for e in logs
    ), "a forgotten override must be visible in logs, not a silent no-op"


# ---------------------------------------------------------------------------
# All-family sweep: every driver with TE attrs nulls EXACTLY what its own
# get_text_encoders() reads -- no more, no less. Table mirrors the manual
# enumeration via `git grep -n "def get_text_encoders"
# backend/app/engine/models/families` (see W2.T5 report for the full list).
# ---------------------------------------------------------------------------

_FAMILY_TE_ATTRS: list[tuple[str, str, str, list[str]]] = [
    (
        "ace_step15",
        "app.engine.models.families.ace_step15.driver",
        "AceStep15Driver",
        ["text_encoder", "condition_encoder"],
    ),
    (
        "boogu_image",
        "app.engine.models.families.boogu_image.driver",
        "BooguImageDriver",
        ["text_encoder"],
    ),
    (
        "chroma",
        "app.engine.models.families.chroma.driver",
        "ChromaDriver",
        ["text_encoder"],
    ),
    (
        "dreamlite",
        "app.engine.models.families.dreamlite.driver",
        "DreamLiteDriver",
        ["text_encoder"],
    ),
    (
        "ernie_image",
        "app.engine.models.families.ernie_image.driver",
        "ErnieImageDriver",
        ["text_encoder"],
    ),
    (
        "flux1",
        "app.engine.models.families.flux1.driver",
        "Flux1Driver",
        ["clip_encoder", "t5_encoder"],
    ),
    (
        "flux2",
        "app.engine.models.families.flux2.driver",
        "Flux2Driver",
        ["text_encoder"],
    ),
    (
        "hidream_o1",
        "app.engine.models.families.hidream_o1.driver",
        "HiDreamO1Driver",
        [],
    ),  # unified model -- no standalone TE at all
    (
        "hunyuan_video15",
        "app.engine.models.families.hunyuan_video15.driver",
        "Hv15Driver",
        ["text_encoder", "text_encoder_2"],
    ),
    (
        "ideogram4",
        "app.engine.models.families.ideogram4.driver",
        "IdeogramV4Driver",
        ["text_encoder"],
    ),
    (
        "kandinsky5",
        "app.engine.models.families.kandinsky5.driver",
        "Kandinsky5Driver",
        ["text_encoder", "text_encoder_2"],
    ),
    (
        "krea2",
        "app.engine.models.families.krea2.driver",
        "Krea2Driver",
        ["text_encoder"],
    ),
    (
        "longcat_image",
        "app.engine.models.families.longcat_image.driver",
        "LongCatImageDriver",
        ["text_encoder"],
    ),
    ("ltx2", "app.engine.models.families.ltx2.driver", "Ltx2Driver", ["text_encoder"]),
    (
        "lumina2",
        "app.engine.models.families.lumina2.driver",
        "Lumina2Driver",
        ["text_encoder"],
    ),
    (
        "microsoft_lens",
        "app.engine.models.families.microsoft_lens.driver",
        "MicrosoftLensDriver",
        ["text_encoder"],
    ),
    (
        "nucleus_image",
        "app.engine.models.families.nucleus_image.driver",
        "NucleusImageDriver",
        ["text_encoder"],
    ),
    (
        "omnigen2",
        "app.engine.models.families.omnigen2.driver",
        "OmniGen2Driver",
        ["text_encoder"],
    ),
    (
        "ovis_image",
        "app.engine.models.families.ovis_image.driver",
        "OvisImageDriver",
        ["text_encoder"],
    ),
    ("prx", "app.engine.models.families.prx.driver", "PRXDriver", ["text_encoder"]),
    (
        "prx_pixel",
        "app.engine.models.families.prx_pixel.driver",
        "PRXPixelDriver",
        ["text_encoder"],
    ),
    (
        "qwen_image",
        "app.engine.models.families.qwen_image.driver",
        "QwenImageDriver",
        ["text_encoder"],
    ),
    (
        "sdxl",
        "app.engine.models.families.sdxl.driver",
        "SDXLDriver",
        ["text_encoder_1", "text_encoder_2"],
    ),
    (
        "zimage",
        "app.engine.models.families.zimage.driver",
        "ZImageDriver",
        ["text_encoder"],
    ),
    # WAN 2.1 / WAN 2.2 / Bernini-R / WAN 2.2 TI2V-5B all subclass
    # WanDriverBase and inherit its get_text_encoders/release_text_encoders
    # verbatim (none override either) -- one override covers all four.
    (
        "wan_shared",
        "app.engine.models.families.wan_shared.driver_base",
        "WanDriverBase",
        ["text_encoder"],
    ),
]


def _import_driver_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.mark.parametrize(
    "family,module_path,class_name,attrs",
    _FAMILY_TE_ATTRS,
    ids=[f[0] for f in _FAMILY_TE_ATTRS],
)
def test_family_release_nulls_exactly_its_own_te_attrs(
    family,
    module_path,
    class_name,
    attrs,
):
    driver_cls = _import_driver_class(module_path, class_name)
    driver = object.__new__(driver_cls)

    sentinels = {attr: _RecordingTE() for attr in attrs}
    for attr, sentinel in sentinels.items():
        setattr(driver, attr, sentinel)

    if attrs:
        # Sanity: get_text_encoders() actually reports what we just set,
        # otherwise this case is testing nothing.
        before = driver.get_text_encoders()
        assert before, f"{family}: get_text_encoders() reported nothing for {attrs}"

    driver.release_text_encoders()

    for attr in attrs:
        assert getattr(driver, attr) is None, (
            f"{family}: release_text_encoders() did not null '{attr}'"
        )
    assert driver.get_text_encoders() == {}, (
        f"{family}: driver still reports a text encoder after release"
    )


def test_ace_step15_release_preserves_condition_stashes():
    """ace_step15's driver-owned CPU stashes (silence_latent /
    null_condition_emb) must survive release -- forward_pass needs them
    every step even after the TE is unloaded; they are NOT part of
    get_text_encoders() and must not be touched."""
    driver_cls = _import_driver_class(
        "app.engine.models.families.ace_step15.driver",
        "AceStep15Driver",
    )
    driver = object.__new__(driver_cls)
    driver.text_encoder = _RecordingTE()
    driver.condition_encoder = _RecordingTE()
    sentinel_silence = torch.zeros(4)
    sentinel_null = torch.ones(4)
    driver._silence_latent = sentinel_silence
    driver._null_condition_emb = sentinel_null

    driver.release_text_encoders()

    assert driver.text_encoder is None
    assert driver.condition_encoder is None
    assert driver._silence_latent is sentinel_silence
    assert driver._null_condition_emb is sentinel_null
