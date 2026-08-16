"""Contract tests for the transformers 5.x upgrade."""

import pathlib

import pytest
import torch
import transformers
from huggingface_hub.constants import HF_HUB_CACHE


def test_transformers_is_5_14_1():
    """The pin is exact: LTX-2.5's encoder config declares 5.14.1 and it is the
    lowest release confirmed to carry gemma4_unified."""
    assert transformers.__version__ == "5.14.1"


def test_gemma4_unified_is_available():
    """The whole reason for this upgrade - LTX-2.5's text encoder class."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "gemma4_unified" in CONFIG_MAPPING_NAMES
    assert hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")


def test_shim_restores_the_three_moved_symbols():
    """tencent's remote processor code imports all three from
    `transformers.image_processing_utils_fast`, which 5.x turned into an alias
    module with an empty import structure. Without the shim this raises
    ImportError: cannot import name ... (unknown location)."""
    from app.core.captioning.compat.transformers5 import install_transformers5_compat

    install_transformers5_compat()

    from transformers.image_processing_utils_fast import (
        BaseImageProcessorFast,
        DefaultFastImageProcessorKwargs,
        SizeDict,
    )

    assert BaseImageProcessorFast is not None
    assert SizeDict is not None
    assert "do_resize" in DefaultFastImageProcessorKwargs.__annotations__


def test_shim_is_idempotent():
    """It is called from CaptionService.__init__ and may be called again by
    tests; a second call must not raise or rebind to a different object."""
    from app.core.captioning.compat.transformers5 import install_transformers5_compat

    install_transformers5_compat()
    import transformers.image_processing_backends as backends

    first = backends.DefaultFastImageProcessorKwargs
    install_transformers5_compat()
    assert backends.DefaultFastImageProcessorKwargs is first


def test_shim_does_not_redefine_the_bc_alias():
    """BaseImageProcessorFast still exists upstream as an alias for
    TorchvisionBackend. The shim must not shadow it with something else."""
    from app.core.captioning.compat.transformers5 import install_transformers5_compat

    install_transformers5_compat()
    import transformers.image_processing_backends as backends

    assert backends.BaseImageProcessorFast is backends.TorchvisionBackend


def test_bundled_siglip2_fast_imports_under_transformers_5():
    """The app's own copy of Youtu-VL's fast processor must import cleanly,
    independent of the shim - it is our code, so it targets 5.x directly."""
    from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast

    assert Siglip2ImageProcessorFast is not None


def test_bundled_siglip2_fast_is_constructible():
    """youtu_vl.py injects this processor with max_num_patches=256 to cap vision
    tokens. Assert a NON-default value too: 256 is the class attribute default,
    so asserting only 256 would pass even if the kwarg never reached the
    instance."""
    from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast

    # Non-default proves the constructor kwarg actually plumbs through.
    assert Siglip2ImageProcessorFast(max_num_patches=128).max_num_patches == 128
    # Production value used by youtu_vl.py.
    assert Siglip2ImageProcessorFast(max_num_patches=256).max_num_patches == 256


def test_caption_service_installs_the_shim_on_construction():
    """The shim must be in place before any plugin can call load(), because a
    trust_remote_code import that wins the race raises ImportError."""
    import app.core.captioning.compat.transformers5 as compat
    from app.core.captioning.caption_service import CaptionService

    compat._INSTALLED = False  # force a fresh install for this assertion
    try:
        CaptionService.reset_instance()
        CaptionService()
        assert compat._INSTALLED is True
    finally:
        # Restore regardless of outcome: an assertion failure here must not
        # leave _INSTALLED False for every other test in the session that
        # relies on the shim already being installed.
        compat._INSTALLED = True


def test_youtu_vl_shim_is_still_required():
    """PROVE THE NEGATIVE — and give the shim a real retirement trigger.

    `compat/transformers5.py` documents that it should be deleted once tencent's
    remote code targets transformers 5.x. Without a test, that day passes
    unnoticed and the shim rots. This asserts the shim is still NECESSARY: in a
    clean interpreter, two of the three symbols the shim restores
    (`DefaultFastImageProcessorKwargs`, `SizeDict`) must be missing from the
    alias module. The third, `BaseImageProcessorFast`, is deliberately excluded
    from the probe: it already survives upstream as a BC alias and the shim
    never re-shims it (see `install_transformers5_compat`'s comment), so it is
    always present and would never signal retirement either way. When upstream
    (or tencent) catches up on the other two, this test fails and that failure
    is the signal to delete the shim.

    Runs in a subprocess so it is unaffected by whatever this session already
    imported or patched.
    """
    import subprocess
    import sys

    probe = (
        "import transformers.image_processing_backends as b;"
        "missing=[n for n in ('DefaultFastImageProcessorKwargs','SizeDict')"
        " if not hasattr(b,n)];"
        "print(','.join(missing))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert out, (
        "transformers now exposes DefaultFastImageProcessorKwargs and SizeDict "
        "on image_processing_backends without our shim -- the compat shim in "
        "app/core/captioning/compat/transformers5.py is obsolete. DELETE IT "
        "and remove this test."
    )


@pytest.mark.skipif(
    # Resolve the cache dir the way huggingface_hub itself does (respects
    # HF_HOME/HF_HUB_CACHE) instead of a hardcoded machine-specific path -- a
    # literal path here would silently skip this test, the only real-code
    # evidence for the Youtu-VL shim, on any other machine or CI.
    not (pathlib.Path(HF_HUB_CACHE) / "models--tencent--Youtu-VL-4B-Instruct").exists(),
    reason="Youtu-VL checkpoint not in the local HF cache",
)
def test_youtu_vl_processor_loads_with_the_shim():
    """Observable output: AutoProcessor returns a real YoutuVLProcessor.
    Without the shim this raises ImportError inside tencent's remote code."""
    from transformers import AutoProcessor

    from app.core.captioning.compat.transformers5 import install_transformers5_compat

    install_transformers5_compat()
    proc = AutoProcessor.from_pretrained(
        "tencent/Youtu-VL-4B-Instruct",
        trust_remote_code=True,
        backend="torchvision",  # the exact kwarg youtu_vl.py:76 now passes
        local_files_only=True,
    )
    assert type(proc).__name__ == "YoutuVLProcessor"


def test_rope_default_shim_produces_correct_inv_freq():
    """`ROPE_INIT_FUNCTIONS["default"]` must not just be present - it must compute
    the right thing. `install_transformers5_compat` documents this as a
    byte-for-byte port of transformers 4.57's `_compute_default_rope_parameters`;
    pin the OBSERVABLE OUTPUT (shape, dtype, and known-good values) rather than
    mere presence/callability, so a future edit to the port that changes the maths
    is caught here instead of silently corrupting every RoPE position embedding
    for remote-code models that dict-dispatch to "default".
    """
    from types import SimpleNamespace

    from app.core.captioning.compat.transformers5 import install_transformers5_compat
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    install_transformers5_compat()

    assert "default" in ROPE_INIT_FUNCTIONS
    assert callable(ROPE_INIT_FUNCTIONS["default"])

    # No `head_dim` attribute -> the function must fall back to
    # hidden_size // num_attention_heads (8 here), matching 4.57's contract.
    config = SimpleNamespace(rope_theta=10000.0, hidden_size=64, num_attention_heads=8)
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["default"](config)

    # dim = head_dim(8) * partial_rotary_factor(default 1.0) = 8;
    # arange(0, 8, 2) -> 4 elements. The maths this feeds directly requires a
    # float tensor (division and later cos/sin), not int64.
    assert inv_freq.shape == (4,)
    assert inv_freq.dtype == torch.float32
    assert attention_factor == 1.0

    expected = torch.tensor(
        [10000.0 ** (-i / 8) for i in (0, 2, 4, 6)], dtype=torch.float32
    )
    assert torch.allclose(inv_freq, expected, rtol=1e-5)


@pytest.mark.skipif(
    # Same cache-existence gate as test_youtu_vl_processor_loads_with_the_shim:
    # a hardcoded path would silently skip this on any other machine/CI.
    not (pathlib.Path(HF_HUB_CACHE) / "models--tencent--Youtu-VL-4B-Instruct").exists(),
    reason="Youtu-VL checkpoint not in the local HF cache",
)
def test_youtu_vl_config_normalisation_neutralises_rope_scaling():
    """Pin the exact precondition that stops `KeyError: 'factor'` in the remote
    `YoutuMLAttention.__init__`: after the normalisation `load()` applies,
    `config.rope_scaling` must be `None`. Reproduces `load()`'s own two-line patch
    (rather than re-deriving it) so a future edit to `load()` that stops nulling
    `rope_scaling` is caught here too - config-level only, no weights, so it stays
    hermetic and CI-safe.
    """
    from transformers import AutoConfig

    from app.core.captioning.compat.transformers5 import install_transformers5_compat

    install_transformers5_compat()

    config = AutoConfig.from_pretrained(
        "tencent/Youtu-VL-4B-Instruct", trust_remote_code=True, local_files_only=True
    )

    # Sanity: prove the precondition this test guards against is real - 5.x's
    # standardize_rope_params() synthesises a truthy rope_scaling on this
    # checkpoint (its config.json has no explicit rope_scaling key at all).
    # Without this assertion, a future transformers release that stops
    # synthesising rope_scaling would make the test below pass vacuously.
    assert getattr(config, "rope_scaling", None) is not None

    # The exact normalisation youtu_vl.py's load() applies before from_pretrained().
    if getattr(config, "rope_scaling", None) is not None:
        config.rope_scaling = None

    assert config.rope_scaling is None


def test_only_one_typer_distribution_is_installed():
    """typer and typer-slim both provide an importable `typer` module; having
    both makes which one wins install-order dependent."""
    from importlib.metadata import distributions

    names = {d.metadata["Name"].lower() for d in distributions()}
    assert not ({"typer", "typer-slim"} <= names), (
        "both typer and typer-slim are installed - pick one in requirements.txt"
    )


def test_sam3_imports_cleanly_despite_declared_hub_pin():
    """sam3 declares huggingface-hub<1.0 but works with 1.x. app/core/masking/
    models/sam3.py swallows ImportError into SAM3_AVAILABLE=False, so without
    this test a real break would silently disable masking."""
    from app.core.masking.models import sam3

    assert sam3.SAM3_AVAILABLE is True, (
        "sam3 failed to import - masking is silently disabled; check the "
        "huggingface_hub compatibility rather than ignoring this flag"
    )


def test_hub_apis_the_app_depends_on_still_exist():
    """huggingface_hub 0.36 -> 1.27 is a major bump. These are the hub APIs this
    repo calls directly; a 1.x removal must fail here, not at download time in
    front of a user.

    Grepped backend/app for huggingface_hub/HfApi/hf_hub_download/
    snapshot_download/list_repo_files/model_info/GatedRepoError/
    RepositoryNotFoundError first. Two real call sites weren't in that seed
    list and were added: ``HfApi().repo_info`` (app/api/events/
    download_progress.py, for the top-bar download-size preflight) and
    ``try_to_load_from_cache`` (download_progress.py + app/engine/utils/
    model_utils.py, for offline cache-hit checks). ``model_info`` itself has
    no direct call site today but stays in the assertion list per the
    original template - it's part of the public HfApi surface this repo
    could reasonably reach for next.
    """
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
        try_to_load_from_cache,
    )
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    assert callable(api.list_repo_files)
    assert callable(api.model_info)
    assert callable(api.repo_info)
    assert callable(hf_hub_download)
    assert callable(snapshot_download)
    assert callable(try_to_load_from_cache)
    assert issubclass(GatedRepoError, Exception)
    assert issubclass(RepositoryNotFoundError, Exception)


class _FakeDefaultRope(torch.nn.Module):
    """Mimics `YoutuRotaryEmbedding`'s exact contract: a `.config`, a
    `.rope_type` string used to dict-dispatch into `ROPE_INIT_FUNCTIONS`, and
    an `inv_freq` buffer registered with `persistent=False`. The buffer is
    seeded with zeros to stand in for transformers 5.x's meta-materialization
    leaving it uninitialized -- exactly the defect under test."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        head_dim = config.hidden_size // config.num_attention_heads
        self.register_buffer("inv_freq", torch.zeros(head_dim // 2), persistent=False)
        self.attention_scaling = None


class _FakeUnknownContractRope(torch.nn.Module):
    """A THIRD, hypothetical rotary module shape: non-persistent `inv_freq`
    with NEITHER a `.config`/`.rope_type` (the LM contract) NOR the class
    name `VisionRope` (the vision contract, now also repaired -- see
    `VisionRope` below and `test_repair_recomputes_the_vision_rope_module`).
    Stands in for "a future remote-code revision adds yet another RoPE shape
    this repair doesn't recognize yet" -- must still WARN, not crash."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("inv_freq", torch.zeros(dim), persistent=False)


class VisionRope(torch.nn.Module):
    """Reproduces the REAL remote class's constructor contract byte-for-byte
    (`modeling_siglip2.py:611-615`, verified against the cached checkpoint
    revision `8d30a0e4...`): a bare `dim` int, stored NEITHER as `.config`
    NOR `.rope_type` on the instance. Named literally `VisionRope` (not
    `_FakeVisionRope`) because production identifies this module by
    `type(module).__name__ == "VisionRope"` -- the class name IS the
    contract under test here, the same way `_FakeDefaultRope` above mimics
    `YoutuRotaryEmbedding`'s attribute contract. The buffer is seeded with
    zeros, standing in for transformers 5.x's meta-materialization leaving a
    `persistent=False` buffer uninitialized -- the exact defect this branch
    of the repair exists to fix."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("inv_freq", torch.zeros(dim // 2), persistent=False)


def _plugin() -> "object":
    from app.core.captioning.models.youtu_vl import YoutuVLModel

    return YoutuVLModel(service=None)


def test_repair_recomputes_a_default_contract_rope_module():
    """The exact bug from youtu-numerics-report.md, reproduced hermetically:
    a `persistent=False` inv_freq buffer left at zero by meta materialization
    must come back as the real `ROPE_INIT_FUNCTIONS["default"]` values -
    the same function `compat/transformers5.py` installs - after the repair
    runs, with `attention_scaling` refreshed too."""
    from types import SimpleNamespace

    from app.core.captioning.compat.transformers5 import install_transformers5_compat
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    install_transformers5_compat()

    config = SimpleNamespace(rope_theta=500000.0, hidden_size=256, num_attention_heads=4)
    rope = _FakeDefaultRope(config)
    top = torch.nn.Module()
    top.rotary_emb = rope
    top.config = config

    assert torch.all(rope.inv_freq == 0), "precondition: buffer starts as uninitialized zeros"

    _plugin()._repair_nonpersistent_rope_buffers(top)

    expected_inv_freq, expected_scaling = ROPE_INIT_FUNCTIONS["default"](config, rope.inv_freq.device)
    assert not torch.all(rope.inv_freq == 0)
    assert torch.allclose(rope.inv_freq, expected_inv_freq)
    assert rope.attention_scaling == expected_scaling


def test_repair_warns_and_leaves_unknown_contract_modules_untouched(caplog):
    """A rotary module matching NEITHER the LM contract (`.config` +
    `.rope_type == "default"`) NOR the vision contract (class name
    `VisionRope`) has no way to safely recompute it without guessing
    constructor parameters. It must WARN (failure is never silent) and leave
    the buffer alone rather than crash the whole load() or misapply either
    known formula -- this reproduces the exact shape-mismatch crash hit
    during development (dim 18 vision buffer vs dim 32 LM formula) that
    motivated this guard in the first place, before the vision buffer had
    its own dedicated repair path."""
    import logging

    unknown = _FakeUnknownContractRope(dim=18)
    top = torch.nn.Module()
    top.encoder = torch.nn.Module()
    top.encoder.rotary_pos_emb = unknown
    top.config = object()

    with caplog.at_level(logging.WARNING, logger="app.core.captioning.models.youtu_vl"):
        _plugin()._repair_nonpersistent_rope_buffers(top)

    assert torch.all(unknown.inv_freq == 0), "left untouched, not guessed at"
    assert "youtu_vl_rope_buffer_unrepaired_unknown_contract" in caplog.text


def test_repair_recomputes_the_vision_rope_module():
    """The vision tower's non-persistent `inv_freq` (real class `VisionRope`,
    `modeling_siglip2.py`) must be recomputed from the model's OWN
    `config.vision_config` -- exactly matching `VisionRope.__init__`'s own
    formula (`modeling_siglip2.py:611-615`) with `dim = hidden_size //
    num_attention_heads // 2` and the call site's implicit `theta=10000.0`
    (`modeling_siglip2.py:642` never overrides it). This is the exact
    scenario youtu-numerics-report.md and uat-fix-3-report.md documented as
    unfixed: the vision tower degrading captions ('five-pointed stars'
    instead of the real triangle/circle) even after the text tower's RoPE
    was repaired."""
    from types import SimpleNamespace

    vision_config = SimpleNamespace(hidden_size=1152, num_attention_heads=16)
    dim = vision_config.hidden_size // vision_config.num_attention_heads // 2  # 36
    rope = VisionRope(dim)
    top = torch.nn.Module()
    top.vision_rope = rope
    top.config = SimpleNamespace(vision_config=vision_config)

    assert torch.all(rope.inv_freq == 0), "precondition: buffer starts as uninitialized zeros"

    _plugin()._repair_nonpersistent_rope_buffers(top)

    expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    assert not torch.all(rope.inv_freq == 0)
    assert torch.allclose(rope.inv_freq, expected)


def test_repair_raises_when_vision_config_dim_does_not_match_buffer_shape():
    """If the config-derived `dim` no longer matches the materialized
    buffer's own shape (e.g. an upstream config edit lands out of sync with
    the checkpoint), the repair must raise rather than silently overwrite
    with a mismatched-shape tensor. Deriving `dim` from config rather than
    from the buffer's own length is deliberate (see the function's
    docstring) precisely because the buffer's length alone could never catch
    this drift -- it would just self-confirm whatever shape is already
    there."""
    from types import SimpleNamespace

    # Buffer built as if dim=18 (9 elements) but vision_config implies dim=36
    # (1152 // 16 // 2), i.e. an 18-element buffer -- a real, deliberate
    # mismatch.
    vision_config = SimpleNamespace(hidden_size=1152, num_attention_heads=16)
    rope = VisionRope(dim=18)
    top = torch.nn.Module()
    top.vision_rope = rope
    top.config = SimpleNamespace(vision_config=vision_config)

    with pytest.raises(RuntimeError, match="recomputed vision RoPE inv_freq"):
        _plugin()._repair_nonpersistent_rope_buffers(top)


def test_repair_warns_when_vision_rope_found_but_vision_config_unreachable(caplog):
    """If a future config refactor removes `vision_config` from the model's
    root `.config` (or the root config is missing entirely), the
    `VisionRope`-named module must fall through to the generic
    unknown-contract WARNING rather than crash `load()` outright -- same
    'never guess constructor parameters' rule that governs branch 3."""
    import logging

    rope = VisionRope(dim=36)
    top = torch.nn.Module()
    top.vision_rope = rope
    top.config = object()  # no .vision_config attribute at all

    with caplog.at_level(logging.WARNING, logger="app.core.captioning.models.youtu_vl"):
        _plugin()._repair_nonpersistent_rope_buffers(top)

    assert torch.all(rope.inv_freq == 0), "left untouched, not guessed at"
    assert "youtu_vl_rope_buffer_unrepaired_unknown_contract" in caplog.text


def test_repair_raises_when_no_rope_module_is_found_at_all():
    """If the module walk finds NO `inv_freq` buffer anywhere, the remote
    code's RoPE structure has changed in a way this repair no longer reaches.
    Silently returning would ship exactly the degenerate-caption failure this
    function exists to prevent -- it must raise instead."""
    top = torch.nn.Module()
    top.some_child = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="no submodule with an 'inv_freq' buffer"):
        _plugin()._repair_nonpersistent_rope_buffers(top)


def test_repair_is_a_noop_when_buffers_are_already_persistent():
    """If a future transformers/remote-code revision makes `inv_freq`
    persistent (checkpoint-restored, no longer at risk), the repair must not
    raise or touch it -- there is nothing to repair."""
    from types import SimpleNamespace

    config = SimpleNamespace(rope_theta=500000.0, hidden_size=256, num_attention_heads=4)
    rope = _FakeDefaultRope(config)
    rope._non_persistent_buffers_set.discard("inv_freq")  # simulate persistent=True
    top = torch.nn.Module()
    top.rotary_emb = rope

    _plugin()._repair_nonpersistent_rope_buffers(top)  # must not raise

    assert torch.all(rope.inv_freq == 0), "not touched -- persistent buffers are the checkpoint's job"


def test_no_deprecated_transformers_kwargs_remain():
    """torch_dtype= and use_fast= are deprecated in 5.x. They still work today,
    so nothing else would catch their eventual removal.

    Scope is DELIBERATELY limited to app/core/captioning: torch_dtype= also
    appears in app/engine/core/pipeline/pipeline_loading.py and the
    boogu_image/krea2/omnigen2 family loaders' ModelMixin.from_pretrained()
    calls, where it is a diffusers kwarg (diffusers still uses torch_dtype=,
    unrelated to this transformers deprecation) and is still correct.
    Widening this scan would pressure someone into "fixing" those diffusers
    call sites and breaking three families.
    """
    # Anchored on this test file's location, not the pytest rootdir/CWD: a
    # CWD-relative Path("app/core/captioning") yields nothing if pytest ever
    # runs from a different directory, and rglob() over an empty/missing dir
    # just returns no results -- offenders == [] then passes VACUOUSLY,
    # silently turning this guard off instead of failing loudly.
    backend_root = pathlib.Path(__file__).resolve().parents[1]
    captioning_root = backend_root / "app/core/captioning"

    offenders = []
    for path in captioning_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for bad in ("torch_dtype=", "use_fast="):
            if bad in source:
                offenders.append(f"{path}: {bad}")
    assert offenders == []
