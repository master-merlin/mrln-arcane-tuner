"""Contract tests for the transformers 5.x upgrade."""

import pathlib

import pytest
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
