"""Contract tests for the transformers 5.x upgrade."""

import pytest
import transformers


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
    not __import__("pathlib").Path(
        r"D:\AI\huggingface\hub\hub\models--tencent--Youtu-VL-4B-Instruct"
    ).exists(),
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


def test_no_deprecated_transformers_kwargs_remain():
    """torch_dtype= and use_fast= are deprecated in 5.x. They still work today,
    so nothing else would catch their eventual removal."""
    import pathlib

    offenders = []
    for path in pathlib.Path("app/core/captioning").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for bad in ("torch_dtype=", "use_fast="):
            if bad in source:
                offenders.append(f"{path}: {bad}")
    assert offenders == []
