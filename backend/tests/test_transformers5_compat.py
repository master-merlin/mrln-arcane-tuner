"""Contract tests for the transformers 5.x upgrade."""

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
