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
