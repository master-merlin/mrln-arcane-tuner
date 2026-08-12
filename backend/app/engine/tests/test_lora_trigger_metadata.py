"""The trigger word must reach BOTH metadata keys, in every saver.

Kohya's ``ss_training_comment`` is where CivitAI and most LoRA browsers look;
``modelspec.trigger_phrase`` is the field the Stability ModelSpec defines for
it (mirrored by kohya's own ``library/sai_model_spec.py``). Writing only the
comment leaves a spec-following tool with nothing.

Assertions are on produced metadata — the bytes in the file where a model can
be built cheaply, the returned mapping otherwise — never on "a helper was
called".
"""

import pathlib

import pytest

from app.engine.utils.lora_metadata import (
    TRIGGER_COMMENT_KEY,
    TRIGGER_PHRASE_KEY,
    trigger_metadata,
)
from app.engine.utils.lora_tools import _parse_training_params

_SAVER_SOURCES = sorted(
    pathlib.Path("backend/app/engine").rglob("saver*.py")
) or sorted(pathlib.Path(__file__).resolve().parents[1].rglob("saver*.py"))


def test_writes_both_keys_with_the_same_value():
    meta = trigger_metadata({"global_triggerword": "McLarenSenna"})
    assert meta == {
        TRIGGER_COMMENT_KEY: "McLarenSenna",
        TRIGGER_PHRASE_KEY: "McLarenSenna",
    }
    # Pin the spellings themselves: these are an external contract, and a typo
    # in either is invisible until a third-party tool silently finds nothing.
    assert TRIGGER_COMMENT_KEY == "ss_training_comment"
    assert TRIGGER_PHRASE_KEY == "modelspec.trigger_phrase"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"global_triggerword": ""},
        {"global_triggerword": "   "},
        {"global_triggerword": None},
        {"lora_name": "x"},  # key absent entirely
        None,  # not a dict at all
        "nonsense",
    ],
)
def test_no_trigger_word_writes_no_key(config):
    """An empty value would advertise a trigger phrase that does not exist, and
    a reader cannot tell "" apart from a deliberate blank. Absence is the only
    unambiguous encoding."""
    assert trigger_metadata(config) == {}


def test_surrounding_whitespace_is_not_part_of_the_trigger():
    meta = trigger_metadata({"global_triggerword": "  neon rim  "})
    assert meta[TRIGGER_PHRASE_KEY] == "neon rim"
    assert meta[TRIGGER_COMMENT_KEY] == "neon rim"


def test_every_kohya_metadata_builder_uses_the_shared_helper():
    """Anti-drift guard: the map is copied per family, and ltx2 shipped without
    the trigger word at all because its copy predated that line. Any module
    that builds the ``ss_*`` header must take the trigger keys from one place.
    """
    builders = [
        p for p in _SAVER_SOURCES
        if "ss_output_name" in p.read_text(encoding="utf-8")
    ]
    assert len(builders) >= 8, f"expected the family savers, found {builders}"

    missing = [
        p.as_posix() for p in builders
        if "trigger_metadata" not in p.read_text(encoding="utf-8")
    ]
    assert not missing, f"these build ss_ metadata without the trigger word: {missing}"

    # …and none of them re-types the mapping the helper owns.
    retyped = [
        p.as_posix() for p in builders
        if '"global_triggerword": "ss_training_comment"' in p.read_text(encoding="utf-8")
    ]
    assert not retyped, f"duplicate trigger mapping is back in: {retyped}"


def test_ltx2_config_metadata_carries_the_trigger_word():
    """ltx2 built its own map and omitted the trigger word entirely — the drift
    this centralisation exists to end."""
    from app.engine.models.families.ltx2.saver import Ltx2Saver

    out = Ltx2Saver._config_metadata(
        {"lora_name": "my_lora", "global_triggerword": "SennaGTR"}
    )
    assert out["ss_output_name"] == "my_lora"  # not vacuous
    assert out[TRIGGER_COMMENT_KEY] == "SennaGTR"
    assert out[TRIGGER_PHRASE_KEY] == "SennaGTR"


def test_inspector_reads_the_spec_field_and_falls_back_to_the_comment():
    """Round-trip: what a saver writes is what the LoRA inspector reports."""
    written = trigger_metadata({"global_triggerword": "SennaGTR"})
    assert _parse_training_params(written)["trigger_phrase"] == "SennaGTR"

    # A LoRA from before this key existed carries the comment only.
    legacy = {TRIGGER_COMMENT_KEY: "OldTrigger"}
    assert _parse_training_params(legacy)["trigger_phrase"] == "OldTrigger"

    # The spec field wins when both are present and disagree — a third-party
    # file may use the comment for a genuine free-text note.
    both = {TRIGGER_COMMENT_KEY: "a note about the run", TRIGGER_PHRASE_KEY: "RealTrigger"}
    assert _parse_training_params(both)["trigger_phrase"] == "RealTrigger"

    # Nothing is invented when the file has neither.
    assert "trigger_phrase" not in _parse_training_params({"ss_network_dim": "32"})
