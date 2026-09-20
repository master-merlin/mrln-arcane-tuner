"""LANE-92 row 5.1: the shipped minimax_h3 definitions are FIXED POINTS of the
definition enricher.

``TrainingPipeline`` calls ``ModelRegistry.enrich_definition`` on every load;
it harvests the snapshot's component ``config.json`` files into
``architecture_params`` and SAVES the definition back to its YAML. Enrichment
into the YAML is the designed behaviour (every other shipped family carries its
harvested ``te.*`` keys) — so a definition that ships un-enriched is rewritten
by the user's first run, which dirties a tracked file in their checkout.

No network, no weights: the harvester reads four small JSON files, copied
byte-for-byte into ``fixtures/minimax_h3_hf_configs`` from the
``MiniMaxAI/MiniMax-H3`` snapshot ``42ed227ee7df40d41602854ae760620d6eb651fe``
(``transformer/``, ``text_encoder/``, ``vae/`` ``config.json`` and
``scheduler/scheduler_config.json`` — the four the harvester's ``COMPONENTS``
table finds in that repo). The real ``enrich_definition`` -> ``save_definition``
path runs; only the file it saves to is a copy under ``tmp_path``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.engine.models.registry import registry

BACKEND = Path(__file__).resolve().parents[1]
DEFINITIONS = BACKEND / "app/engine/models/families/minimax_h3/definitions"
HF_CONFIGS = Path(__file__).resolve().parent / "fixtures/minimax_h3_hf_configs"

H3_FILES = {
    "minimax-h3-t2va": "minimax_h3_t2va.yaml",
    "minimax-h3-fl2va": "minimax_h3_fl2va.yaml",
    "minimax-h3-ref2va": "minimax_h3_ref2va.yaml",
}


@pytest.fixture
def enrich_a_copy(tmp_path):
    """Run the real enricher against a COPY of a shipped definition and hand
    back the copy's path. Registry state is restored afterwards."""
    registry.initialize()
    saved: dict[str, tuple] = {}

    def run(did: str, text: str | None = None) -> Path:
        shipped = DEFINITIONS / H3_FILES[did]
        copy = tmp_path / shipped.name
        if text is None:
            shutil.copyfile(shipped, copy)
        else:
            copy.write_text(text, encoding="utf-8", newline="")
        saved.setdefault(did, (registry._definitions[did], registry._paths[did]))
        registry.load_definition(str(copy))  # in-memory state == the copy
        registry.enrich_definition(did, {}, root_path=str(HF_CONFIGS))
        return copy

    try:
        yield run
    finally:
        for did, (defn, path) in saved.items():
            registry._definitions[did] = defn
            registry._paths[did] = path


def test_the_fixture_is_what_the_harvester_reads() -> None:
    from app.engine.utils.config_harvester import harvest

    harvested = harvest(str(HF_CONFIGS))
    assert {k.split(".")[0] for k in harvested} == {"transformer", "te", "scheduler", "vae"}
    assert harvested["te.model_type"] == "qwen3_vl"


@pytest.mark.parametrize("did", sorted(H3_FILES))
def test_shipped_definition_is_an_enricher_fixed_point(did: str, enrich_a_copy) -> None:
    # Text, newline-normalised: `save_definition` writes the platform's line
    # ending and `.gitattributes` (`* text=auto eol=lf`) normalises it away, so
    # EOL alone is not dirt in a checkout; every other byte is.
    shipped = (DEFINITIONS / H3_FILES[did]).read_text(encoding="utf-8")
    copy = enrich_a_copy(did)
    assert copy.read_text(encoding="utf-8") == shipped, (
        f"{did}: the enricher rewrites the shipped YAML — a user's first run "
        "would dirty a tracked file. Ship the enricher's output."
    )


def test_the_enricher_does_rewrite_an_unenriched_definition(enrich_a_copy) -> None:
    """Prove the negative: the fixed-point test can fail. One harvested key
    removed from the copy -> the enricher puts it back, comments intact."""
    did = "minimax-h3-t2va"
    shipped = (DEFINITIONS / H3_FILES[did]).read_text(encoding="utf-8")
    line = "  te.model_type: qwen3_vl\n"
    assert shipped.count(line) == 1
    stripped = shipped.replace(line, "")
    copy = enrich_a_copy(did, stripped)
    after = copy.read_text(encoding="utf-8")
    assert after != stripped, "the enricher did not run / did not save"
    assert "te.model_type: qwen3_vl" in after
    comments = [ln for ln in shipped.splitlines() if ln.lstrip().startswith("#")]
    assert comments and all(c in after for c in comments), "a comment was lost"
