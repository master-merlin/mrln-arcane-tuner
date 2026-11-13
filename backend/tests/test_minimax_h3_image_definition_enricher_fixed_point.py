"""The shipped minimax_h3_image definition is a FIXED POINT of the definition enricher.

``TrainingPipeline`` calls ``ModelRegistry.enrich_definition`` on every load; it harvests the
snapshot's component config files into ``architecture_params`` and SAVES the definition back to
its YAML. A definition that ships un-enriched is rewritten by the user's first run, which dirties
a tracked file in their checkout.

No network, no weights: the harvester reads four small JSON files, kept byte for byte in
``fixtures/minimax_h3_image_hf_configs`` from ``MiniMaxAI/MiniMax-H3-Image`` at ``00dd081e``:
``transformer/config.json``, ``vae/config.json``, ``text_encoder/config.json`` and
``scheduler/scheduler_config.json``, the four folders ``config_harvester.COMPONENTS`` finds in
that repo (``transformer/`` and ``vae/`` hold the model's native configs; the family loads the
``*_diffusers`` folders, which the harvester does not read). The real ``enrich_definition`` ->
``save_definition`` path runs; only the file it saves to is a copy under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from app.engine.models.registry import registry

BACKEND = Path(__file__).resolve().parents[1]
DEFINITION = BACKEND / "app/engine/models/families/minimax_h3_image/definitions/minimax_h3_image_t2i.yaml"
DEFINITION_ID = "minimax-h3-image-t2i"
HF_CONFIGS = Path(__file__).resolve().parent / "fixtures/minimax_h3_image_hf_configs"

# sha256 of each file as served at https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/raw/00dd081e/<path>.
FIXTURE_SHA256 = {
    "transformer/config.json": "67ceb1d7752942ee984405c26c1187ff58c073b698839ea0ca289e34b8dc61a9",
    "vae/config.json": "7fcc21bec549804ff0a04d8246e7f90e679d4f8d4b0a27a252bbd18c30e3e4bb",
    "text_encoder/config.json": "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427",
    "scheduler/scheduler_config.json": "ce1792ea34cef021fec782abf9f45125f6bf0438f7c424e567ce28b246ad0a08",
}

# modular_model_index.json `sample_sigmas` at 00dd081e, without the terminal 0.0.
HEAD_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132]


@pytest.fixture
def enrich_a_copy(tmp_path):
    """Run the real enricher against a COPY of the shipped definition and hand back the copy's
    path. Registry state is restored afterwards."""
    registry.initialize()
    saved: dict[str, tuple] = {}

    def run(text: str | None = None) -> Path:
        copy = tmp_path / DEFINITION.name
        if text is None:
            shutil.copyfile(DEFINITION, copy)
        else:
            copy.write_text(text, encoding="utf-8", newline="")
        saved.setdefault(DEFINITION_ID, (registry._definitions[DEFINITION_ID], registry._paths[DEFINITION_ID]))
        registry.load_definition(str(copy))  # in-memory state == the copy
        registry.enrich_definition(DEFINITION_ID, {}, root_path=str(HF_CONFIGS))
        return copy

    try:
        yield run
    finally:
        for did, (defn, path) in saved.items():
            registry._definitions[did] = defn
            registry._paths[did] = path


@pytest.mark.parametrize("rel", sorted(FIXTURE_SHA256))
def test_the_fixture_is_the_snapshot_byte_for_byte(rel: str) -> None:
    assert hashlib.sha256((HF_CONFIGS / rel).read_bytes()).hexdigest() == FIXTURE_SHA256[rel]


def test_the_fixture_is_what_the_harvester_reads() -> None:
    from app.engine.utils.config_harvester import harvest

    harvested = harvest(str(HF_CONFIGS))
    assert {k.split(".")[0] for k in harvested} == {"transformer", "te", "scheduler", "vae"}
    assert harvested["te.model_type"] == "qwen3_vl"
    assert harvested["transformer.num_layers"] == 50
    assert harvested["transformer.token_refiner_num_layers"] == 2
    assert harvested["scheduler.euler_update_form"] == "velocity"


def test_shipped_definition_is_an_enricher_fixed_point(enrich_a_copy) -> None:
    # Text, newline-normalised: `save_definition` writes the platform's line ending and
    # `.gitattributes` (`* text=auto eol=lf`) normalises it away, so EOL alone is not dirt in a
    # checkout; every other byte is.
    shipped = DEFINITION.read_text(encoding="utf-8")
    copy = enrich_a_copy()
    assert copy.read_text(encoding="utf-8") == shipped, (
        "the enricher rewrites the shipped YAML: a user's first run would dirty a tracked file. "
        "Ship the enricher's output."
    )


def test_the_enricher_does_rewrite_an_unenriched_definition(enrich_a_copy) -> None:
    """Prove the negative: the fixed-point test can fail. One harvested key removed from the
    copy -> the enricher puts it back, comments intact."""
    shipped = DEFINITION.read_text(encoding="utf-8")
    line = "  te.model_type: qwen3_vl\n"
    assert shipped.count(line) == 1
    stripped = shipped.replace(line, "")
    copy = enrich_a_copy(stripped)
    after = copy.read_text(encoding="utf-8")
    assert after != stripped, "the enricher did not run / did not save"
    assert "te.model_type: qwen3_vl" in after
    comments = [ln for ln in shipped.splitlines() if ln.lstrip().startswith("#")]
    assert comments and all(c in after for c in comments), "a comment was lost"


def test_the_timestep_default_is_pinned_and_survives_enrichment(enrich_a_copy) -> None:
    """``timestep.default_mode`` and ``timestep.fixed_sigmas`` are the family's default draw and
    its head grid; they are pinned, so no harvested config can ever replace them."""
    doc = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    arch = doc["architecture_params"]
    assert arch["timestep.default_mode"] == "fixed_sigmas"
    assert arch["timestep.fixed_sigmas"] == HEAD_SIGMAS
    assert {"timestep.default_mode", "timestep.fixed_sigmas"} <= set(doc["enrich_pinned_keys"])

    enriched = yaml.safe_load(enrich_a_copy().read_text(encoding="utf-8"))["architecture_params"]
    assert enriched["timestep.default_mode"] == "fixed_sigmas"
    assert enriched["timestep.fixed_sigmas"] == HEAD_SIGMAS


def test_a_defaults_block_never_restates_the_timestep_default() -> None:
    """The form default is derived from ``timestep.*`` (``archetypes._timestep_defaults``); the
    YAML states it once, so the two can never drift apart."""
    doc = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    assert "timestep_sampling" not in doc["defaults"]
    assert "fixed_sigmas" not in doc["defaults"]
