"""Tests for the shared run-name derivation helper.

`model_part_from_definition_id` was previously copy-pasted across run_trainer,
job_routes, job_manager, pipeline_optimization and sampling. These tests pin
its behavior so the single source of truth can't silently drift.
"""
from app.core.naming import model_part_from_definition_id


def test_strips_namespace():
    assert model_part_from_definition_id("ostris/flux-dev") == "flux-dev"


def test_replaces_colon_with_underscore():
    assert model_part_from_definition_id("ostris/flux-dev:turbo") == "flux-dev_turbo"


def test_no_namespace_passthrough():
    assert model_part_from_definition_id("flux2") == "flux2"


def test_multi_segment_takes_last():
    assert model_part_from_definition_id("a/b/c:d") == "c_d"


def test_matches_legacy_inline_expression():
    # The exact expression the 5 call sites used before consolidation.
    for did in ("ostris/flux-dev:turbo", "flux2", "a/b/c:d", "foo:bar"):
        assert model_part_from_definition_id(did) == did.split("/")[-1].replace(":", "_")
