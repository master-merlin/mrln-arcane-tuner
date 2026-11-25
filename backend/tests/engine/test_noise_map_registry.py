"""The noise-coordinate report over every registered definition, and its text form.

CPU only, no weights: each definition's own trainer class draws its own timesteps.
"""
import pytest

from app.engine.core.noise_map import (
    Refusal,
    build_report,
    format_report,
    parse_sampler,
    sampling_shell,
    select_definitions,
)
from app.engine.models.registry import registry

pytestmark = pytest.mark.xdist_group("noise_map_registry")

OPTIONS = {"steps": 64, "phases": 2, "draws_per_step": 16}
RADC = "radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0.15"


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


def _default_failures(names=None) -> dict[str, str]:
    """``family/id`` -> reason of every definition whose ``default`` row is not mapped."""
    report = build_report(names, [parse_sampler("default")], [1024], **OPTIONS)
    return {f"{row['family']}/{row['definition']}": row["reason"] for row in report["rows"]
            if row["status"] != "mapped"}


def test_every_registered_definition_has_its_default_row_mapped():
    ids = select_definitions(None)
    assert len(ids) >= 32 and len(ids) == len(registry.list_models())
    report = build_report(None, [parse_sampler("default")], [1024], **OPTIONS)
    assert sorted(row["definition"] for row in report["rows"]) == sorted(registry.list_models())
    failures = _default_failures()
    assert not failures, "\n".join(f"{name}: {reason}" for name, reason in failures.items())


def test_every_other_row_is_mapped_or_says_why():
    report = build_report(None, [parse_sampler("uniform"), parse_sampler(RADC)], [1024], **OPTIONS)
    assert len(report["rows"]) == 2 * len(registry.list_models())
    for row in report["rows"]:
        where = f"{row['family']}/{row['definition']} {row['sampler']}"
        assert row["status"] in ("mapped", "not_mapped"), where
        if row["status"] == "not_mapped":
            assert row["reason"], where


def test_a_definition_that_can_not_be_drawn_fails_the_coverage_check(monkeypatch):
    broken = type(sampling_shell(registry.get_definition("moss-image-base"), {}, 1))
    expected = {f"{registry.get_definition(i).family}/{i}" for i in registry.list_models()
                if isinstance(sampling_shell(registry.get_definition(i), {}, 1), broken)}
    assert "moss_image/moss-image-base" in expected

    def boom(self, batch_size, latents=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(broken, "sample_timesteps", boom)
    failures = _default_failures()
    assert set(failures) == expected
    assert all(reason == "RuntimeError: boom" for reason in failures.values())


def test_a_family_name_selects_all_its_definitions():
    assert select_definitions(["moss_image"]) == ["moss-image-base", "moss-image-sft"]


def test_an_unknown_name_is_refused():
    with pytest.raises(Refusal, match="nope"):
        select_definitions(["moss-image-base", "nope"])


def test_the_text_puts_two_families_side_by_side():
    report = build_report(["flux3-image-klein-base", "moss-image-base"], [parse_sampler("uniform")], [1024],
                          bands=4, **OPTIONS)
    text = format_report(report)
    assert "flux3-image-klein-base" in text and "moss-image-base" in text
    assert "0.00-0.25" in text
    assert "Not mapped:" not in text


def test_the_text_lists_what_was_not_mapped():
    report = build_report(["sdxl_base_1.0"], [parse_sampler("uniform")], [1024], **OPTIONS)
    text = format_report(report)
    assert "Not mapped:" in text
    assert report["rows"][0]["reason"] in text
