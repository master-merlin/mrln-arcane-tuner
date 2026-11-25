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

pytestmark = [pytest.mark.xdist_group("noise_map_registry"), pytest.mark.usefixtures("initialized_registry")]

OPTIONS = {"steps": 64, "phases": 2, "draws_per_step": 16}


def _default_report(names=None) -> dict:
    return build_report(names, [parse_sampler("default")], [1024], **OPTIONS)


def _default_failures(report: dict) -> dict[str, str]:
    """``family/id`` -> reason of every definition whose ``default`` row is not mapped."""
    return {f"{row['family']}/{row['definition']}": row["reason"] for row in report["rows"]
            if row["status"] != "mapped"}


def test_every_registered_definition_has_its_default_row_mapped():
    ids = select_definitions(None)
    assert len(ids) >= 32 and len(ids) == len(registry.list_models())
    report = _default_report()
    assert sorted(row["definition"] for row in report["rows"]) == sorted(registry.list_models())
    failures = _default_failures(report)
    assert not failures, "\n".join(f"{name}: {reason}" for name, reason in failures.items())


def test_every_other_row_is_mapped_or_says_why(radc_curriculum):
    report = build_report(None, [parse_sampler("uniform"), parse_sampler(radc_curriculum)], [1024], **OPTIONS)
    assert len(report["rows"]) == 2 * len(registry.list_models())
    for row in report["rows"]:
        where = f"{row['family']}/{row['definition']} {row['sampler']}"
        assert row["status"] in ("mapped", "not_mapped"), where
        if row["status"] == "not_mapped":
            assert row["reason"], where


def test_a_definition_that_can_not_be_drawn_fails_the_coverage_check(draw_raises):
    names = ["moss_image", "flux3-image-klein-base", "sdxl_base_1.0"]
    broken = draw_raises("moss-image-base")
    selected = select_definitions(names)
    expected = {f"{registry.get_definition(i).family}/{i}" for i in selected
                if isinstance(sampling_shell(registry.get_definition(i), {}, 1), broken)}
    assert "moss_image/moss-image-base" in expected and len(expected) < len(selected)
    failures = _default_failures(_default_report(names))
    assert set(failures) == expected
    assert all(reason == "RuntimeError: boom" for reason in failures.values())


def test_a_family_name_selects_all_its_definitions():
    assert select_definitions(["moss_image"]) == ["moss-image-base", "moss-image-sft"]


def test_an_unknown_name_is_refused():
    with pytest.raises(Refusal, match="nope"):
        select_definitions(["moss-image-base", "nope"])


def test_a_row_without_a_grid_is_drawn_once_for_every_resolution(monkeypatch):
    import app.engine.core.noise_map as nm

    calls = []
    real = nm._map
    monkeypatch.setattr(nm, "_map", lambda *args: calls.append(args[2]) or real(*args))
    report = build_report(["flux1-schnell", "moss-image-base"], [parse_sampler("uniform")], [512, 1024], **OPTIONS)
    assert calls == [512, 512, 1024]
    flux1 = [row for row in report["rows"] if row["definition"] == "flux1-schnell"]
    assert [row["resolution"] for row in flux1] == [512, 1024]
    assert flux1[0]["all"] == flux1[1]["all"] and flux1[0]["status"] == "mapped"


def test_the_text_puts_two_families_side_by_side():
    report = build_report(["flux3-image-klein-base", "moss-image-base"], [parse_sampler("uniform")], [1024],
                          bands=4, **OPTIONS)
    assert report["any_mapped"] is True
    text = format_report(report)
    assert "flux3-image-klein-base" in text and "moss-image-base" in text
    assert "0.00-0.25" in text
    assert "Not mapped:" not in text


def test_the_text_shows_a_family_without_flow_sampling_on_its_own_draw():
    report = build_report(["sdxl_base_1.0"], [parse_sampler("default")], [1024], **OPTIONS)
    line = next(text for text in format_report(report).splitlines() if text.strip().startswith("default @"))
    assert "own" in line and "logit_normal" not in line


def test_the_text_shows_a_scheduler_fallback(monkeypatch):
    def raising(self):
        raise RuntimeError("needs the loaded betas")

    monkeypatch.setattr(type(sampling_shell(registry.get_definition("moss-image-base"), {}, 1)),
                        "init_scheduler", raising)
    report = build_report(["moss-image-base"], [parse_sampler("uniform")], [1024], **OPTIONS)
    assert report["rows"][0]["scheduler_fallback"] == "RuntimeError: needs the loaded betas"
    assert "RuntimeError: needs the loaded betas" in format_report(report)


def test_the_text_lists_what_was_not_mapped():
    report = build_report(["sdxl_base_1.0"], [parse_sampler("uniform")], [1024], **OPTIONS)
    assert report["any_mapped"] is False
    text = format_report(report)
    assert "Not mapped:" in text
    assert report["rows"][0]["reason"] in text
