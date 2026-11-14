"""The browser check's mocks for the timestep defaults are the real backend output.

frontend/e2e/specs/timestep-defaults.spec.ts drives the real Training form against mocks in
frontend/e2e/fixtures/api-data.ts. Its block between the two markers below is generated from the
backend: ``resolve_capabilities`` for MOSS-Image (base and SFT: logit-normal 1.09 / 0.95), FLUX.2
Klein Base 4B (no timestep default: the schema's) and FLUX.3 Klein Base (``uniform``), and the
training schema's own properties for the fields the spec checks. The test fails when the backend
changes and the mocks do not, and its message carries the block to paste.
"""

from __future__ import annotations

import json
import pathlib

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.registry import registry

E2E_FIXTURE = pathlib.Path(__file__).resolve().parents[3] / "frontend" / "e2e" / "fixtures" / "api-data.ts"
E2E_BEGIN = "// ── BEGIN GENERATED: timestep defaults (backend/tests/engine/test_timestep_defaults_e2e_mocks.py) ──"
E2E_END = "// ── END GENERATED: timestep defaults ──"
E2E_DEFINITIONS = ("moss-image-base", "moss-image-sft", "flux2-klein-base-4b", "flux3-image-klein-base")
E2E_SCHEMA_KEYS = ("timestep_sampling", "logit_normal_mu", "logit_normal_sigma")


def _e2e_data() -> dict[str, object]:
    from app.engine.models.training_plugin import Config

    registry.initialize()
    definitions = [registry.get_definition(i) for i in E2E_DEFINITIONS]
    properties = Config.model_json_schema()["properties"]
    return {
        "timestepModels": [
            {"id": d.id, "name": d.name, "family": d.family, "architecture_params": {}} for d in definitions
        ],
        "timestepCapabilities": {d.id: resolve_capabilities(d) for d in definitions},
        "timestepSchemaProperties": {k: properties[k] for k in E2E_SCHEMA_KEYS},
    }


def e2e_block() -> str:
    """The generated block of api-data.ts, markers included, as the backend makes it now."""
    lines = [E2E_BEGIN]
    for name, value in _e2e_data().items():
        lines.append(f"export const {name} = {json.dumps(value, indent=4, ensure_ascii=False)};")
    lines.append(E2E_END)
    return "\n".join(lines)


def test_the_e2e_mocks_are_the_real_backend_output():
    text = E2E_FIXTURE.read_text(encoding="utf-8")
    assert E2E_BEGIN in text and E2E_END in text, f"{E2E_FIXTURE.name} has no generated timestep block"
    current = text[text.index(E2E_BEGIN) : text.index(E2E_END) + len(E2E_END)]
    expected = e2e_block()
    assert current == expected, (
        f"{E2E_FIXTURE.name}'s timestep mocks are not what the backend returns; replace its "
        f"generated block with:\n{expected}"
    )


def test_the_e2e_mocks_cover_each_case_the_spec_checks():
    caps = _e2e_data()["timestepCapabilities"]
    moss = caps["moss-image-base"]["defaults"]
    assert (moss["timestep_sampling"], moss["logit_normal_mu"], moss["logit_normal_sigma"]) == (
        "logit_normal", 1.09, 0.95)
    assert caps["moss-image-sft"]["defaults"]["logit_normal_mu"] == 1.09
    flux2 = caps["flux2-klein-base-4b"]["defaults"]
    assert not {"timestep_sampling", "logit_normal_mu", "logit_normal_sigma"} & set(flux2)
    assert caps["flux3-image-klein-base"]["defaults"]["timestep_sampling"] == "uniform"
