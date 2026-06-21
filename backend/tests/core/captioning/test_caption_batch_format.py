# backend/tests/core/captioning/test_caption_batch_format.py
import json
import types
from app.core.captioning import caption_batch as cb


def _make_dataset(tmp_path):
    return types.SimpleNamespace(path=str(tmp_path), media_metadata={})


def test_structured_definition_normalizes_caption_before_write(tmp_path, monkeypatch):
    written = {}

    # Fake service returns messy JSON with x-first style fields out of order.
    raw = json.dumps(
        {
            "high_level_description": "hello",
            "style_description": {
                "aesthetics": "a",
                "lighting": "l",
                "medium": "Painting.",
                "photo": "p",
                "color_palette": ["#abc"],
            },
            "compositional_deconstruction": {"background": "bg", "elements": []},
        }
    )

    class _Svc:
        def generate_caption(self, **kw):
            return raw

    monkeypatch.setattr(cb, "_get_service", lambda: _Svc())
    monkeypatch.setattr(cb, "_full_path", lambda ds, rel: f"/img/{rel}")
    monkeypatch.setattr(cb, "_video_meta", lambda ds, rel: {})
    monkeypatch.setattr(cb, "_emit_caption_written", lambda **kw: None)

    def _capture(dataset_name, rel, text, target, definition_id=None):
        written[rel] = text

    monkeypatch.setattr(cb, "_write_caption", _capture)

    # task_manager stubs
    monkeypatch.setattr(cb.task_manager, "is_cancelled", lambda tid: False)
    monkeypatch.setattr(cb.task_manager, "update", lambda *a, **k: None)
    monkeypatch.setattr(cb.task_manager, "complete", lambda tid: None)

    # Make the definition resolve to the ideogram4 format.
    from app.core.captioning import formats

    monkeypatch.setattr(
        cb,
        "_get_caption_format",
        lambda definition_id: formats.Ideogram4Format(),
        raising=False,
    )

    cb.run_caption_batch(
        "t1",
        dataset_name="ds",
        image_rel_paths=["a.png"],
        model_id="qwen3-vl-8B-Instruct",
        params={"definition_id": "ideogram4-fp8"},
        system_prompt=None,
        target="original",
        definition_id="ideogram4-fp8",
    )

    out = json.loads(written["a.png"])
    # medium canonicalized + compact + key order normalized
    assert out["style_description"]["medium"] == "painting"
    assert list(out.keys())[0] == "high_level_description"


def test_plain_definition_is_unchanged(tmp_path, monkeypatch):
    written = {}

    class _Svc:
        def generate_caption(self, **kw):
            return "a, b, c"

    monkeypatch.setattr(cb, "_get_service", lambda: _Svc())
    monkeypatch.setattr(cb, "_full_path", lambda ds, rel: f"/img/{rel}")
    monkeypatch.setattr(cb, "_video_meta", lambda ds, rel: {})
    monkeypatch.setattr(cb, "_emit_caption_written", lambda **kw: None)
    monkeypatch.setattr(
        cb,
        "_write_caption",
        lambda dn, rel, text, target, definition_id=None: written.__setitem__(
            rel, text
        ),
    )
    monkeypatch.setattr(cb.task_manager, "is_cancelled", lambda tid: False)
    monkeypatch.setattr(cb.task_manager, "update", lambda *a, **k: None)
    monkeypatch.setattr(cb.task_manager, "complete", lambda tid: None)

    cb.run_caption_batch(
        "t1",
        dataset_name="ds",
        image_rel_paths=["a.png"],
        model_id="qwen3-vl-8B-Instruct",
        params={},
        system_prompt=None,
        target="original",
    )
    assert written["a.png"] == "a, b, c"
