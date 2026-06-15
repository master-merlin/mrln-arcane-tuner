"""LTX 2.3 loader manifest tests.

Pins each component's ``subfolder`` to the real Lightricks/LTX-2 diffusers
repo layout (as declared in the repo's ``model_index.json`` and present on
disk).  A wrong subfolder name combined with ``fallback_to_root=True`` does
NOT raise where the mistake is — it silently falls back to the repo root,
which has only ``model_index.json`` (no ``config.json``), producing a
confusing "no file named config.json" failure deep in ``from_pretrained``.
This guard keeps the manifest honest about where each component lives.
"""

from app.engine.models.families.ltx2.loader import Ltx2Loader


def test_ltx2_component_subfolders_match_repo_layout():
    loader = Ltx2Loader("cpu", train_audio=True)
    manifest = loader.get_component_manifest(None)
    subfolders = {spec.key: spec.subfolder for spec in manifest}

    assert subfolders == {
        "tokenizer": "tokenizer",
        "text_encoder": "text_encoder",
        # The text connectors live in ``connectors/`` — NOT ``text_connectors/``.
        # model_index.json declares: "connectors": ["ltx2", "LTX2TextConnectors"].
        "connectors": "connectors",
        "vae": "vae",
        "unet": "transformer",
        "audio_vae": "audio_vae",
        "vocoder": "vocoder",
    }


def test_ltx2_connectors_subfolder_is_connectors():
    """Focused regression for the original load failure."""
    loader = Ltx2Loader("cpu")
    manifest = loader.get_component_manifest(None)
    connectors = next(s for s in manifest if s.key == "connectors")

    assert connectors.subfolder == "connectors"
