"""Component-size lookup rules for the VRAM estimator.

``model_size_mb`` is a per-component map. The primary trainable component is
named ``transformer`` by some definitions and ``unet`` by others, so a lookup
for one must answer for the other. Every OTHER component must resolve on its
exact key only.

Regression: the alias walk used to run for every key, so a definition shipping
a primary size but no ``text_encoder``/``vae`` had the transformer's size
returned as the size of BOTH — inflating ``caching_peak_mb`` (= te + vae +
overhead) by roughly two transformers and, because ``peak_mb`` is
``max(training, caching)``, producing a false "won't fit" verdict. The two
shipped definitions that match this shape are pinned below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.engine.utils.vram_estimator import _get_component_disk_mb

# Anchored on this file, not the CWD: the suite is run both from ``backend/``
# and from the repo root (``pytest backend``), and a relative glob silently
# returns [] in the second case — which would turn every definition assertion
# below into a vacuous pass rather than a failure.
_FAMILIES = Path(__file__).resolve().parents[1] / "app/engine/models/families"
_DEFS = sorted(str(p) for p in _FAMILIES.glob("*/definitions/*.yaml"))


class TestComponentDiskLookup:
    def test_primary_aliases_are_interchangeable(self):
        assert _get_component_disk_mb({"transformer": 100.0}, "unet") == 100.0
        assert _get_component_disk_mb({"unet": 100.0}, "transformer") == 100.0

    @pytest.mark.parametrize("component", ["text_encoder", "vae"])
    def test_non_primary_never_inherits_the_primary_size(self, component):
        """A missing te/vae must return 0 (→ param-count fallback), never the
        transformer's on-disk size."""
        assert _get_component_disk_mb({"transformer": 27275.0}, component) == 0.0

    def test_exact_key_still_wins_for_non_primary(self):
        size_mb = {"transformer": 27275.0, "text_encoder": 11400.0, "vae": 254.0}
        assert _get_component_disk_mb(size_mb, "text_encoder") == 11400.0
        assert _get_component_disk_mb(size_mb, "vae") == 254.0


class TestPartialSizeDefinitions:
    """The estimate for a partial-``model_size_mb`` definition must stay in a
    physically plausible band instead of double-counting the transformer."""

    @pytest.mark.parametrize(
        "def_path,max_caching_gb",
        [
            ("bernini_r/definitions/bernini_r_14b.yaml", 20.0),
            ("kandinsky5/definitions/k5_i2v_pro_sft_5s.yaml", 26.0),
        ],
    )
    def test_caching_peak_is_not_two_transformers(self, def_path, max_caching_gb):
        from app.engine.utils.vram_estimator import VRAMEstimator

        matches = [p for p in _DEFS if p.replace("\\", "/").endswith(def_path)]
        assert matches, f"definition not found: {def_path}"
        data = yaml.safe_load(open(matches[0], encoding="utf-8"))

        class _Defn:
            family = data.get("family")
            detected_precision = data.get("detected_precision") or {}
            architecture_params = data.get("architecture_params") or {}
            model_size_mb = data.get("model_size_mb") or {}

        report = VRAMEstimator.estimate(_Defn(), {"lora_rank": 32, "batch_size": 1})
        primary_mb = float(
            _Defn.model_size_mb.get("transformer")
            or _Defn.model_size_mb.get("unet")
            or 0.0
        )
        assert primary_mb > 0, "fixture no longer has a partial size map"
        # The caching phase holds the TE + VAE, never the trainable backbone.
        assert report.caching_peak_mb < primary_mb
        assert report.caching_peak_mb / 1024 < max_caching_gb
