"""Cache-correctness and tooling regressions from the full-backend review.

Four independent defects, all of the same shape — a check that looked right but
could never succeed, or a resource whose cleanup assumed strict nesting:

* ``_is_repo_cached`` probed only for a root ``config.json``; diffusers repos
  ship ``model_index.json`` instead, so the download-bar suppression it exists
  to provide never fired for most of this app's models.
* ``load_quantized`` compared the cache's POST-quantization parameter count
  against the freshly-loaded PRE-quantization module, which packing backends
  can never satisfy.
* ``_capture_per_file`` patched a process-global and restored it in
  ``finally`` — correct only while calls strictly nest.
* ``resize_lora`` wrote its output non-atomically, and ``inspect_lora``
  computed every layer's delta matmul twice.
"""

from __future__ import annotations

import sys

import pytest
import torch


# ── HF cache detection ────────────────────────────────────────────────────


class TestIsRepoCached:
    def _patch_cache(self, monkeypatch, present: set[str]):
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub,
            "try_to_load_from_cache",
            lambda repo_id, filename, **kw: (
                f"/cache/{repo_id}/{filename}" if filename in present else None
            ),
            raising=False,
        )

    def test_diffusers_repo_is_detected(self, monkeypatch):
        """model_index.json is the diffusers manifest — a diffusers repo has no
        ROOT config.json at all (component configs live in subfolders)."""
        from app.api.events import download_progress as dp

        self._patch_cache(monkeypatch, {"model_index.json"})
        assert dp._is_repo_cached("Some/Diffusers-Repo") is True

    def test_single_model_repo_is_still_detected(self, monkeypatch):
        from app.api.events import download_progress as dp

        self._patch_cache(monkeypatch, {"config.json"})
        assert dp._is_repo_cached("Some/Transformers-Repo") is True

    def test_uncached_repo_is_not_detected(self, monkeypatch):
        from app.api.events import download_progress as dp

        self._patch_cache(monkeypatch, set())
        assert dp._is_repo_cached("Some/Missing-Repo") is False


# ── quantized-weight disk cache ───────────────────────────────────────────


class _Packed(torch.nn.Module):
    """Stands in for a packing backend: half the elements after quantization."""

    def __init__(self, n: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(n))


class TestQuantizedCacheParamGuard:
    def test_packing_backend_cache_is_not_written(self, tmp_path):
        """A packed state dict can never be loaded back into the unquantized
        module ``load_quantized`` receives, so writing the cache was pure cost:
        a multi-GB serialization every run that always ended in a silent miss.
        It must be skipped up front, not written and then rejected."""
        from app.engine.factories.quantization import QuantizationFactory

        unquantized = _Packed(100)
        packed = _Packed(50)  # packing backend: fewer elements, same key

        cache = tmp_path / "c"
        QuantizationFactory.save_quantized(
            packed, str(cache), "nf4",
            source_param_count=100,
            source_signature=QuantizationFactory.state_shape_signature(unquantized),
        )
        assert not (cache / "metadata.json").exists()
        assert QuantizationFactory.load_quantized(
            _Packed(100), str(cache), "nf4"
        ) is None

    def test_structure_preserving_backend_cache_round_trips(self, tmp_path):
        """torchao-style quantization keeps the state-dict layout, so its cache
        is written and loads back."""
        from app.engine.factories.quantization import QuantizationFactory

        unquantized = _Packed(100)
        quantized = _Packed(100)  # same layout, different values

        cache = tmp_path / "c"
        QuantizationFactory.save_quantized(
            quantized, str(cache), "int8",
            source_param_count=100,
            source_signature=QuantizationFactory.state_shape_signature(unquantized),
        )
        assert (cache / "metadata.json").exists()
        assert QuantizationFactory.load_quantized(
            _Packed(100), str(cache), "int8"
        ) is not None

    def test_genuine_shape_change_still_rejected(self, tmp_path):
        from app.engine.factories.quantization import QuantizationFactory

        cache = tmp_path / "c"
        QuantizationFactory.save_quantized(
            _Packed(50), str(cache), "nf4", source_param_count=100
        )
        # A different model entirely — 80 source params, not 100.
        assert QuantizationFactory.load_quantized(
            _Packed(80), str(cache), "nf4"
        ) is None

    def test_scheme_mismatch_still_rejected(self, tmp_path):
        from app.engine.factories.quantization import QuantizationFactory

        cache = tmp_path / "c"
        QuantizationFactory.save_quantized(
            _Packed(50), str(cache), "nf4", source_param_count=100
        )
        assert QuantizationFactory.load_quantized(
            _Packed(100), str(cache), "int8"
        ) is None

    @pytest.mark.parametrize("bad_id", ["..", "../..", "./.."])
    def test_cache_path_cannot_escape_the_cache_root(self, bad_id):
        from app.engine.factories.quantization import QuantizationFactory

        root = QuantizationFactory._get_cache_root()
        import os

        resolved = os.path.abspath(
            QuantizationFactory.resolve_cache_path(bad_id, "transformer", "nf4")
        )
        assert resolved.startswith(os.path.abspath(root))


# ── per-file tqdm capture ─────────────────────────────────────────────────


class TestCapturePerFileNesting:
    def _mod(self):
        import huggingface_hub.utils.tqdm  # noqa: F401

        return sys.modules["huggingface_hub.utils.tqdm"]

    def test_out_of_order_exit_restores_the_original(self):
        """Two concurrent captures finishing in START order used to leave the
        first one's subclass installed for the life of the process."""
        from app.api.events.download_progress import (
            SnapshotProgressRegistry,
            _capture_per_file,
        )

        mod = self._mod()
        original = mod.tqdm

        a = _capture_per_file(SnapshotProgressRegistry(100))
        b = _capture_per_file(SnapshotProgressRegistry(200))
        a.__enter__()
        b.__enter__()
        assert mod.tqdm is not original  # patched while any capture is active
        a.__exit__(None, None, None)     # first-in exits FIRST (not nested)
        assert mod.tqdm is not original  # b is still live — keep capturing
        b.__exit__(None, None, None)
        assert mod.tqdm is original

    def test_single_capture_restores_the_original(self):
        from app.api.events.download_progress import (
            SnapshotProgressRegistry,
            _capture_per_file,
        )

        mod = self._mod()
        original = mod.tqdm
        with _capture_per_file(SnapshotProgressRegistry(1)):
            assert mod.tqdm is not original
        assert mod.tqdm is original

    def test_bars_are_unattributed_while_two_captures_overlap(self):
        """With two downloads in flight a bar cannot be attributed to either,
        so it must go to neither rather than to the wrong one."""
        from app.api.events.download_progress import (
            SnapshotProgressRegistry,
            _capture_per_file,
            _current_registry,
        )

        reg_a = SnapshotProgressRegistry(100)
        reg_b = SnapshotProgressRegistry(200)
        with _capture_per_file(reg_a):
            assert _current_registry() is reg_a
            with _capture_per_file(reg_b):
                assert _current_registry() is None
            assert _current_registry() is reg_a
        assert _current_registry() is None


# ── LoRA tooling ──────────────────────────────────────────────────────────


def _write_lora(path, rank=4, dim=8, modules=3):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    sd = {}
    for i in range(modules):
        sd[f"lora_unet_block{i}.lora_A.weight"] = torch.randn(rank, dim)
        sd[f"lora_unet_block{i}.lora_B.weight"] = torch.randn(dim, rank)
    save_file(sd, str(path), metadata={"ss_network_dim": str(rank)})
    return sd


class TestLoraTooling:
    def test_norm_summary_agrees_with_layer_details(self, tmp_path):
        """The summary is now derived from layer_details instead of redoing
        every delta matmul — the numbers must not move."""
        from app.engine.utils.lora_tools import inspect_lora

        path = tmp_path / "a.safetensors"
        _write_lora(path)
        out = inspect_lora(str(path))

        details = out["layer_details"]
        summary = out["norm_summary"]
        norms = [d["norm_delta"] for d in details]
        assert summary["total_layers"] == len(details)
        assert summary["max_norm"] == pytest.approx(max(norms))
        assert summary["min_norm"] == pytest.approx(min(norms))
        assert summary["mean_norm"] == pytest.approx(
            sum(norms) / len(norms), rel=1e-5
        )

    def test_resize_round_trips_and_leaves_no_tmp_file(self, tmp_path):
        from app.engine.utils.lora_tools import inspect_lora, resize_lora

        src = tmp_path / "in.safetensors"
        dst = tmp_path / "out.safetensors"
        _write_lora(src, rank=4, dim=8)

        result = resize_lora(str(src), str(dst), new_rank=2)
        assert result["new_rank"] == 2
        assert dst.exists()
        assert not (tmp_path / "out.safetensors.tmp").exists()
        assert inspect_lora(str(dst))["rank"] == 2

    def test_resize_rejects_an_uncontractable_pair(self, tmp_path):
        """A real spatial kernel on the UP weight cannot contract; that must be
        a named error, not a bare torch shape RuntimeError."""
        from safetensors.torch import save_file

        from app.engine.utils.lora_tools import resize_lora

        src = tmp_path / "conv.safetensors"
        save_file(
            {
                "lora_unet_c.lora_A.weight": torch.randn(4, 8, 3, 3),
                "lora_unet_c.lora_B.weight": torch.randn(8, 4, 3, 3),
            },
            str(src),
        )
        with pytest.raises(ValueError, match="does not contract"):
            resize_lora(str(src), str(tmp_path / "o.safetensors"), new_rank=2)

    def test_standard_conv_layout_still_resizes(self, tmp_path):
        """Kernel on A, 1x1 on B — the layout kohya actually emits."""
        from safetensors.torch import save_file

        from app.engine.utils.lora_tools import resize_lora

        src = tmp_path / "conv.safetensors"
        save_file(
            {
                "lora_unet_c.lora_A.weight": torch.randn(4, 8, 3, 3),
                "lora_unet_c.lora_B.weight": torch.randn(16, 4, 1, 1),
            },
            str(src),
        )
        out = resize_lora(str(src), str(tmp_path / "o.safetensors"), new_rank=2)
        assert out["modules_resized"] == 1


# ── orphan removal ────────────────────────────────────────────────────────


def test_text_encoding_module_exposes_only_the_live_type():
    """TextEncodingCache/_CacheEntry were never instantiated anywhere in the
    app; TextEncoderOutput (46 files) is the live part of the module."""
    from app.engine.core import text_encoding

    assert hasattr(text_encoding, "TextEncoderOutput")
    assert not hasattr(text_encoding, "TextEncodingCache")
    assert not hasattr(text_encoding, "_CacheEntry")
