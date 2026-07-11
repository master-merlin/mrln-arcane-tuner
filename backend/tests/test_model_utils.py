"""
Tests for ModelPathResolver — covers local paths, HuggingFace URIs, find_component.
"""

import os
from unittest.mock import patch, MagicMock


from app.engine.utils.model_utils import ModelPathResolver


class TestResolveLocal:
    def test_empty_string_returns_none(self):
        assert ModelPathResolver.resolve("") is None

    def test_none_returns_none(self):
        assert ModelPathResolver.resolve(None) is None

    def test_absolute_path_returned_as_is(self):
        path = os.path.abspath("/some/absolute/path")
        result = ModelPathResolver.resolve(path)
        assert result == path

    def test_relative_path_joined_with_base(self, tmp_path):
        result = ModelPathResolver.resolve("model.safetensors", base_dir=str(tmp_path))
        assert result == os.path.join(str(tmp_path), "model.safetensors")

    def test_relative_path_without_base_uses_cwd(self):
        result = ModelPathResolver.resolve("relative/path")
        assert os.path.isabs(result)


class TestResolveHuggingFace:
    # ── Snapshot WARM-CACHE fast path (W5-2) ────────────────────────────
    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=True)
    @patch(
        "app.engine.utils.model_utils.snapshot_download",
        return_value="/cache/repo",
    )
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_fully_cached_snapshot_resolves_in_process_no_spawn(
        self, mock_guard, mock_inproc, _mock_cached,
    ):
        """A snapshot confirmed fully-cached is resolved IN-PROCESS with
        ``local_files_only=True`` — the killable child (the ~2-4s spawn paid
        twice per job) is skipped entirely. Nothing can stall because an
        offline resolve never touches the network."""
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        # spawn entry point (download_with_stall_guard) is the thing NOT called
        mock_guard.assert_not_called()
        mock_inproc.assert_called_once_with(
            repo_id="org/model", local_files_only=True,
        )

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=True)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch(
        "app.engine.utils.model_utils.snapshot_download",
        return_value="/cache/repo",
    )
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_fully_cached_snapshot_skips_byte_progress(
        self, mock_guard, mock_inproc, mock_progress, _mock_cached,
    ):
        """A fully-cached snapshot is LOADED, not transferred — the top-bar
        byte-progress poller must never start (no spurious flash)."""
        ModelPathResolver.resolve("huggingface:org/model")
        mock_progress.assert_not_called()

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=True)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch(
        "app.engine.utils.model_utils.snapshot_download",
        side_effect=OSError("cache evicted mid-resolve"),
    )
    @patch(
        "app.engine.utils.model_utils.download_with_stall_guard",
        return_value="/cache/guarded",
    )
    def test_snapshot_fastpath_error_falls_through_to_guard(
        self, mock_guard, mock_inproc, mock_progress, _mock_cached,
    ):
        """SAFETY RAIL: if the in-process fast-path resolve unexpectedly
        raises (e.g. the cache was evicted between the manifest check and the
        offline resolve), fall through to the guarded download — the
        optimization must NEVER fail a resolve."""
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/guarded"
        mock_inproc.assert_called_once()
        mock_guard.assert_called_once_with(repo_id="org/model", revision=None)

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch("app.engine.utils.model_utils.snapshot_download")
    @patch(
        "app.engine.utils.model_utils.download_with_stall_guard",
        return_value="/cache/repo",
    )
    def test_uncached_snapshot_uses_guarded_child_with_progress(
        self, mock_guard, mock_inproc, mock_progress, _mock_cached,
    ):
        """A snapshot NOT fully cached (partial or absent) takes the guarded
        killable/resumable child path wrapped in byte progress — NEVER an
        in-process resolve (an in-process HF call can't be stall-aborted)."""
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        mock_guard.assert_called_once_with(repo_id="org/model", revision=None)
        mock_inproc.assert_not_called()
        mock_progress.assert_called_once()
        assert mock_progress.call_args.kwargs.get("repo_id") == "org/model"

    # ── Single-file WARM-CACHE fast path (W5-2) ─────────────────────────
    @patch("app.engine.utils.model_utils._file_fully_cached", return_value=True)
    @patch(
        "app.engine.utils.model_utils.hf_hub_download",
        return_value="/cache/model.safetensors",
    )
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_fully_cached_single_file_resolves_in_process_no_spawn(
        self, mock_guard, mock_inproc, _mock_cached,
    ):
        """A single file already complete in the cache resolves in-process —
        no child spawn. ``try_to_load_from_cache`` (behind ``_file_fully_cached``)
        is a purely-offline per-blob check, so no Hub manifest call is needed."""
        result = ModelPathResolver.resolve(
            "huggingface:org/model:model.safetensors",
        )
        assert result == "/cache/model.safetensors"
        mock_guard.assert_not_called()
        mock_inproc.assert_called_once_with(
            repo_id="org/model", filename="model.safetensors",
            local_files_only=True,
        )

    @patch("app.engine.utils.model_utils._file_fully_cached", return_value=False)
    @patch("app.engine.utils.model_utils.hf_hub_download")
    @patch(
        "app.engine.utils.model_utils.download_with_stall_guard",
        return_value="/cache/model.safetensors",
    )
    def test_uncached_single_file_uses_guarded_child(
        self, mock_guard, mock_inproc, _mock_cached,
    ):
        """huggingface:<repo>:<file> not in cache runs the guarded
        hf_hub_download (killable child), not an in-process call."""
        result = ModelPathResolver.resolve(
            "huggingface:org/model:model.safetensors",
        )
        assert result == "/cache/model.safetensors"
        mock_guard.assert_called_once_with(
            repo_id="org/model", filename="model.safetensors", revision=None,
        )
        mock_inproc.assert_not_called()

    @patch("app.engine.utils.model_utils._file_fully_cached", return_value=True)
    @patch(
        "app.engine.utils.model_utils.hf_hub_download",
        side_effect=OSError("blob evicted"),
    )
    @patch(
        "app.engine.utils.model_utils.download_with_stall_guard",
        return_value="/cache/guarded-file",
    )
    def test_single_file_fastpath_error_falls_through_to_guard(
        self, mock_guard, mock_inproc, _mock_cached,
    ):
        """SAFETY RAIL (single file): an in-process fast-path resolve that
        raises falls through to the guarded download."""
        result = ModelPathResolver.resolve(
            "huggingface:org/model:model.safetensors",
        )
        assert result == "/cache/guarded-file"
        mock_guard.assert_called_once()

    # ── Offline (skip-update) — unchanged by the fast path ──────────────
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_hf_offline_uses_local_cache_only(self, mock_download, mock_guard):
        """Offline / skip-update mode reads the cache and never downloads —
        and never touches the guard (nothing to stall on offline)."""
        mock_download.return_value = "/cache/repo"
        result = ModelPathResolver.resolve(
            "huggingface:org/model", local_files_only=True,
        )
        mock_download.assert_called_once_with(
            repo_id="org/model", local_files_only=True,
        )
        assert result == "/cache/repo"
        mock_guard.assert_not_called()


class TestSnapshotFullyCached:
    """``_snapshot_fully_cached`` — the completeness signal that GATES the
    snapshot warm-cache fast path (and the byte-progress bar). It cross-checks
    the Hub's authoritative file list against ``try_to_load_from_cache`` so a
    *partial* snapshot reports False (a bare ``local_files_only`` probe would
    accept it as complete)."""

    @patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/f")
    @patch("huggingface_hub.HfApi")
    def test_all_files_present_is_true(self, mock_api, _mock_ttl):
        from app.engine.utils.model_utils import _snapshot_fully_cached

        mock_api.return_value.list_repo_files.return_value = ["a.json", "b.bin"]
        assert _snapshot_fully_cached("org/repo") is True

    @patch("huggingface_hub.HfApi")
    def test_one_file_missing_is_false(self, mock_api):
        """Partial download: one file not materialized → NOT cached →
        guarded (resumable) path."""
        from app.engine.utils.model_utils import _snapshot_fully_cached

        mock_api.return_value.list_repo_files.return_value = ["a.json", "b.bin"]
        with patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=["/cache/a.json", None],
        ):
            assert _snapshot_fully_cached("org/repo") is False

    @patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/f")
    @patch("huggingface_hub.HfApi")
    def test_revision_threaded_through(self, mock_api, mock_ttl):
        """A pinned revision must be passed to BOTH the manifest listing and
        each cache probe (a different revision cached is NOT this one)."""
        from app.engine.utils.model_utils import _snapshot_fully_cached

        mock_api.return_value.list_repo_files.return_value = ["a.json"]
        assert _snapshot_fully_cached("org/repo", revision="diffusers") is True
        assert (
            mock_api.return_value.list_repo_files.call_args.kwargs.get("revision")
            == "diffusers"
        )
        assert mock_ttl.call_args.kwargs.get("revision") == "diffusers"

    @patch("huggingface_hub.HfApi")
    def test_empty_file_list_is_false(self, mock_api):
        from app.engine.utils.model_utils import _snapshot_fully_cached

        mock_api.return_value.list_repo_files.return_value = []
        assert _snapshot_fully_cached("org/repo") is False

    @patch("huggingface_hub.HfApi")
    def test_network_error_is_false(self, mock_api):
        """Any error (offline, repo removed, auth) → False: prefer the
        guarded download over a misleading silent/broken fast path."""
        from app.engine.utils.model_utils import _snapshot_fully_cached

        mock_api.return_value.list_repo_files.side_effect = RuntimeError("offline")
        assert _snapshot_fully_cached("org/repo") is False


class TestFileFullyCached:
    """``_file_fully_cached`` — purely-offline per-blob completeness check that
    gates the single-file fast path (no Hub manifest call needed for one
    file: ``try_to_load_from_cache`` returns a path only for a complete blob,
    never a partial ``*.incomplete``)."""

    @patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/f.bin")
    def test_present_blob_is_true(self, _mock_ttl):
        from app.engine.utils.model_utils import _file_fully_cached

        assert _file_fully_cached("org/repo", "f.bin") is True

    @patch("huggingface_hub.try_to_load_from_cache", return_value=None)
    def test_missing_blob_is_false(self, _mock_ttl):
        from app.engine.utils.model_utils import _file_fully_cached

        assert _file_fully_cached("org/repo", "f.bin") is False

    def test_known_nonexistent_sentinel_is_false(self):
        """``try_to_load_from_cache`` returns a non-str sentinel for a
        known-missing file — that is NOT a cache hit."""
        from app.engine.utils.model_utils import _file_fully_cached

        sentinel = object()
        with patch(
            "huggingface_hub.try_to_load_from_cache", return_value=sentinel,
        ):
            assert _file_fully_cached("org/repo", "f.bin") is False

    @patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/f.bin")
    def test_revision_threaded_through(self, mock_ttl):
        from app.engine.utils.model_utils import _file_fully_cached

        assert _file_fully_cached("org/repo", "f.bin", revision="rev1") is True
        assert mock_ttl.call_args.kwargs.get("revision") == "rev1"

    def test_error_is_false(self):
        from app.engine.utils.model_utils import _file_fully_cached

        with patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=RuntimeError("boom"),
        ):
            assert _file_fully_cached("org/repo", "f.bin") is False


class TestEnsureDefinitionCached:
    """Pre-fetch a definition's HF components in-process (so the trainer
    subprocess loads from a warm cache and the top-bar download bar updates)."""

    @staticmethod
    def _defn(components: dict, definition_id: str = "fake-model"):
        defn = MagicMock()
        defn.id = definition_id
        comps = {}
        for key, path in components.items():
            comp = MagicMock()
            comp.path = path
            comps[key] = comp
        defn.components = comps
        return defn

    @patch("app.engine.utils.model_override_manager.ModelOverrideManager.resolve_effective_source")
    @patch.object(ModelPathResolver, "resolve")
    def test_prefetches_each_hf_component(self, mock_resolve, mock_source):
        from app.core.schemas.model_overrides import ModelSourceType
        mock_source.return_value = (ModelSourceType.HF_HUB, None, False)
        defn = self._defn({
            "repo": "huggingface:org/dit",
            "text_encoder": "huggingface:org/te",
        })

        ModelPathResolver.ensure_definition_cached(defn)

        resolved = {c.args[0] for c in mock_resolve.call_args_list}
        assert resolved == {"huggingface:org/dit", "huggingface:org/te"}
        # Online → cache-only must NOT be forced (would mask a partial cache).
        for c in mock_resolve.call_args_list:
            assert c.kwargs.get("local_files_only") is False

    @patch("app.engine.utils.model_override_manager.ModelOverrideManager.resolve_effective_source")
    @patch.object(ModelPathResolver, "resolve")
    def test_skips_local_override(self, mock_resolve, mock_source):
        from app.core.schemas.model_overrides import ModelSourceType
        mock_source.return_value = (ModelSourceType.LOCAL_DIFFUSERS, "/local/model", False)
        defn = self._defn({"repo": "huggingface:org/dit"})

        ModelPathResolver.ensure_definition_cached(defn)

        mock_resolve.assert_not_called()

    @patch("app.engine.utils.model_override_manager.ModelOverrideManager.resolve_effective_source")
    @patch.object(ModelPathResolver, "resolve")
    def test_skip_update_resolves_cache_only(self, mock_resolve, mock_source):
        from app.core.schemas.model_overrides import ModelSourceType
        mock_source.return_value = (ModelSourceType.HF_HUB, None, True)
        defn = self._defn({"repo": "huggingface:org/dit"})

        ModelPathResolver.ensure_definition_cached(defn)

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs.get("local_files_only") is True

    @patch("app.engine.utils.model_override_manager.ModelOverrideManager.resolve_effective_source")
    @patch.object(ModelPathResolver, "resolve")
    def test_ignores_non_hf_components(self, mock_resolve, mock_source):
        from app.core.schemas.model_overrides import ModelSourceType
        mock_source.return_value = (ModelSourceType.HF_HUB, None, False)
        defn = self._defn({
            "repo": "huggingface:org/dit",
            "vae": "/abs/local/vae.safetensors",
        })

        ModelPathResolver.ensure_definition_cached(defn)

        resolved = {c.args[0] for c in mock_resolve.call_args_list}
        assert resolved == {"huggingface:org/dit"}


class TestFindComponent:
    def test_explicit_definition(self, tmp_path):
        """If definition.components has the key, use it."""
        comp = MagicMock()
        comp.path = str(tmp_path / "vae")
        defn = MagicMock()
        defn.components = {"vae": comp}

        result = ModelPathResolver.find_component(defn, "vae")
        assert result == str(tmp_path / "vae")

    def test_discovery_fallback(self, tmp_path):
        """If not in definition, scan root_path for candidate files."""
        defn = MagicMock()
        defn.components = {}

        vae_dir = tmp_path / "vae"
        vae_dir.mkdir()

        result = ModelPathResolver.find_component(
            defn, "vae",
            root_path=str(tmp_path),
            candidates=["vae", "vae_model"]
        )
        assert result == str(vae_dir)

    def test_no_match_returns_none(self, tmp_path):
        """If nothing is found, return None."""
        defn = MagicMock()
        defn.components = {}
        result = ModelPathResolver.find_component(
            defn, "vae",
            root_path=str(tmp_path),
            candidates=["nonexistent"]
        )
        assert result is None

    def test_no_root_no_candidates_returns_none(self):
        """Without root_path or candidates, return None."""
        defn = MagicMock()
        defn.components = {}
        result = ModelPathResolver.find_component(defn, "vae")
        assert result is None
