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
    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=True)
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_hf_full_repo(self, mock_guard, _mock_cached):
        """huggingface:<repo_id> resolves via the killable stall guard, not
        an in-process snapshot_download — an in-process call can't be
        aborted on a stall (Python threads/socket reads are un-abortable)."""
        mock_guard.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        mock_guard.assert_called_once_with(repo_id="org/model", revision=None)

    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_hf_single_file(self, mock_guard):
        """huggingface:<repo_id>:<filename> runs the guarded hf_hub_download."""
        mock_guard.return_value = "/cache/model.safetensors"
        result = ModelPathResolver.resolve("huggingface:org/model:model.safetensors")
        assert result == "/cache/model.safetensors"
        mock_guard.assert_called_once_with(
            repo_id="org/model", filename="model.safetensors", revision=None,
        )

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=True)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_cached_snapshot_skips_progress_wrapper(
        self, mock_guard, mock_progress, _mock_cached,
    ):
        """A snapshot FULLY on disk is loaded, not transferred — so the
        byte-progress poller is never started (the top-bar bar must not
        flash) even though the guard call itself still happens (stall
        protection applies regardless of the cache-hit fast path)."""
        mock_guard.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        mock_progress.assert_not_called()

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_uncached_snapshot_emits_byte_progress(
        self, mock_guard, mock_progress, _mock_cached,
    ):
        """A snapshot missing from the cache is a real download → byte progress
        on, wrapping the guarded call. The poller reads on-disk growth, so it
        keeps working even though the bytes are written by a child process."""
        mock_guard.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        mock_progress.assert_called_once()
        assert mock_progress.call_args.kwargs.get("repo_id") == "org/model"

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.api.events.download_progress.snapshot_byte_progress")
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_partial_cache_emits_byte_progress(
        self, mock_guard, mock_progress, _mock_cached,
    ):
        """REGRESSION: a *partial* cache (an interrupted download) must still
        show progress. ``_snapshot_fully_cached`` reports False for a partial
        cache, so the byte poller wraps the guarded (self-healing, resumable)
        download and the remainder downloads WITH progress."""
        mock_guard.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/model")
        assert result == "/cache/repo"
        mock_progress.assert_called_once()
        mock_guard.assert_called_once_with(repo_id="org/model", revision=None)

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
