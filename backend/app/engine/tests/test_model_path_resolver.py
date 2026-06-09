"""Tests for ModelPathResolver — local paths and HuggingFace URI handling."""

from unittest.mock import MagicMock, patch


from app.engine.utils.model_utils import ModelPathResolver


class TestResolve:
    """Test ModelPathResolver.resolve() static method."""

    def test_resolve_empty_string(self):
        assert ModelPathResolver.resolve("") is None

    def test_resolve_none_like_empty(self):
        assert ModelPathResolver.resolve(None) is None

    def test_resolve_absolute_path(self, tmp_path):
        path = str(tmp_path / "model")
        result = ModelPathResolver.resolve(path)
        assert result == path

    def test_resolve_relative_path_with_base(self, tmp_path):
        result = ModelPathResolver.resolve("subdir", base_dir=str(tmp_path))
        expected = str(tmp_path / "subdir")
        assert result == expected

    def test_resolve_relative_path_without_base(self):
        import os
        result = ModelPathResolver.resolve("relative/path")
        assert os.path.isabs(result)
        assert result.endswith(os.path.join("relative", "path"))

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_snapshot_uri(self, mock_download):
        mock_download.return_value = "/cache/models/repo-id"
        result = ModelPathResolver.resolve("huggingface:org/repo-id")
        assert result == "/cache/models/repo-id"

    @patch("app.engine.utils.model_utils.hf_hub_download")
    def test_resolve_hf_single_file_uri(self, mock_download):
        mock_download.return_value = "/cache/models/file.safetensors"
        result = ModelPathResolver.resolve("huggingface:org/repo:file.safetensors")
        assert result == "/cache/models/file.safetensors"

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_online_always_runs_resumable_download(self, mock_download):
        """Online mode must run the real resumable snapshot_download.

        Regression: the old fast-path called ``snapshot_download(..,
        local_files_only=True)`` first and, if it returned ANY cached
        snapshot, skipped the real download.  A previously interrupted
        download leaves a *partial* snapshot (missing e.g. ``tokenizer/``);
        the fast-path returned it as if complete, so the loader then failed
        with "Unrecognized model ... should have a ``model_type`` key".

        The fix: online mode always calls the resumable
        ``snapshot_download`` (no ``local_files_only=True`` pre-check), which
        re-verifies every file's etag and fetches only what is missing —
        self-healing a partial cache.
        """
        mock_download.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/repo")
        assert result == "/cache/repo"
        # No cache-only short-circuit when online.
        for call in mock_download.call_args_list:
            assert call.kwargs.get("local_files_only") is not True, (
                "online resolve must not short-circuit on a cache-only "
                "(possibly partial) snapshot"
            )
        # Exactly one (resumable) download call.
        assert mock_download.call_count == 1

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_offline_cache_hit(self, mock_download):
        """Offline / skip-update mode reads cache only — single call."""
        mock_download.return_value = "/cache/local"
        result = ModelPathResolver.resolve(
            "huggingface:org/repo", local_files_only=True,
        )
        mock_download.assert_called_once_with(
            repo_id="org/repo", local_files_only=True,
        )
        assert result == "/cache/local"

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_offline_cache_miss_raises(self, mock_download):
        """Offline mode with nothing cached raises a clear error."""
        import pytest

        mock_download.side_effect = Exception("Not cached")
        with pytest.raises(FileNotFoundError):
            ModelPathResolver.resolve(
                "huggingface:org/repo", local_files_only=True,
            )


class TestFindComponent:
    """Test ModelPathResolver.find_component() static method."""

    def test_explicit_definition_hit(self):
        """Component found via definition.components[key]."""
        definition = MagicMock()
        comp = MagicMock()
        comp.path = "/explicit/path"
        definition.components = {"vae": comp}

        result = ModelPathResolver.find_component(
            definition, "vae", root_path="/root",
        )
        assert result == "/explicit/path"

    def test_discovery_in_root(self, tmp_path):
        """Component found by scanning root_path for candidates."""
        # Create a candidate directory
        (tmp_path / "vae").mkdir()

        definition = MagicMock()
        definition.components = {}

        result = ModelPathResolver.find_component(
            definition, "vae", root_path=str(tmp_path),
            candidates=["vae"],
        )
        assert result == str(tmp_path / "vae")

    def test_not_found_returns_none(self):
        """No match → returns None."""
        definition = MagicMock()
        definition.components = {}

        result = ModelPathResolver.find_component(
            definition, "vae", root_path="/nonexistent",
            candidates=["vae"],
        )
        assert result is None
