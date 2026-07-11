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

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_resolve_hf_snapshot_uri(self, mock_guard, _mock_cached):
        mock_guard.return_value = "/cache/models/repo-id"
        result = ModelPathResolver.resolve("huggingface:org/repo-id")
        assert result == "/cache/models/repo-id"
        mock_guard.assert_called_once_with(repo_id="org/repo-id", revision=None)

    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_resolve_hf_single_file_uri(self, mock_guard):
        mock_guard.return_value = "/cache/models/file.safetensors"
        result = ModelPathResolver.resolve("huggingface:org/repo:file.safetensors")
        assert result == "/cache/models/file.safetensors"
        mock_guard.assert_called_once_with(
            repo_id="org/repo", filename="file.safetensors", revision=None,
        )

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.engine.utils.model_utils.snapshot_download")
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_resolve_hf_online_routes_through_killable_guard(
        self, mock_guard, mock_inprocess_snapshot, _mock_cached,
    ):
        """Online resolve must go through the killable guard (a child
        process), never call snapshot_download in-process directly.

        Regression context: a previously interrupted download leaves a
        *partial* snapshot (missing e.g. ``tokenizer/``); resolving from an
        in-process, un-abortable call risks the exact wedge this guard exists
        to fix — an in-process HF call cannot be aborted on a stall (Python
        threads/socket reads are un-abortable), and the trainer subprocess
        that hit this in production has no way to recover from one. The
        guard's own resumable retry (see test_hf_download_guard.py) is what
        re-verifies etags and self-heals a partial cache now — not this
        resolver directly.
        """
        mock_guard.return_value = "/cache/online"
        result = ModelPathResolver.resolve("huggingface:org/repo")
        assert result == "/cache/online"
        mock_guard.assert_called_once_with(repo_id="org/repo", revision=None)
        mock_inprocess_snapshot.assert_not_called()

    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_offline_cache_hit(self, mock_download, mock_guard):
        """Offline / skip-update mode reads cache only — single in-process
        call, and never touches the guard (nothing to stall on offline)."""
        mock_download.return_value = "/cache/local"
        result = ModelPathResolver.resolve(
            "huggingface:org/repo", local_files_only=True,
        )
        mock_download.assert_called_once_with(
            repo_id="org/repo", local_files_only=True,
        )
        assert result == "/cache/local"
        mock_guard.assert_not_called()

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_resolve_hf_offline_cache_miss_raises(self, mock_download):
        """Offline mode with nothing cached raises a clear error."""
        import pytest

        mock_download.side_effect = Exception("Not cached")
        with pytest.raises(FileNotFoundError):
            ModelPathResolver.resolve(
                "huggingface:org/repo", local_files_only=True,
            )


class TestRevision:
    """``huggingface:<repo>[@<revision>][:<filename>]`` — revision plumbing.

    Added for the dreamlite family: the DreamLite checkpoints carry the
    diffusers layout on the ``diffusers`` branch (the pipeline docs'
    canonical revision), so the resolver must be able to pin a revision.
    """

    @patch("app.engine.utils.model_utils._snapshot_fully_cached", return_value=False)
    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_snapshot_uri_with_revision(self, mock_guard, _mock_cached):
        mock_guard.return_value = "/cache/models/repo-rev"
        result = ModelPathResolver.resolve(
            "huggingface:carlofkl/DreamLite-base@diffusers",
        )
        assert result == "/cache/models/repo-rev"
        # The @revision must be split OFF the repo_id and passed as revision=
        mock_guard.assert_called_once_with(
            repo_id="carlofkl/DreamLite-base", revision="diffusers",
        )

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_offline_snapshot_uri_with_revision(self, mock_download):
        mock_download.return_value = "/cache/local-rev"
        result = ModelPathResolver.resolve(
            "huggingface:org/repo@mybranch", local_files_only=True,
        )
        mock_download.assert_called_once_with(
            repo_id="org/repo", local_files_only=True, revision="mybranch",
        )
        assert result == "/cache/local-rev"

    @patch("app.engine.utils.model_utils.download_with_stall_guard")
    def test_single_file_uri_with_revision(self, mock_guard):
        mock_guard.return_value = "/cache/file.safetensors"
        result = ModelPathResolver.resolve(
            "huggingface:org/repo@rev1:file.safetensors",
        )
        assert result == "/cache/file.safetensors"
        mock_guard.assert_called_once_with(
            repo_id="org/repo", filename="file.safetensors", revision="rev1",
        )

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_no_revision_keeps_legacy_call_shape(self, mock_download):
        """Without @revision the kwargs stay exactly as before (no
        ``revision=None`` noise — keeps older call-shape assertions valid)."""
        mock_download.return_value = "/cache/local"
        ModelPathResolver.resolve(
            "huggingface:org/repo", local_files_only=True,
        )
        mock_download.assert_called_once_with(
            repo_id="org/repo", local_files_only=True,
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
