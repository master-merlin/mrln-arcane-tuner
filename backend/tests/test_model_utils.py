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
    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_hf_full_repo(self, mock_download):
        """huggingface:<repo_id> should trigger snapshot_download."""
        mock_download.side_effect = [
            Exception("not cached"),  # local_files_only first
            "/cache/repo",  # actual download
        ]
        # local only fails, then full download succeeds
        mock_download.side_effect = Exception("not cached")

        with patch("app.engine.utils.model_utils.snapshot_download") as m:
            m.side_effect = [Exception("not cached"), "/cache/repo"]
            result = ModelPathResolver.resolve("huggingface:org/model")
            assert result == "/cache/repo"

    @patch("app.engine.utils.model_utils.hf_hub_download")
    def test_hf_single_file(self, mock_download):
        """huggingface:<repo_id>:<filename> should trigger hf_hub_download."""
        mock_download.side_effect = [
            Exception("not cached"),
            "/cache/model.safetensors",
        ]
        result = ModelPathResolver.resolve("huggingface:org/model:model.safetensors")
        assert result == "/cache/model.safetensors"

    @patch("app.engine.utils.model_utils.snapshot_download")
    def test_hf_uses_local_cache_first(self, mock_download):
        """Should try local_files_only=True first and skip download if cached."""
        mock_download.return_value = "/cache/repo"
        result = ModelPathResolver.resolve("huggingface:org/model")
        # First call should be local_files_only=True
        mock_download.assert_called_once_with(repo_id="org/model", local_files_only=True)
        assert result == "/cache/repo"


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
