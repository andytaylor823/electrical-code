"""Unit tests for the embedding config module."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path

from nec_rag.data_preprocessing.embedding.config import COLLECTION_NAME, MODELS, ROOT, chroma_path


class TestChromaPath:

    def test_qwen3_path(self):
        result = chroma_path("qwen3")
        expected = ROOT / "data" / "vectors" / "qwen3-embedding-8b" / "chroma"
        assert result == expected

    def test_azure_large_path(self):
        result = chroma_path("azure-large")
        expected = ROOT / "data" / "vectors" / "text-embedding-3-large" / "chroma"
        assert result == expected

    def test_returns_path_object(self):
        result = chroma_path("qwen3")
        assert isinstance(result, Path)


class TestModelsConfig:

    def test_qwen3_config(self):
        cfg = MODELS["qwen3"]
        assert cfg["type"] == "local"
        assert "batch_size" in cfg
        assert "chroma_dir" in cfg

    def test_azure_large_config(self):
        cfg = MODELS["azure-large"]
        assert cfg["type"] == "azure"
        assert "batch_size" in cfg

    def test_collection_name(self):
        assert COLLECTION_NAME == "nec_subsections"

    def test_root_is_project_root(self):
        """ROOT should point to the project root (contains pyproject.toml)."""
        assert (ROOT / "pyproject.toml").exists()
