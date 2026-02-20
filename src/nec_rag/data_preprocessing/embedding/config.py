"""Shared configuration for the NEC RAG embedding and retrieval pipeline."""

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()

COLLECTION_NAME = "nec_subsections"

MODELS = {
    "qwen3": {
        "display_name": "Qwen/Qwen3-Embedding-8B",
        "type": "local",
        "chroma_dir": "qwen3-embedding-8b",
        "batch_size": 4,
    },
    "azure-large": {
        "display_name": "text-embedding-3-large",
        "type": "azure",
        "chroma_dir": "text-embedding-3-large",
        "batch_size": 50,
    },
}


def chroma_path(model_key: str) -> Path:
    """Return the ChromaDB persistent directory for a given model."""
    return ROOT / "data" / "vectors" / MODELS[model_key]["chroma_dir"] / "chroma"
