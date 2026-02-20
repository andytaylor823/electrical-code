"""Embed NEC subsection chunks and store in ChromaDB.

Supports multiple embedding models via the --model flag. Each model gets its
own ChromaDB persistent directory under data/vectors/{model_dir}/chroma/.

Usage:
    python -m nec_rag.data_preprocessing.embedding.embed --model qwen3         # local Qwen3-Embedding-8B
    python -m nec_rag.data_preprocessing.embedding.embed --model azure-large    # Azure OpenAI text-embedding-3-large
    python -m nec_rag.data_preprocessing.embedding.embed --model all            # both sequentially
    python -m nec_rag.data_preprocessing.embedding.embed --model all --reset    # wipe and re-embed both
"""

import argparse
import logging
import os
import time

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

from nec_rag.data_preprocessing.embedding.chunk import load_and_chunk
from nec_rag.data_preprocessing.embedding.config import COLLECTION_NAME, MODELS, ROOT, chroma_path

logger = logging.getLogger(__name__)

load_dotenv(ROOT / ".env")


def get_chroma_collection(model_key: str, reset: bool = False) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection for the given model."""
    path = chroma_path(model_key)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s' for model '%s'", COLLECTION_NAME, model_key)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _embed_qwen3(texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed texts using the local Qwen3-Embedding-8B model via sentence-transformers."""
    import torch  # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

    logger.info("Loading Qwen3-Embedding-8B (this may take a minute on first run)...")
    t0 = time.time()
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={"torch_dtype": torch.float16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    logger.info("Model loaded in %.1f seconds", time.time() - t0)

    all_embeddings = []
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Embedding (qwen3)"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings


def _embed_azure_large(texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed texts using Azure OpenAI text-embedding-3-large API."""
    from openai import AzureOpenAI  # pylint: disable=import-outside-toplevel

    # Suppress per-request HTTP logs from httpx so they don't clobber the tqdm bar
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )

    all_embeddings = []
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Embedding (azure-large)"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        response = client.embeddings.create(input=batch_texts, model="text-embedding-3-large")
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_for_model(model_key: str, chunks: list[dict], reset: bool = False):
    """Run the full embed-and-store pipeline for a single model."""
    model_cfg = MODELS[model_key]
    logger.info("=== Embedding with '%s' (%s) ===", model_key, model_cfg["display_name"])

    # Initialise ChromaDB collection
    collection = get_chroma_collection(model_key, reset=reset)
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        logger.info("Collection already has %d items -- skipping (use --reset to re-embed)", existing_count)
        return

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    batch_size = model_cfg["batch_size"]

    logger.info("Embedding %d chunks in batches of %d...", len(texts), batch_size)
    t0 = time.time()

    # Dispatch to the appropriate embedding function
    if model_cfg["type"] == "local":
        all_embeddings = _embed_qwen3(texts, batch_size)
    elif model_cfg["type"] == "azure":
        all_embeddings = _embed_azure_large(texts, batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")

    elapsed_embed = time.time() - t0
    logger.info("Embedding complete in %.1f seconds (%.1f chunks/sec)", elapsed_embed, len(texts) / elapsed_embed)

    # Insert into ChromaDB in batches (ChromaDB add can handle large batches)
    logger.info("Storing %d vectors in ChromaDB...", len(all_embeddings))
    chroma_batch_size = 500
    for batch_start in range(0, len(all_embeddings), chroma_batch_size):
        batch_end = min(batch_start + chroma_batch_size, len(all_embeddings))
        collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=all_embeddings[batch_start:batch_end],
            documents=texts[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )

    store_path = chroma_path(model_key)
    logger.info("ChromaDB collection '%s' now has %d items at %s", COLLECTION_NAME, collection.count(), store_path)


def main():
    """Chunk the structured JSON, embed with selected model(s), and store in ChromaDB."""
    parser = argparse.ArgumentParser(description="Embed NEC subsections into ChromaDB")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all", help="Which embedding model to use (default: all)")
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection(s) before embedding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load chunks once (model-agnostic)
    chunks = load_and_chunk()
    logger.info("Loaded %d chunks to embed", len(chunks))

    # Determine which models to run
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in model_keys:
        embed_for_model(model_key, chunks, reset=args.reset)

    logger.info("Done.")


if __name__ == "__main__":
    main()
