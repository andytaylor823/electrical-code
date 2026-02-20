"""Centralized resource initialization for the NEC agent.

Handles loading and caching of:
- Embedding models (local sentence-transformers or Azure OpenAI)
- ChromaDB vector store collections
- LLM clients (agent chat model, standalone vision model)
- Table index (table_id -> table dict from structured JSON)
"""

import json
import logging
import os

import chromadb
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from nec_rag.data_preprocessing.embedding.config import COLLECTION_NAME, MODELS, ROOT, chroma_path

logger = logging.getLogger(__name__)

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Cached resources -- populated lazily on first access
# ---------------------------------------------------------------------------
_CACHE: dict = {
    "embed_fn": None,
    "collection": None,
    "agent_llm": None,
    "vision_client": None,
    "table_index": None,
}


# ---------------------------------------------------------------------------
# Embedding + ChromaDB
# ---------------------------------------------------------------------------


def load_embedding_resources(model_key: str = "azure-large"):
    """Load the embedding function and ChromaDB collection, caching for reuse.

    Returns (embed_fn, collection) where embed_fn(str) -> list[float].
    """
    if _CACHE["embed_fn"] is not None and _CACHE["collection"] is not None:
        return _CACHE["embed_fn"], _CACHE["collection"]

    model_cfg = MODELS[model_key]

    # Build embedding function based on model type
    if model_cfg["type"] == "local":
        import torch  # pylint: disable=import-outside-toplevel
        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

        logger.info("Loading local embedding model '%s'...", model_cfg["display_name"])
        st_model = SentenceTransformer(
            model_cfg["display_name"],
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )

        def _local_embed(text: str) -> list[float]:
            return st_model.encode(text, prompt_name="query").tolist()

        _CACHE["embed_fn"] = _local_embed

    elif model_cfg["type"] == "azure":
        embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        logger.info("Using Azure OpenAI embedding model '%s'", model_cfg["display_name"])

        def _azure_embed(text: str) -> list[float]:
            response = embedding_client.embeddings.create(input=text, model=model_cfg["display_name"])
            return response.data[0].embedding

        _CACHE["embed_fn"] = _azure_embed

    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")

    # Load ChromaDB collection
    store_path = chroma_path(model_key)
    client = chromadb.PersistentClient(path=str(store_path))
    _CACHE["collection"] = client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB collection '%s' loaded (%d items) from %s", COLLECTION_NAME, _CACHE["collection"].count(), store_path)

    return _CACHE["embed_fn"], _CACHE["collection"]


# ---------------------------------------------------------------------------
# Table index (table_id -> table dict from structured JSON)
# ---------------------------------------------------------------------------

STRUCTURED_JSON_PATH = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"


def load_table_index() -> dict[str, dict]:
    """Build a lookup from normalised table ID to its structured dict, caching for reuse.

    Walks every chapter > article in the structured JSON and indexes each table
    by its 'id' field (e.g. 'Table220.55').
    """
    if _CACHE["table_index"] is not None:
        return _CACHE["table_index"]

    logger.info("Building table index from %s", STRUCTURED_JSON_PATH)
    with open(STRUCTURED_JSON_PATH, "r", encoding="utf-8") as fopen:
        data = json.load(fopen)

    # Walk the hierarchy and collect every table, keyed by normalised ID
    index: dict[str, dict] = {}
    for chapter in data["chapters"]:
        for article in chapter["articles"]:
            for table in article["tables"]:
                index[table["id"]] = table

    _CACHE["table_index"] = index
    logger.info("Table index built: %d tables", len(index))
    return index


# ---------------------------------------------------------------------------
# Agent LLM (LangChain wrapper for the main reasoning model)
# ---------------------------------------------------------------------------


def get_agent_llm() -> AzureChatOpenAI:
    """Return the cached AzureChatOpenAI instance used as the agent's reasoning model."""
    if _CACHE["agent_llm"] is None:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")
        _CACHE["agent_llm"] = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        logger.info("Agent LLM initialised: %s", deployment)
    return _CACHE["agent_llm"]


# ---------------------------------------------------------------------------
# Vision LLM (raw OpenAI client for image analysis, outside agent context)
# ---------------------------------------------------------------------------


def get_vision_client() -> AzureOpenAI:
    """Return the cached AzureOpenAI client used for standalone vision calls."""
    if _CACHE["vision_client"] is None:
        _CACHE["vision_client"] = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        logger.info("Vision client initialised")
    return _CACHE["vision_client"]


def get_vision_deployment() -> str:
    """Return the Azure deployment name to use for vision requests."""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")
