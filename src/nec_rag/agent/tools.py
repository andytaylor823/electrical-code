"""Tool definitions for the NEC expert agent.

Provides two LangChain tools:
- rag_search: embed a query and retrieve relevant NEC subsections from ChromaDB
- explain_image: send an image to a vision LLM for detailed description
"""

import base64
import logging
from pathlib import Path

import chromadb
from langchain_core.tools import tool

from nec_rag.agent.prompts import VISION_SYSTEM_PROMPT
from nec_rag.agent.resources import get_vision_client, get_vision_deployment, load_embedding_resources

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Private helpers for RAG retrieval
# ---------------------------------------------------------------------------


def _retrieve(query: str, embed_fn, collection: chromadb.Collection, n_results: int = 20) -> list[dict]:
    """Embed the query and retrieve the top-N most relevant subsections."""
    query_embedding = embed_fn(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB nested-list structure (single query -> index 0)
    retrieved = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})
    return retrieved


def _build_context(retrieved: list[dict]) -> str:
    """Format retrieved subsections into a markdown context string with source annotations."""
    sections = []
    for item in retrieved:
        meta = item["metadata"]
        header = f"[Section {meta['section_id']}, Article {meta['article_num']}, page {meta['page']}]"
        sections.append(f"{header}\n{item['document']}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tool: rag_search
# ---------------------------------------------------------------------------


@tool
def rag_search(query: str, num_results: int = 20) -> str:
    """Search the National Electrical Code (NEC) 2023 for sections relevant to a query.

    Embeds the query and retrieves the most relevant NEC subsections from the
    vector database. Returns formatted context with section IDs, article numbers,
    and page references.

    Args:
        query: The user's search query describing what NEC content to find.
        num_results: Number of subsections to retrieve (default 20).
    """
    embed_fn, collection = load_embedding_resources()
    logger.info("rag_search: query=%r  num_results=%d", query, num_results)
    retrieved = _retrieve(query, embed_fn, collection, n_results=num_results)
    context = _build_context(retrieved)
    logger.info("rag_search: retrieved %d subsections", len(retrieved))
    return context


# ---------------------------------------------------------------------------
# Tool: explain_image
# ---------------------------------------------------------------------------


@tool
def explain_image(file_path: str, user_question: str = "") -> str:
    """Analyze an image related to electrical wiring, installations, or the NEC.

    Reads an image from disk and sends it to a vision-capable LLM for a detailed
    description. The image is NOT loaded into the agent's context -- only the
    returned text summary is.

    Args:
        file_path: Absolute or relative path to the image file on disk.
        user_question: Optional context about what the user is asking, so the
            vision model can focus its analysis.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"Error: image file not found at {path}"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"Error: unsupported image format '{path.suffix}'. Supported: {IMAGE_EXTENSIONS}"

    # Base64-encode the image for the OpenAI vision API
    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    suffix = path.suffix.lower().lstrip(".")
    mime_type = "jpeg" if suffix == "jpg" else suffix
    data_uri = f"data:image/{mime_type};base64,{image_b64}"

    # Build the user message with optional question context
    user_text = "Describe this image in detail."
    if user_question:
        user_text = f'The user asked: "{user_question}"\n\nDescribe this image in detail, focusing on aspects relevant to the user\'s question.'

    logger.info("explain_image: sending %s (%.1f KB) to vision LLM", path.name, len(image_b64) / 1024)

    # Standalone vision LLM call (separate from the agent's own LLM context)
    vision_client = get_vision_client()
    response = vision_client.chat.completions.create(
        model=get_vision_deployment(),
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            },
        ],
    )
    description = response.choices[0].message.content
    logger.info("explain_image: received %d-char description", len(description))
    return description
