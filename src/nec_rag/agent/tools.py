"""Tool definitions for the NEC expert agent.

Provides LangChain tools:
- rag_search: embed a query and retrieve relevant NEC subsections from ChromaDB
- explain_image: send an image to a vision LLM for detailed description
- nec_lookup: fetch exact subsection text or table content by ID
- browse_nec_structure: navigate the NEC hierarchy and list section outlines
"""

import base64
import logging
from pathlib import Path

from langchain_core.tools import tool

from nec_rag.agent.loaders import load_section_index, load_structured_json
from nec_rag.agent.prompts import VISION_SYSTEM_PROMPT
from nec_rag.agent.resources import get_vision_client, get_vision_deployment, load_embedding_resources, load_table_index
from nec_rag.agent.utils import (
    _INT_TO_ROMAN,
    _build_context,
    _build_subsection_text,
    _format_article_outline,
    _format_table_as_markdown,
    _retrieve,
    normalize_table_id,
    suggest_similar_ids,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# ---------------------------------------------------------------------------
# Vision token usage accumulator (not captured by LangChain's callback)
# ---------------------------------------------------------------------------
_vision_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def reset_vision_usage() -> None:
    """Zero out the vision token counters before an agent invocation."""
    _vision_usage["prompt_tokens"] = 0
    _vision_usage["completion_tokens"] = 0
    _vision_usage["total_tokens"] = 0


def get_vision_usage() -> dict[str, int]:
    """Return accumulated token usage from standalone vision LLM calls."""
    return dict(_vision_usage)


# ---------------------------------------------------------------------------
# Tool: rag_search
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
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
    logger.info("rag_search: query=%s  num_results=%d", query, num_results)
    retrieved = _retrieve(query, embed_fn, collection, n_results=num_results)
    context = _build_context(retrieved)
    logger.info("rag_search: retrieved %d subsections", len(retrieved))
    return context


# ---------------------------------------------------------------------------
# Tool: explain_image
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
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
    # Track vision token usage (not captured by LangChain's get_openai_callback)
    if response.usage:
        _vision_usage["prompt_tokens"] += response.usage.prompt_tokens
        _vision_usage["completion_tokens"] += response.usage.completion_tokens
        _vision_usage["total_tokens"] += response.usage.total_tokens
        logger.info(
            "explain_image: vision usage — prompt=%d, completion=%d, total=%d",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )

    description = response.choices[0].message.content
    logger.info("explain_image: received %d-char description", len(description))
    return description


# ---------------------------------------------------------------------------
# Tool: nec_lookup
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def nec_lookup(section_id: str = "", table_id: str = "") -> str:
    """Fetch exact NEC subsection text or table content by ID.

    Use this when you already know the specific section or table you need
    (e.g. from a prior rag_search result or a user-cited reference). This is
    cheaper and more precise than rag_search -- prefer it for known references.

    At least one of section_id or table_id must be provided. Both may be
    provided in a single call to retrieve a section and a table together.

    Args:
        section_id: NEC section number, e.g. "250.50" or "90.1".
        table_id: Table identifier, e.g. "Table 220.55" or "Table220.55".
    """
    section_id = section_id.strip()
    table_id = table_id.strip()

    if not section_id and not table_id:
        return "Error: at least one of section_id or table_id must be provided. " 'Example: nec_lookup(section_id="250.50") or nec_lookup(table_id="Table 220.55")'

    output_parts: list[str] = []

    # --- Section lookup ---
    if section_id:
        section_index = load_section_index()
        subsection = section_index.get(section_id)

        if subsection is None:
            valid_ids = sorted(section_index.keys())
            suggestions = suggest_similar_ids(section_id, valid_ids)
            hint = ", ".join(suggestions) if suggestions else "(no similar IDs found)"
            output_parts.append(f"Error: section '{section_id}' not found. Similar section IDs: {hint}")
            logger.warning("nec_lookup: section '%s' not found", section_id)
        else:
            header = f"[Section {section_id}, Article {subsection['article_num']}, page {subsection['page']}]"
            text = _build_subsection_text(subsection)
            output_parts.append(f"{header}\n{text}")
            logger.info("nec_lookup: resolved section '%s' (%d chars)", section_id, len(text))

    # --- Table lookup ---
    if table_id:
        normalised = normalize_table_id(table_id)
        table_index = load_table_index()
        table = table_index.get(normalised)

        if table is None:
            valid_ids = sorted(table_index.keys())
            suggestions = suggest_similar_ids(normalised, valid_ids)
            hint = ", ".join(suggestions) if suggestions else "(no similar IDs found)"
            output_parts.append(f"Error: table '{table_id}' (normalised: '{normalised}') not found. Similar table IDs: {hint}")
            logger.warning("nec_lookup: table '%s' (normalised '%s') not found", table_id, normalised)
        else:
            output_parts.append(_format_table_as_markdown(table))
            logger.info("nec_lookup: resolved table '%s'", normalised)

    return "\n\n".join(output_parts)


# ---------------------------------------------------------------------------
# Tool: browse_nec_structure
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def browse_nec_structure(chapter: int | None = None, article: int | None = None, part: int | None = None) -> str:
    """Browse the structure of the NEC to discover what sections exist.

    Returns a plain-text outline of chapters, articles, parts, or subsections
    depending on the specificity of the arguments. Use this as a coarse-grained
    discovery tool before doing a fine-grained rag_search. When an article is
    specified, the full text of the Scope subsection (XXX.1) is always included.

    Args:
        chapter: Chapter number (1-8). Returns articles in that chapter.
        article: Article number (e.g. 705). Returns parts and subsections in that article.
        part: Part number within an article (e.g. 1). Requires article to also be set.
    """
    data = load_structured_json()
    chapters = data["chapters"]  # pylint: disable=unsubscriptable-object

    # Build lookup dicts for fast navigation
    chapter_by_num = {ch["chapter_num"]: ch for ch in chapters}
    article_by_num: dict[int, dict] = {}
    for ch in chapters:
        for art in ch["articles"]:
            article_by_num[art["article_num"]] = art

    # --- Article + optional part: show subsection outline ---
    if article is not None:
        art = article_by_num.get(article)
        if art is None:
            return f"Error: Article {article} not found. Valid articles: {sorted(article_by_num.keys())}"

        # Validate part filter if provided (convert int -> Roman for comparison)
        if part is not None:
            roman_part = _INT_TO_ROMAN.get(part)
            valid_romans = [p["part_num"] for p in art["parts"] if p["part_num"] is not None]
            valid_ints = [k for k, v in _INT_TO_ROMAN.items() if v in valid_romans]
            if roman_part is None or roman_part not in valid_romans:
                return f"Error: Part {part} not found in Article {article}. Valid parts: {sorted(set(valid_ints))}"

        logger.info("browse_nec_structure: article=%d  part=%s", article, part)
        return _format_article_outline(art, part_filter=part)

    # --- Chapter only: list articles ---
    if chapter is not None:
        ch = chapter_by_num.get(chapter)
        if ch is None:
            return f"Error: Chapter {chapter} not found. Valid chapters: {sorted(chapter_by_num.keys())}"

        logger.info("browse_nec_structure: chapter=%d", chapter)
        lines = [f"Chapter {ch['chapter_num']}: {ch['title']}"]
        for art in ch["articles"]:
            lines.append(f"  Article {art['article_num']}: {art['title']}")
        return "\n".join(lines)

    # --- No args: list all chapters and their articles ---
    logger.info("browse_nec_structure: listing all chapters")
    lines = []
    for ch in chapters:
        lines.append(f"Chapter {ch['chapter_num']}: {ch['title']}")
        for art in ch["articles"]:
            lines.append(f"  Article {art['article_num']}: {art['title']}")
        lines.append("")
    return "\n".join(lines)
