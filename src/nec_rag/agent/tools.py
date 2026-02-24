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


_RAG_SEARCH_NUM_RESULTS = 20  # number of subsections to retrieve per search


@tool(parse_docstring=True)
def rag_search(user_request: str) -> str:
    """Search the NEC 2023 vector database for sections relevant to a natural-language request.

    Embeds the request and retrieves the most relevant NEC subsections from the
    vector database via cosine-similarity. Returns formatted context with section
    IDs, article numbers, and page references. Each call returns up to 20
    subsections, which is usually more than enough to answer a question.

    USE WHEN: You do not yet know which NEC articles, sections, or tables are
    relevant to the user's question. This is a discovery tool for open-ended
    questions.

    DO NOT USE WHEN:
    - You already know specific section or table IDs (use nec_lookup instead).
    - You want to follow up on results from a prior rag_search (use nec_lookup
      for the exact text of sections you already discovered).
    - You have already called rag_search twice for this user question.

    LIMIT: Maximum 2 calls per user question. One well-crafted query usually
    suffices.

    Args:
        user_request: A single, focused natural-language question describing what
            NEC content to find. This is a semantic vector search -- write it as a
            natural sentence or question for best results.
            MUST NOT contain section numbers (e.g. "250.50"), article numbers
            (e.g. "Article 705"), or table IDs (e.g. "Table 220.55"). If you
            already know the ID, use nec_lookup instead.
            MUST NOT contain quotation marks, Boolean operators, or keyword
            fragments -- these degrade retrieval quality.
            MUST NOT combine multiple unrelated topics in a single query.
            GOOD example -- "grounding requirements for residential service entrances"
            BAD example -- "NEC 2023 Article 250 grounding; Article 705 705.12 interconnection"
            BAD example -- "'110.25' lockable page number marking"

    Returns:
        str: Formatted context blocks, each containing the subsection ID, article number, page reference, and full text of the retrieved subsections.
    """
    embed_fn, collection = load_embedding_resources()
    logger.info("rag_search: user_request=%s  num_results=%d", user_request, _RAG_SEARCH_NUM_RESULTS)
    retrieved = _retrieve(user_request, embed_fn, collection, n_results=_RAG_SEARCH_NUM_RESULTS)
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
    text description. The image itself is NOT loaded into the agent's context --
    only the returned text summary is. Supported formats: PNG, JPG, JPEG, GIF, WEBP.

    USE WHEN: The user attaches or references an image file in their request
    (e.g. a photo of a panel, wiring diagram, or NEC figure).

    DO NOT USE WHEN: The user has not attached or referenced any image file.
    This tool requires an image on disk and will error without one.

    Args:
        file_path: Absolute or relative path to the image file on disk.
        user_question: Optional context about what the user is asking, so the
            vision model can focus its analysis.

    Returns:
        str: A detailed text description of the image produced by the vision LLM.
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


_NEC_LOOKUP_MAX_IDS = 10  # combined cap on section_ids + table_ids per call


@tool(parse_docstring=True)
def nec_lookup(section_ids: list[str] | None = None, table_ids: list[str] | None = None) -> str:
    """Fetch exact NEC subsection text or table content by ID.

    Retrieves the full verbatim text of one or more NEC subsections and/or
    tables by their identifiers. Multiple IDs may be provided in a single call
    (up to 10 total across both lists). This is the most precise and
    context-cheap retrieval tool -- it returns only the requested items with no
    extra results. At least one ID must be provided across section_ids and
    table_ids.

    USE WHEN: You already know the specific section or table ID(s) you need --
    for example, from a prior rag_search hit, a browse_nec_structure outline, or
    because the user cited particular references (e.g. "What does 250.50 say?").
    Prefer this over rag_search whenever the targets are already known. You may
    batch up to 10 IDs in a single call to reduce round-trips.

    DO NOT USE WHEN: You do not yet know which section is relevant. Because this
    tool requires exact IDs, it should NOT be the first tool called unless the
    user explicitly asks about specific sections or tables. Use rag_search or
    browse_nec_structure first to identify candidate sections, then follow up
    with nec_lookup for the precise text.

    Args:
        section_ids: List of NEC section numbers, e.g. ["250.50", "90.1"]. Max 10 total IDs across both lists.
        table_ids: List of table identifiers, e.g. ["Table 220.55", "Table310.16"]. Max 10 total IDs across both lists.

    Returns:
        str: The full text of each requested subsection and/or markdown-formatted table,
            separated by blank lines. If an ID is not found, that entry contains an error
            message with suggested similar IDs.
    """
    section_ids = [s.strip() for s in (section_ids or []) if s.strip()]
    table_ids = [t.strip() for t in (table_ids or []) if t.strip()]

    if not section_ids and not table_ids:
        return "Error: at least one of section_ids or table_ids must be provided. " 'Example: nec_lookup(section_ids=["250.50"]) or nec_lookup(table_ids=["Table 220.55"])'

    # Enforce combined limit
    total_requested = len(section_ids) + len(table_ids)
    if total_requested > _NEC_LOOKUP_MAX_IDS:
        return (
            f"Error: requested {total_requested} IDs ({len(section_ids)} sections + {len(table_ids)} tables) "
            f"but the maximum is {_NEC_LOOKUP_MAX_IDS} total per call. Split the request into multiple calls."
        )

    output_parts: list[str] = []

    # --- Section lookups ---
    if section_ids:
        section_index = load_section_index()
        for sid in section_ids:
            subsection = section_index.get(sid)
            if subsection is None:
                valid_ids = sorted(section_index.keys())
                suggestions = suggest_similar_ids(sid, valid_ids)
                hint = ", ".join(suggestions) if suggestions else "(no similar IDs found)"
                output_parts.append(f"Error: section '{sid}' not found. Similar section IDs: {hint}")
                logger.warning("nec_lookup: section '%s' not found", sid)
            else:
                header = f"[Section {sid}, Article {subsection['article_num']}, page {subsection['page']}]"
                text = _build_subsection_text(subsection)
                output_parts.append(f"{header}\n{text}")
                logger.info("nec_lookup: resolved section '%s' (%d chars)", sid, len(text))

    # --- Table lookups ---
    if table_ids:
        table_index = load_table_index()
        for tid in table_ids:
            normalised = normalize_table_id(tid)
            table = table_index.get(normalised)
            if table is None:
                valid_ids = sorted(table_index.keys())
                suggestions = suggest_similar_ids(normalised, valid_ids)
                hint = ", ".join(suggestions) if suggestions else "(no similar IDs found)"
                output_parts.append(f"Error: table '{tid}' (normalised: '{normalised}') not found. Similar table IDs: {hint}")
                logger.warning("nec_lookup: table '%s' (normalised '%s') not found", tid, normalised)
            else:
                output_parts.append(_format_table_as_markdown(table))
                logger.info("nec_lookup: resolved table '%s'", normalised)

    return "\n\n".join(output_parts)


# ---------------------------------------------------------------------------
# Tool: browse_nec_structure
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def browse_nec_structure(chapter: int | None = None, article: int | None = None, part: int | None = None) -> str:
    """Browse the hierarchical structure of the NEC to discover what sections exist.

    Returns a plain-text outline of chapters, articles, parts, or subsections
    depending on the specificity of the arguments. This is a lightweight,
    context-inexpensive way to explore the NEC hierarchy and narrow down where
    relevant content lives before committing to a full-text retrieval. When an
    article is specified, the full text of the Scope subsection (XXX.1) is
    always included so you can gauge relevance.

    USE WHEN: You have a general idea of the relevant chapter or article but not
    the exact subsection. This is an excellent mid-level navigation tool --
    coarser than nec_lookup but more targeted than rag_search. There is rarely
    a bad time to use this tool; it provides a cheap overview of what content
    exists in any part of the NEC.

    DO NOT USE WHEN: There is no specific scenario where this tool should be
    avoided. It is safe to call at any point in the search process.

    Args:
        chapter: Chapter number (1-8). Returns articles in that chapter.
        article: Article number (e.g. 705). Returns parts and subsections in that article.
        part: Part number within an article (e.g. 1). Requires article to also be set.

    Returns:
        str: A plain-text outline of chapters, articles, parts, or subsections depending on the specificity of the arguments.
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
