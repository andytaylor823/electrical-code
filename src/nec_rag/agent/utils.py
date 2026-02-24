"""Utility helpers for the NEC agent (ID normalisation, retrieval, formatting, structure browsing)."""

import logging
import re
from difflib import get_close_matches

import chromadb

from nec_rag.agent.resources import load_table_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID normalisation and fuzzy matching
# ---------------------------------------------------------------------------

_TABLE_PREFIX_RE = re.compile(r"^table\s*", re.IGNORECASE)


def normalize_table_id(raw_id: str) -> str:
    """Normalise a table ID to the canonical format used in the structured JSON.

    Handles case variations and extra spaces so that inputs like
    ``"Table 220.55"``, ``"table220.55"``, and ``"TABLE  220.55"`` all
    become ``"Table220.55"``.
    """
    bare = _TABLE_PREFIX_RE.sub("", raw_id.strip()).replace(" ", "")
    return f"Table{bare}"


def suggest_similar_ids(query_id: str, valid_ids: list[str], n: int = 5) -> list[str]:
    """Return up to *n* IDs from *valid_ids* that are close to *query_id*."""
    return get_close_matches(query_id, valid_ids, n=n, cutoff=0.4)


# ---------------------------------------------------------------------------
# RAG retrieval helpers
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
    """Format retrieved subsections into a markdown context string with source annotations.

    Reads the pre-computed ``referenced_tables`` metadata on each chunk to
    collect every table mentioned across the retrieved subsections, then
    appends the full table content so the agent can reason over it.
    """
    sections = []
    all_ref_ids: set[str] = set()

    for item in retrieved:
        meta = item["metadata"]
        header = f"[Section {meta['section_id']}, Article {meta['article_num']}, page {meta['page']}]"
        sections.append(f"{header}\n{item['document']}")

        # Gather table IDs from the comma-separated metadata field
        refs_csv = meta.get("referenced_tables", "")
        if refs_csv:
            all_ref_ids.update(refs_csv.split(","))

    context_body = "\n\n".join(sections)

    # Look up and format every referenced table
    table_blocks = _resolve_table_refs(sorted(all_ref_ids))
    if table_blocks:
        context_body += "\n\n" + "=" * 60 + "\nREFERENCED TABLES\n" + "=" * 60 + "\n\n"
        context_body += "\n\n".join(table_blocks)

    return context_body


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _format_table_as_markdown(table: dict) -> str:
    """Render a structured table dict as a readable markdown table with footnotes."""
    lines = [f"**{table['title']}**"]

    headers = table["column_headers"]
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in table["data_rows"]:
            lines.append("| " + " | ".join(row) + " |")

    for footnote in table.get("footnotes", []):
        lines.append(f"> {footnote}")

    return "\n".join(lines)


def _resolve_table_refs(ref_ids: list[str]) -> list[str]:
    """Look up table IDs in the structured data and return formatted markdown blocks."""
    if not ref_ids:
        return []

    table_index = load_table_index()
    blocks = []
    for ref_id in ref_ids:
        table = table_index.get(ref_id)
        if table:
            blocks.append(_format_table_as_markdown(table))
        else:
            logger.debug("Table reference '%s' not found in index", ref_id)
    return blocks


# ---------------------------------------------------------------------------
# Subsection text helpers
# ---------------------------------------------------------------------------


def _build_subsection_text(subsection: dict) -> str:
    """Assemble a subsection's full text from front_matter and sub_items."""
    parts = [subsection["front_matter"]]
    for item in subsection.get("sub_items", []):
        parts.append(item["content"])
    return "\n".join(parts)


def _build_subsection_full_text(subsection: dict) -> str:
    """Assemble the full text for a subsection from its front_matter and sub_items."""
    parts = [subsection["front_matter"]]
    for item in subsection.get("sub_items", []):
        parts.append(item["content"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Structure browsing helpers
# ---------------------------------------------------------------------------

_INT_TO_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}


def _find_scope_subsection(article: dict) -> str | None:
    """Return the full text of the XXX.1 subsection (Scope) for an article, or None."""
    scope_id = f"{article['article_num']}.1"
    for part in article["parts"]:
        for subsection in part["subsections"]:
            if subsection["id"] == scope_id:
                return _build_subsection_full_text(subsection)
    return None


def _format_article_outline(article: dict, part_filter: int | None = None) -> str:
    """Build a plain-text outline for an article, optionally filtered to one part.

    ``part_filter`` is an integer (e.g. 1) which is converted to a Roman numeral
    for matching against the structured data's ``part_num`` field.
    """
    # Convert integer part filter to Roman numeral string for comparison
    roman_filter = _INT_TO_ROMAN.get(part_filter) if part_filter is not None else None

    lines = [f"Article {article['article_num']}: {article['title']}"]

    for part in article["parts"]:
        # If a specific part was requested, skip others
        if roman_filter is not None and part["part_num"] != roman_filter:
            continue

        # Part header (skip for articles with a single unnamed part)
        if part["part_num"] is not None:
            lines.append(f"  Part {part['part_num']}: {part['title']}")

        for subsection in part["subsections"]:
            indent = "    " if part["part_num"] is not None else ""
            title = subsection["title"]
            if len(title) > 120:
                title = title[:117] + "..."
            lines.append(f"{indent}  {subsection['id']} {title} (page {subsection['page']})")

    # Append the scope (XXX.1) full text as a summary
    scope_text = _find_scope_subsection(article)
    if scope_text:
        lines.append("")
        lines.append("--- Scope ---")
        lines.append(scope_text)

    return "\n".join(lines)
