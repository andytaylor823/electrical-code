"""Utility helpers for the NEC agent (ID normalisation, retrieval, formatting, structure browsing)."""

import logging
import re
from difflib import get_close_matches

import chromadb

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


def _rerank(query: str, retrieved: list[dict], cross_encoder, top_n_rerank: int = 10, top_n_embed: int = 5) -> list[dict]:
    """Re-rank retrieved chunks and return the union of top embedding and top re-ranked results.

    Scores every chunk in *retrieved* with the cross-encoder, then merges:
    - the top ``top_n_rerank`` chunks by re-rank score (highest first)
    - the top ``top_n_embed`` chunks by embedding distance (already sorted)

    Duplicates are removed by index position; re-ranked items come first,
    followed by any embedding-only items that weren't already included.
    """
    if not retrieved:
        return []

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, item["document"]) for item in retrieved]
    scores = cross_encoder.predict(pairs)

    # Indices of top-N by re-rank score (descending)
    scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rerank_indices = scored_indices[:top_n_rerank]

    # Indices of top-N by embedding distance (already in order from _retrieve)
    embed_indices = list(range(min(top_n_embed, len(retrieved))))

    # Union: re-ranked first, then embedding-only extras
    seen: set[int] = set()
    merged_indices: list[int] = []
    for idx in rerank_indices:
        if idx not in seen:
            merged_indices.append(idx)
            seen.add(idx)
    for idx in embed_indices:
        if idx not in seen:
            merged_indices.append(idx)
            seen.add(idx)

    merged = [retrieved[i] for i in merged_indices]
    logger.info(
        "_rerank: %d candidates scored, returning %d (top-%d rerank + top-%d embed, %d overlap)",
        len(retrieved),
        len(merged),
        top_n_rerank,
        top_n_embed,
        len(set(rerank_indices) & set(embed_indices)),
    )
    return merged


def _build_context(retrieved: list[dict]) -> str:
    """Format retrieved chunks (subsections and tables) into a markdown context string.

    Collects ``referenced_tables`` IDs from subsection chunk metadata and lists
    them at the end so the agent knows which tables are available for follow-up
    via ``nec_lookup``.  Table chunks that were directly retrieved are excluded
    from the referenced-tables hint since their content is already inline.
    """
    sections = []
    all_ref_ids: set[str] = set()
    inline_table_ids: set[str] = set()

    for item in retrieved:
        meta = item["metadata"]
        chunk_type = meta.get("chunk_type", "subsection")

        # Use "Table" prefix for table chunks, "Section" for subsections
        if chunk_type == "table":
            header = f"[Table {meta['section_id']}, Article {meta['article_num']}, page {meta['page']}]"
            inline_table_ids.add(meta["section_id"])
        else:
            header = f"[Section {meta['section_id']}, Article {meta['article_num']}, page {meta['page']}]"

        sections.append(f"{header}\n{item['document']}")

        # Gather table IDs from the comma-separated metadata field
        refs_csv = meta.get("referenced_tables", "")
        if refs_csv:
            all_ref_ids.update(refs_csv.split(","))

    context_body = "\n\n".join(sections)

    # Exclude tables already shown inline from the "fetch via nec_lookup" hint
    remaining_refs = sorted(all_ref_ids - inline_table_ids)
    if remaining_refs:
        table_list = ", ".join(remaining_refs)
        context_body += f"\n\n[Tables referenced by these sections: {table_list}. " "Use nec_lookup(table_ids=[...]) to retrieve any you need.]"

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
