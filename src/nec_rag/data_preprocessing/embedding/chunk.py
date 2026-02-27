"""Chunk the structured NEC JSON into subsection-level and table-level pieces.

Reads data/prepared/NFPA 70 NEC 2023_structured.json and produces a list of
dicts ready for ChromaDB ingestion: each dict has 'id', 'text', 'document'
(optional), and 'metadata'.

Subsection chunks use 'text' for both embedding and document storage.
Table chunks use a compact 'text' (title + column headers) for embedding,
and a full markdown 'document' for context display.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
STRUCTURED_JSON_PATH = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"

# Subsections longer than this are split at lettered boundaries (A), (B), etc.
SPLIT_THRESHOLD = 3000

_LETTERED_RE = re.compile(r"^\([A-Z]\)\s")


def _build_subsection_text(subsection: dict) -> str:
    """Assemble the full text for a subsection from its front_matter and sub_items."""
    parts = [subsection["front_matter"]]
    for item in subsection.get("sub_items", []):
        parts.append(item["content"])
    return "\n".join(parts)


def _iter_subsections(data: dict):
    """Yield (subsection, parent_metadata) tuples by walking chapter > article > part > subsection."""
    for chapter in data["chapters"]:
        for article in chapter["articles"]:
            for part in article["parts"]:
                parent_meta = {
                    "part_num": part["part_num"] if part["part_num"] is not None else -1,  # ChromaDB requires non-None
                    "part_title": part["title"] or "",
                    "article_num": article["article_num"],
                    "article_title": article["title"],
                    "chapter_num": chapter["chapter_num"],
                    "chapter_title": chapter["title"],
                }
                for subsection in part["subsections"]:
                    yield subsection, parent_meta


def _deduplicated_id(article_num: int, section_id: str, seen_ids: set) -> str:
    """Generate a unique document ID, appending a suffix if the base ID already exists."""
    doc_id = f"{article_num}_{section_id}"
    if doc_id in seen_ids:
        counter = 2
        while f"{doc_id}_{counter}" in seen_ids:
            counter += 1
        doc_id = f"{doc_id}_{counter}"
    seen_ids.add(doc_id)
    return doc_id


def _group_by_letter(sub_items: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group sub_items by lettered subsection boundaries like (A), (B), etc.

    Returns a list of (letter, items) tuples.  Items before the first lettered
    entry are grouped under letter "_pre".  Numbered items (1), (2) stay with
    their preceding lettered parent.

    Returns an empty list if no lettered sub_items are found.
    """
    groups: list[tuple[str, list[dict]]] = []
    current_letter = "_pre"
    current_items: list[dict] = []
    found_any = False

    for item in sub_items:
        match = _LETTERED_RE.match(item["content"].strip())
        if match:
            found_any = True
            if current_items:
                groups.append((current_letter, current_items))
            current_letter = item["content"].strip()[1]  # Extract letter from "(A) ..."
            current_items = [item]
        else:
            current_items.append(item)

    if current_items:
        groups.append((current_letter, current_items))

    if not found_any:
        return []

    # Merge any pre-lettered items into the first lettered group
    if groups and groups[0][0] == "_pre" and len(groups) > 1:
        pre_items = groups[0][1]
        first_letter, first_items = groups[1]
        groups = [(first_letter, pre_items + first_items)] + groups[2:]

    return groups


def chunk_subsections(data: dict) -> list[dict]:
    """Walk the chapter > article > part > subsection hierarchy and emit chunks.

    Small subsections emit a single chunk.  Large subsections (exceeding
    ``SPLIT_THRESHOLD`` chars) are split at lettered sub-item boundaries
    with the parent front_matter prepended to each child chunk for context.
    """
    chunks = []
    seen_ids: set[str] = set()
    split_count = 0

    for subsection, parent_meta in _iter_subsections(data):
        section_id = subsection["id"]
        full_text = _build_subsection_text(subsection)
        referenced_tables = ",".join(subsection.get("referenced_tables", []))

        base_metadata = {
            "section_id": section_id,
            "title": subsection["title"][:500],
            "page": subsection["page"],
            "referenced_tables": referenced_tables,
            "chunk_type": "subsection",
            **parent_meta,
        }

        # Check if this subsection should be split
        sub_items = subsection.get("sub_items", [])
        groups = _group_by_letter(sub_items) if len(full_text) > SPLIT_THRESHOLD else []

        if not groups:
            # Emit as a single chunk (small section, or no lettered sub_items)
            doc_id = _deduplicated_id(parent_meta["article_num"], section_id, seen_ids)
            chunks.append({"id": doc_id, "text": full_text, "metadata": base_metadata})
        else:
            # Split into one chunk per lettered group, prepending front_matter
            split_count += 1
            front_matter = subsection["front_matter"]
            for letter, items in groups:
                group_text = front_matter + "\n" + "\n".join(item["content"] for item in items)
                doc_id = _deduplicated_id(parent_meta["article_num"], f"{section_id}_{letter}", seen_ids)
                chunks.append({"id": doc_id, "text": group_text, "metadata": base_metadata})

    logger.info("Chunked subsections: %d chunks (%d sections split at lettered boundaries)", len(chunks), split_count)
    return chunks


# ---------------------------------------------------------------------------
# Table chunking
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


def _build_table_embedding_text(table: dict) -> str:
    """Build a compact text for embedding: title + column headers.

    Keeps the embedding semantically dense (what the table is about)
    without inflating the vector with row data.
    """
    title = table["title"]
    headers = table.get("column_headers", [])
    if headers:
        return f"{title}\nColumns: {' | '.join(headers)}"
    return title


def _build_table_page_index(data: dict) -> dict[str, int]:
    """Map table IDs to page numbers from the first subsection that references them."""
    index: dict[str, int] = {}
    for subsection, _ in _iter_subsections(data):
        for ref in subsection.get("referenced_tables", []):
            table_id = ref.strip().split("\n")[0].strip()
            if table_id and table_id not in index:
                index[table_id] = subsection["page"]
    return index


def chunk_tables(data: dict) -> list[dict]:
    """Walk chapter > article > tables and emit one chunk per table.

    Each table chunk has a compact embedding text (title + column headers)
    and the full markdown rendering as the document for RAG context.
    """
    page_index = _build_table_page_index(data)
    chunks = []

    for chapter in data["chapters"]:
        for article in chapter["articles"]:
            for table in article["tables"]:
                table_id = table["id"]
                doc_id = f"table_{table_id}"

                embedding_text = _build_table_embedding_text(table)
                document = _format_table_as_markdown(table)

                chunks.append(
                    {
                        "id": doc_id,
                        "text": embedding_text,
                        "document": document,
                        "metadata": {
                            "section_id": table_id,
                            "title": table["title"][:500],
                            "page": page_index.get(table_id, -1),
                            "referenced_tables": table_id,
                            "chunk_type": "table",
                            "part_num": -1,
                            "part_title": "",
                            "article_num": article["article_num"],
                            "article_title": article["title"],
                            "chapter_num": chapter["chapter_num"],
                            "chapter_title": chapter["title"],
                        },
                    }
                )

    logger.info("Chunked %d tables from structured JSON", len(chunks))
    return chunks


def load_and_chunk() -> list[dict]:
    """Load the structured JSON from disk and return subsection + table chunks."""
    logger.info("Loading structured JSON from %s", STRUCTURED_JSON_PATH)
    with open(STRUCTURED_JSON_PATH, "r", encoding="utf-8") as fopen:
        data = json.load(fopen)
    return chunk_subsections(data) + chunk_tables(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = load_and_chunk()
    logger.info("Sample chunk (first): id=%s, text length=%d chars", result[0]["id"], len(result[0]["text"]))
    logger.info("Sample chunk (last): id=%s, text length=%d chars", result[-1]["id"], len(result[-1]["text"]))

    # Print quick stats
    text_lengths = [len(c["text"]) for c in result]
    logger.info(
        "Stats: total=%d, avg_chars=%d, min_chars=%d, max_chars=%d",
        len(result),
        sum(text_lengths) // len(text_lengths),
        min(text_lengths),
        max(text_lengths),
    )
