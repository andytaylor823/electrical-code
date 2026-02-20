"""Chunk the structured NEC JSON into subsection-level pieces with full parent metadata.

Reads data/prepared/NFPA 70 NEC 2023_structured.json and produces a list of
dicts ready for ChromaDB ingestion: each dict has 'id', 'text', and 'metadata'.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
STRUCTURED_JSON_PATH = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"


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


def chunk_subsections(data: dict) -> list[dict]:
    """Walk the chapter > article > part > subsection hierarchy and emit one chunk per subsection.

    Each chunk carries the subsection text plus metadata identifying every
    ancestor level so retrieval results can be placed in full NEC context.
    """
    chunks = []
    seen_ids: set[str] = set()

    for subsection, parent_meta in _iter_subsections(data):
        section_id = subsection["id"]
        text = _build_subsection_text(subsection)
        doc_id = _deduplicated_id(parent_meta["article_num"], section_id, seen_ids)

        # Comma-separated string because ChromaDB metadata values must be scalars
        referenced_tables = ",".join(subsection.get("referenced_tables", []))

        chunks.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": {
                    "section_id": section_id,
                    "title": subsection["title"][:500],  # ChromaDB metadata values must be <32KB
                    "page": subsection["page"],
                    "referenced_tables": referenced_tables,
                    **parent_meta,
                },
            }
        )

    logger.info("Chunked %d subsections from structured JSON", len(chunks))
    return chunks


def load_and_chunk() -> list[dict]:
    """Load the structured JSON from disk and return subsection chunks."""
    logger.info("Loading structured JSON from %s", STRUCTURED_JSON_PATH)
    with open(STRUCTURED_JSON_PATH, "r", encoding="utf-8") as fopen:
        data = json.load(fopen)
    return chunk_subsections(data)


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
