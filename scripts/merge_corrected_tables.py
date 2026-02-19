"""Merge corrected table JSON files back into the cleaned output files.

Reads each corrected table_*.json from data/intermediate/tables/, renders it
in the same markdown format used by the clean pipeline, finds the matching
paragraph in NFPA 70 NEC 2023_clean.json, replaces the content, and writes
the updated clean.json and clean.txt.

Usage:
    source .venv/bin/activate
    python scripts/merge_corrected_tables.py
"""

import json
import logging
import re
from pathlib import Path

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
TABLES_DIR = ROOT / "data" / "intermediate" / "tables"
CLEAN_JSON = ROOT / "data" / "intermediate" / "NFPA 70 NEC 2023_clean.json"
CLEAN_TXT = ROOT / "data" / "intermediate" / "NFPA 70 NEC 2023_clean.txt"


# ─── Rendering ────────────────────────────────────────────────────────────────


def render_table_paragraph(table_data: dict) -> str:
    """Render a TableStructure dict into the markdown format used by clean.json.

    Format: **Title**\\n\\n| col1 | col2 |\\n| --- | --- |\\n| ... |\\n\\n> footnote
    """
    lines = [f"**{table_data['title']}**", ""]

    # Column headers and separator
    headers = table_data["column_headers"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for row in table_data["data_rows"]:
        lines.append("| " + " | ".join(row) + " |")

    # Footnotes as blockquotes
    if table_data.get("footnotes"):
        lines.append("")
        for footnote in table_data["footnotes"]:
            lines.append(f"> {footnote}")

    return "\n".join(lines)


# ─── Table ID extraction ─────────────────────────────────────────────────────

TABLE_ID_RE = re.compile(r"(Table\s*\d+\.\d+(?:\s*\([^)]*\))*)")


def extract_table_id(title: str) -> str:
    """Extract and normalize a table ID from a title string."""
    match = TABLE_ID_RE.search(title)
    if match:
        return match.group(1).replace(" ", "")
    return title.replace(" ", "")


# ─── Load corrected tables ───────────────────────────────────────────────────


def load_corrected_tables() -> dict:
    """Load all corrected table JSON files from the tables directory.

    Returns a dict mapping normalized table_id -> rendered paragraph content.
    Special handling for table_310_4_1_reconstructed.json (different naming).
    """
    corrections = {}

    # Load individual corrected table files (table_*.json, excluding cache/corrections/reconstructed)
    skip_files = {"table_llm_cache.json", "table_corrections.json", "table_310_4_1_reconstructed.json"}
    for json_file in sorted(TABLES_DIR.glob("table_*.json")):
        if json_file.name in skip_files:
            continue
        logger.info("Loading corrected table: %s", json_file.name)
        with open(json_file, "r", encoding="utf-8") as fopen:
            table_data = json.load(fopen)
        table_id = extract_table_id(table_data["title"])
        rendered = render_table_paragraph(table_data)
        corrections[table_id] = rendered
        logger.info("  -> %s (%d rows)", table_id, len(table_data["data_rows"]))

    # Load Table 310.4(1) from the reconstructed file
    reconstructed_file = TABLES_DIR / "table_310_4_1_reconstructed.json"
    if reconstructed_file.exists():
        logger.info("Loading reconstructed table: %s", reconstructed_file.name)
        with open(reconstructed_file, "r", encoding="utf-8") as fopen:
            table_data = json.load(fopen)
        table_id = extract_table_id(table_data["title"])
        rendered = render_table_paragraph(table_data)
        corrections[table_id] = rendered
        logger.info("  -> %s (%d rows)", table_id, len(table_data["data_rows"]))

    return corrections


# ─── Merge into clean.json ────────────────────────────────────────────────────


def merge_into_clean_json(corrections: dict) -> dict:
    """Load clean.json, replace matching table paragraphs, return updated dict."""
    logger.info("Loading %s", CLEAN_JSON)
    with open(CLEAN_JSON, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)
    logger.info("Loaded %d paragraphs", len(paragraphs))

    # Build a mapping of paragraph index -> table_id for table paragraphs
    matched = set()
    for i in range(len(paragraphs)):
        content = paragraphs[str(i)]["content"]
        if not content.startswith("**Table "):
            continue

        # Extract table ID from the bold title
        title_match = re.match(r"\*\*(.+?)\*\*", content)
        if not title_match:
            continue
        title = title_match.group(1)
        table_id = extract_table_id(title)

        # Check if we have a correction for this table
        if table_id in corrections:
            old_len = len(content)
            paragraphs[str(i)]["content"] = corrections[table_id]
            new_len = len(corrections[table_id])
            matched.add(table_id)
            logger.info("  Replaced paragraph %d: %s (was %d chars, now %d chars)", i, table_id, old_len, new_len)

    # Report any corrections that didn't find a match
    unmatched = set(corrections.keys()) - matched
    if unmatched:
        logger.warning("No matching paragraph found for: %s", unmatched)

    logger.info("Replaced %d / %d corrected tables", len(matched), len(corrections))
    return paragraphs


# ─── Generate clean.txt ──────────────────────────────────────────────────────


def generate_clean_txt(paragraphs: dict) -> str:
    """Regenerate the clean.txt from the updated paragraphs dict."""
    lines = []
    for i in range(len(paragraphs)):
        content = paragraphs[str(i)]["content"]
        lines.append(content)
    return "\n\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Load corrections, merge into clean files, save."""
    # Load all corrected tables
    corrections = load_corrected_tables()
    logger.info("Loaded %d corrected tables total", len(corrections))

    # Merge into clean.json
    updated_paragraphs = merge_into_clean_json(corrections)

    # Save updated clean.json
    logger.info("Saving updated %s", CLEAN_JSON)
    with open(CLEAN_JSON, "w", encoding="utf-8") as fopen:
        json.dump(updated_paragraphs, fopen, ensure_ascii=False)
    logger.info("Saved clean.json")

    # Regenerate and save clean.txt
    logger.info("Regenerating %s", CLEAN_TXT)
    clean_txt = generate_clean_txt(updated_paragraphs)
    with open(CLEAN_TXT, "w", encoding="utf-8") as fopen:
        fopen.write(clean_txt)
    logger.info("Saved clean.txt (%d chars)", len(clean_txt))

    logger.info("Done! All corrected tables merged.")


if __name__ == "__main__":
    main()
