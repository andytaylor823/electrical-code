"""
Cleaning pipeline for OCR paragraph data.

Reads the raw paragraph JSON (data/raw/) produced by the OCR process, applies
five cleaning steps in order:
  1. remove_junk_pages     -- keep only pages with actual NEC content (pages 26-717)
  2. tables                -- detect tables, format as markdown, repair interrupted paragraphs
  3. sentence_runover      -- merge sentences split across page boundaries
  4. hyphens_endline       -- remove end-of-line hyphenation artifacts
  5. remove_page_furniture -- strip page headers/footers, copyright, watermarks, etc.

Then exports two final outputs into data/intermediate/:
  - NFPA 70 NEC 2023_clean.json  (cleaned paragraphs with page numbers)
  - NFPA 70 NEC 2023_clean.txt   (plain text concatenation of cleaned paragraphs)

Usage:
  python -m nec_rag.data_preprocessing.text_cleaning.clean
"""

import json
import logging
from pathlib import Path

from nec_rag.data_preprocessing.tables import pipeline as tables
from nec_rag.data_preprocessing.text_cleaning import hyphens_endline, remove_junk_pages, remove_page_furniture, sentence_runover

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve paths
ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
PARAGRAPHS_FILE = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
OUTPUT_DIR = ROOT / "data" / "intermediate"


def load_paragraphs(filepath: Path = PARAGRAPHS_FILE) -> dict[str, dict]:
    """Load raw paragraph JSON from disk."""
    logger.info("Loading paragraphs from %s", filepath)
    with open(filepath, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)
    logger.info("Loaded %d paragraphs", len(paragraphs))
    return paragraphs


def run_cleaning_pipeline(paragraphs: dict[str, dict]) -> dict[str, dict]:
    """Run all cleaning steps on the paragraph dict and return the cleaned result."""
    # Step 1: Remove junk pages (cover, TOC, appendices, index, etc.)
    output = remove_junk_pages.run(paragraphs)
    logger.info("After remove_junk_pages: %d paragraphs", len(output))

    # Step 2: Detect tables, format as markdown, repair interrupted paragraphs
    output = tables.run(output)
    logger.info("After tables: %d paragraphs", len(output))

    # Step 3: Merge sentences that were split across page boundaries
    output = sentence_runover.run(output)
    logger.info("After sentence_runover: %d paragraphs", len(output))

    # Step 4: Remove end-of-line hyphenation artifacts
    output = hyphens_endline.run(output)
    logger.info("After hyphens_endline: %d paragraphs", len(output))

    # Step 5: Remove page headers, footers, copyright, watermarks, etc.
    output = remove_page_furniture.run(output)
    logger.info("After remove_page_furniture: %d paragraphs", len(output))

    return output


def paragraphs_to_text(paragraphs: dict[str, dict]) -> str:
    """Convert paragraph dict to plain text, stripping non-ASCII characters."""
    lines = []
    for paragraph in paragraphs.values():
        content = paragraph["content"]
        # Strip non-ASCII via charmap encoding (same approach as OCR script)
        content = content.encode("charmap", errors="ignore").decode("charmap")
        lines.append(content)
    return "\n".join(lines)


def save_outputs(paragraphs: dict[str, dict], output_dir: Path) -> None:
    """Write cleaned paragraphs as JSON and plain text."""
    # Save cleaned JSON (with page numbers)
    json_file = output_dir / "NFPA 70 NEC 2023_clean.json"
    with open(json_file, "w", encoding="utf-8") as fopen:
        json.dump(paragraphs, fopen)
    logger.info("Wrote cleaned JSON to %s", json_file)

    # Save cleaned plain text
    txt_file = output_dir / "NFPA 70 NEC 2023_clean.txt"
    text = paragraphs_to_text(paragraphs)
    with open(txt_file, "w", encoding="utf-8") as fopen:
        fopen.write(text)
    logger.info("Wrote cleaned text to %s", txt_file)


if __name__ == "__main__":
    raw_paragraphs = load_paragraphs()
    cleaned_paragraphs = run_cleaning_pipeline(raw_paragraphs)
    save_outputs(cleaned_paragraphs, OUTPUT_DIR)
