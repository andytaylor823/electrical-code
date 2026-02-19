"""Remove page headers, footers, page numbers, copyright lines, watermarks, and
section-number-only repeats that the OCR captured from each page boundary.

These paragraphs are structural artifacts of the PDF layout, not NEC content.
The sentence_runover step (which runs before this) relies on some of these
markers to detect page boundaries, so this step must run AFTER sentence_runover.

Patterns removed
----------------
- Edition markers:  "2023 Edition NATIONAL ELECTRICAL CODE",
                    "NATIONAL ELECTRICAL CODE 2023 Edition"
- Page numbers:     "70-23", "70-284", etc.
- Copyright lines:  "Copyright @NFPA..." / "Copyright @ NFPA..."
- Watermarks:       "EDUFIRE.IR", "Telegram: EDUFIRE.IR"
- Page headers:     All-caps "ARTICLE 110 - GENERAL REQUIREMENTS ..." at page tops
- Chapter headers:  All-caps "CHAPTER 1", "CHAPTER 2", etc. at page tops
- Chapter titles:   Mixed-case "Chapter 1 General" at chapter boundaries
                    (the article title paragraph already carries this info)
- Section-num only: Bare section number repeats at page tops, e.g. "110.26"
"""

import logging
import re

from nec_rag.cleaning.remove_junk_pages import resort_dict

logger = logging.getLogger(__name__)

# ── Compiled patterns ────────────────────────────────────────────────────────

# Edition footer/header markers (exact matches)
EDITION_MARKERS = frozenset(
    {
        "2023 Edition NATIONAL ELECTRICAL CODE",
        "NATIONAL ELECTRICAL CODE 2023 Edition",
    }
)

# Watermark strings (exact matches)
WATERMARK_STRINGS = frozenset(
    {
        "EDUFIRE.IR",
        "Telegram: EDUFIRE.IR",
    }
)

# Page number like "70-23" or "70-284"
PAGE_NUM_RE = re.compile(r"^70-\d+$")

# Copyright line prefix
COPYRIGHT_PREFIXES = ("Copyright @NFPA", "Copyright @ NFPA")

# All-caps article header at page top, e.g. "ARTICLE 100 - DEFINITIONS"
ARTICLE_HEADER_CAPS_RE = re.compile(r"^ARTICLE \d+\s*[-]?\s*[A-Z]")

# Standalone "CHAPTER 1", "CHAPTER 2", etc.
CHAPTER_CAPS_RE = re.compile(r"^CHAPTER \d+$")

# Mixed-case chapter title like "Chapter 2 Wiring and Protection"
CHAPTER_TITLE_RE = re.compile(r"^Chapter \d+ [A-Z]")

# Bare section number at page top, e.g. "110.26" (2+ digits before dot)
SECTION_NUM_ONLY_RE = re.compile(r"^\d{2,}\.\d+$")


def is_page_furniture(content: str) -> bool:
    """Return True if the paragraph is page furniture that should be removed.

    Checks exact matches first (cheapest), then prefix matches, then regex.
    """
    # Exact-match checks (fastest): edition markers and watermarks
    if content in EDITION_MARKERS or content in WATERMARK_STRINGS:
        return True

    # Prefix-based checks: copyright notice
    if any(content.startswith(prefix) for prefix in COPYRIGHT_PREFIXES):
        return True

    # Regex-based checks: page numbers, article headers, chapter markers,
    # chapter titles, and bare section-number repeats
    checks = [
        PAGE_NUM_RE.match(content),
        ARTICLE_HEADER_CAPS_RE.match(content) and content.isupper(),
        CHAPTER_CAPS_RE.match(content),
        CHAPTER_TITLE_RE.match(content),
        SECTION_NUM_ONLY_RE.match(content),
    ]
    return any(checks)


def run(paragraphs: dict[str, dict]) -> dict[str, dict]:
    """Remove all page-furniture paragraphs and re-index."""
    removed = 0
    output = {}

    for key, val in paragraphs.items():
        content = val["content"]
        if is_page_furniture(content):
            removed += 1
            continue
        # Keep this paragraph
        output[key] = val

    logger.info("Removed %d page-furniture paragraphs", removed)

    # Re-index with consecutive keys
    output = resort_dict(output)
    return output
