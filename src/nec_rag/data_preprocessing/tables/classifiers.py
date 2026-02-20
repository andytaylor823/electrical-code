"""Paragraph classification helpers for NEC table detection.

Each function takes a paragraph's content string and returns True/False
to classify it as a page marker, table title, section boundary, footnote,
continuation marker, or data-like cell value.
"""

import re

from nec_rag.data_preprocessing.tables.patterns import (
    FOOTNOTE_START_RE,
    PAGE_MARKER_PREFIXES,
    PAGE_NUM_RE,
    PART_HEADER_RE,
    PURE_NUMBER_RE,
    SECTION_NUM_ONLY_RE,
    SECTION_WITH_TEXT_RE,
    TABLE_ID_RE,
    TABLE_REFERENCE_WORDS,
    TABLE_TITLE_RE,
)


def is_page_marker(content: str) -> bool:
    """Return True if the paragraph is a page header, footer, page number, or article header."""
    # Check against known prefixes (footers, copyright, watermarks)
    if any(content.startswith(prefix) for prefix in PAGE_MARKER_PREFIXES):
        return True
    # Page number like "70-284"
    if PAGE_NUM_RE.match(content):
        return True
    # Article header at top of page (all-caps, starts with "ARTICLE")
    if content.startswith("ARTICLE ") and content.isupper():
        return True
    # Section number alone at page top (e.g. "400.48")
    if SECTION_NUM_ONLY_RE.match(content):
        return True
    return False


def is_table_title(content: str) -> bool:
    """Return True if the paragraph is a table title like 'Table 400.5(A)(1) ...'."""
    return bool(TABLE_TITLE_RE.match(content))


def get_table_id(content: str) -> str:
    """Extract a normalised table identifier, e.g. 'Table400.5(A)(1)'."""
    match = TABLE_ID_RE.match(content)
    raw_id = match.group(1) if match else content
    # Strip spaces so "Table 400.5(A) (1)" == "Table 400.5(A)(1)"
    return raw_id.replace(" ", "")


def is_section_boundary(content: str) -> bool:
    """Return True if the paragraph marks a new NEC section or Part header."""
    # "Part III. ..." or "400.10 Uses Permitted."
    if PART_HEADER_RE.match(content):
        return True
    if SECTION_WITH_TEXT_RE.match(content) and len(content) > 20:
        return True
    # Lettered or numbered subsection with substantial text
    is_sub = re.match(r"^\([A-Z0-9]+\) [A-Z]", content) and len(content) > 80
    return bool(is_sub)


def is_footnote(content: str) -> bool:
    """Return True if the paragraph looks like a table footnote."""
    # Starts with a footnote marker character (digit, superscript, ?, ', +, *)
    if FOOTNOTE_START_RE.match(content):
        return True
    # References table structural elements (Column A, subheading D, etc.)
    if any(ref in content for ref in TABLE_REFERENCE_WORDS) and len(content) < 400:
        return True
    return False


def is_continuation_marker(content: str) -> bool:
    """Return True for '(continues)' or 'Continued' markers."""
    stripped = content.strip().lower()
    return stripped in ("(continues)", "continued", "(continued)")


def is_data_like(text: str) -> bool:
    """Return True if the text looks like a table cell value (number, short code, etc.)."""
    stripped = text.strip()
    # Explicit short tokens
    if stripped in ("-", "--", "N/A", ".V/A", "No", "Yes"):
        return True
    # Pure number
    if PURE_NUMBER_RE.match(stripped):
        return True
    # Very short text (typical cell value)
    if len(stripped) <= 20:
        return True
    return False
