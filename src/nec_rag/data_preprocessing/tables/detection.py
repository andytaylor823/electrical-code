"""Table boundary detection, content extraction, and paragraph interruption detection.

Operates on the OCR paragraph stream to find where tables start, where they
end (including multi-page continuations), extract the raw cell content, and
detect cases where a table splits a surrounding paragraph mid-sentence.
"""

import logging

from nec_rag.data_preprocessing.tables.classifiers import (
    get_table_id,
    is_continuation_marker,
    is_footnote,
    is_page_marker,
    is_section_boundary,
    is_table_title,
)
from nec_rag.data_preprocessing.tables.patterns import CONTINUATION_ENDINGS

logger = logging.getLogger(__name__)


# ─── Table Start Detection ───────────────────────────────────────────────────


def _is_real_table_start(paragraphs: dict[str, dict], idx: int) -> bool:
    """Heuristic: verify that a 'Table X.Y' paragraph is followed by table-like content.

    A genuine table title is followed by column headers and/or short data
    values.  A textual reference like "Table 220.55. Kilovolt-amperes ..."
    is followed by long body text.
    """
    n = len(paragraphs)
    # Collect the next 5 non-marker paragraphs
    following: list[str] = []
    j = idx + 1
    while j < n and len(following) < 5:
        content = paragraphs[str(j)]["content"]
        if not is_page_marker(content):
            following.append(content)
        j += 1

    if len(following) < 2:
        return False

    # If at least 2 of the next 5 are short (< 80 chars), it's likely a real table
    short_count = sum(1 for f in following if len(f) < 80)
    return short_count >= 2


def find_table_starts(paragraphs: dict[str, dict]) -> list[int]:
    """Return indices of all genuine table-title paragraphs (excludes continuations).

    See docs/table_cleaning.md § "Phase 1 — What the procedural code still handles"
    for why this detection layer was kept while the formatting was moved to the LLM.
    """
    starts: list[int] = []
    n = len(paragraphs)
    i = 0
    while i < n:
        content = paragraphs[str(i)]["content"]
        if is_table_title(content):
            # Skip continuation headers ("Table X.Y" immediately followed by "Continued")
            if i + 1 < n and paragraphs[str(i + 1)]["content"].strip() == "Continued":
                i += 2
                continue
            # Verify it's a real table, not a textual reference
            if _is_real_table_start(paragraphs, i):
                starts.append(i)
        i += 1
    return starts


# ─── Table End Detection ─────────────────────────────────────────────────────


def _is_table_continuation(paragraphs: dict[str, dict], idx: int, table_id: str) -> bool:
    """Return True if the paragraph at *idx* is a 'Table X.Y' + 'Continued' pair."""
    n = len(paragraphs)
    content = paragraphs[str(idx)]["content"]
    if not is_table_title(content):
        return False
    next_content = paragraphs[str(idx + 1)]["content"] if idx + 1 < n else ""
    return get_table_id(content) == table_id and next_content.strip() == "Continued"


def find_table_end(
    paragraphs: dict[str, dict],
    start_idx: int,
    next_table_start: int | None,
    max_scan: int = 300,
) -> int:
    """Return the last paragraph index that belongs to the table at *start_idx*.

    Scans forward, skipping page markers, until hitting a section boundary,
    a different table, or — after a natural page break — long non-footnote
    text that signals the end of the table region.

    Multi-page tables linked by ``(continues)`` / ``Continued`` markers are
    treated as a single region.
    """
    n = len(paragraphs)
    table_id = get_table_id(paragraphs[str(start_idx)]["content"])

    # Don't scan past the next known table start
    limit = min(n, start_idx + max_scan)
    if next_table_start is not None:
        limit = min(limit, next_table_start)

    last_content_idx = start_idx  # index of the last real table-content paragraph
    last_content_page = paragraphs[str(start_idx)]["page"]
    seen_data = False  # whether we've entered the short/numeric data region
    just_continued = False  # True right after processing a continuation marker

    i = start_idx + 1
    while i < limit:
        content = paragraphs[str(i)]["content"]

        # Skip page markers transparently
        if is_page_marker(content):
            i += 1
            continue

        # Handle multi-page continuation (same table ID + "Continued")
        if _is_table_continuation(paragraphs, i, table_id):
            i += 2  # skip "Table X.Y" and "Continued"
            just_continued = True
            continue
        if is_table_title(content):
            break  # different table — we've reached the end

        # "(continues)" marker — keep scanning
        if is_continuation_marker(content):
            last_content_idx = i
            i += 1
            just_continued = True
            continue

        # A clear section boundary ends the table
        if is_section_boundary(content):
            break

        # Detect a "natural" page break (page changed without a continuation marker)
        current_page = paragraphs[str(i)]["page"]
        natural_page_break = (current_page != last_content_page) and not just_continued
        just_continued = False  # reset after the first real paragraph on a new page

        # Short paragraph → treat as table data
        if len(content) < 60:
            seen_data = True
        elif seen_data and not is_footnote(content) and natural_page_break:
            # Long non-footnote text after a natural page break = table ended
            break

        # Accept content as table data (short, footnote, or same-page long cell)
        last_content_idx = i
        last_content_page = current_page
        i += 1

    return last_content_idx


# ─── Content Extraction ──────────────────────────────────────────────────────


def extract_table_content(paragraphs: dict[str, dict], start_idx: int, end_idx: int) -> list[str]:
    """Return content strings for the table, stripping page markers and continuation noise."""
    table_id = get_table_id(paragraphs[str(start_idx)]["content"])
    parts: list[str] = []
    skip_next_title = False  # flag: we saw continuation noise, skip repeated title

    for i in range(start_idx, end_idx + 1):
        content = paragraphs[str(i)]["content"]

        # Always skip page markers
        if is_page_marker(content):
            continue

        # Skip "(continues)" and the repeated "Table X.Y / Continued" pair
        if is_continuation_marker(content):
            skip_next_title = True
            continue
        if skip_next_title:
            if is_table_title(content) and get_table_id(content) == table_id:
                continue
            if content.strip() == "Continued":
                skip_next_title = False
                continue
            skip_next_title = False

        parts.append(content)

    return parts


# ─── Paragraph Interruption Detection ────────────────────────────────────────


def _find_pre_paragraph(paragraphs: dict[str, dict], table_start: int) -> int | None:
    """Walk backwards from *table_start* to find the last real content paragraph index."""
    pre_idx = table_start - 1
    while pre_idx >= 0 and is_page_marker(paragraphs[str(pre_idx)]["content"]):
        pre_idx -= 1
    return pre_idx if pre_idx >= 0 else None


def _find_post_paragraph(paragraphs: dict[str, dict], table_end: int) -> int | None:
    """Walk forward from *table_end* to find the first non-marker paragraph index."""
    n = len(paragraphs)
    post_idx = table_end + 1
    while post_idx < n and is_page_marker(paragraphs[str(post_idx)]["content"]):
        post_idx += 1
    return post_idx if post_idx < n else None


def detect_interruption(
    paragraphs: dict[str, dict],
    table_start: int,
    table_end: int,
) -> tuple[int | None, int | None]:
    """Detect if a table splits a paragraph.

    Returns (pre_idx, post_idx) where:
      - pre_idx  = index of the paragraph before the table that was cut mid-sentence
      - post_idx = index of the paragraph after the table that continues the sentence

    Returns (None, None) if no interruption is detected.
    """
    none_pair: tuple[None, None] = (None, None)

    # Find the content paragraphs immediately before and after the table region
    pre_idx = _find_pre_paragraph(paragraphs, table_start) if table_start > 0 else None
    if pre_idx is None:
        return none_pair
    pre_content = paragraphs[str(pre_idx)]["content"]

    # If the sentence is already complete, no interruption
    if pre_content and pre_content[-1] in ".?!:)":
        return none_pair

    post_idx = _find_post_paragraph(paragraphs, table_end)
    if post_idx is None:
        return none_pair
    post_content = paragraphs[str(post_idx)]["content"]

    # The continuation must not be a new section or table title
    if is_section_boundary(post_content) or is_table_title(post_content):
        return none_pair

    # Check for continuation signals: lowercase start or trailing preposition/conjunction
    starts_lowercase = post_content and post_content[0].islower()
    ends_with_conjunction = any(pre_content.endswith(ending) for ending in CONTINUATION_ENDINGS)
    if starts_lowercase or ends_with_conjunction:
        return pre_idx, post_idx

    return none_pair
