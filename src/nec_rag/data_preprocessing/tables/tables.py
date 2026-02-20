"""Table detection, extraction, and LLM-based markdown formatting for OCR paragraph data.

The Azure Document Intelligence OCR extracted table content as a flat stream
of individual cell values -- one paragraph per cell, in reading order.  This
module:

  1. Detects table boundaries (start / end) in the paragraph stream.
  2. Handles multi-page tables with "(continues)" / "Continued" markers.
  3. Separates tables from paragraphs they interrupt (mid-sentence splits
     caused by page layout).
  4. Sends extracted cell values to an Azure OpenAI LLM with a Pydantic-
     enforced structured output schema to reconstruct column headers, data
     rows, and footnotes.
  5. Falls back to a plain text block if the LLM is unavailable or fails.

Pipeline position: runs AFTER remove_junk_pages but BEFORE sentence_runover,
because raw table content scattered across paragraphs confuses the sentence-
merge heuristic.

Limitations
-----------
- Two-column page layouts can cause the OCR to interleave table content from
  the right column with unrelated section text.  This module handles the main
  table body but may miss orphaned content from the opposite column.
- Very large tables (250+ fragments) may be truncated or summarised by the LLM;
  the text-block fallback preserves all original content in those cases.

For full documentation of the end-to-end table cleaning process — including
the automated triage, manual correction, and final merge phases that follow
this module — see docs/table_cleaning.md.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


# ─── Project Root & Environment ───────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env")


# ─── Regex Patterns ──────────────────────────────────────────────────────────

# Table title such as "Table 400.5(A)(1) Ampacity for ..."
TABLE_TITLE_RE = re.compile(r"^Table \d+\.\d+")

# Extract the table identifier (e.g. "Table 400.5(A)(1)") from a title string
TABLE_ID_RE = re.compile(r"(Table \d+\.\d+(?:\s*\([^)]*\))*)")

# Section number alone at page top, e.g. "400.48" or "90.1".
# NEC article numbers are always 2+ digits (70, 90, 100, ..., 840),
# so require at least 2 digits before the dot to avoid false-positives
# on table data like "2.79" or "3.05".
SECTION_NUM_ONLY_RE = re.compile(r"^\d{2,}\.\d+$")

# Section number followed by title text, e.g. "400.10 Uses Permitted."
SECTION_WITH_TEXT_RE = re.compile(r"^\d+\.\d+ [A-Z]")

# Page number footer like "70-284"
PAGE_NUM_RE = re.compile(r"^70-\d+$")

# Part header like "Part III."
PART_HEADER_RE = re.compile(r"^Part [IVX]+\.")

# Pure number (integer, decimal, or fraction like "1/0")
PURE_NUMBER_RE = re.compile(r"^-?\d+$|^-?\d+\.\d+$|^\d+/\d+$")

# Footnote start characters (digits, superscripts, special markers)
FOOTNOTE_START_RE = re.compile(r"^[\d\u00b9\u00b2\u00b3\u2070-\u2079?'+*]")

# Known page-marker prefixes (headers, footers, copyright, watermarks)
PAGE_MARKER_PREFIXES = (
    "2023 Edition NATIONAL ELECTRICAL CODE",
    "NATIONAL ELECTRICAL CODE 2023 Edition",
    "Copyright @NFPA",
    "Copyright @ NFPA",
    "EDUFIRE.IR",
    "Telegram: EDUFIRE.IR",
)

# Words found in table footnotes that reference table structure
TABLE_REFERENCE_WORDS = ("Column ", "column ", "subheading ", "ampacit")

# Preposition / conjunction endings that signal an interrupted sentence
CONTINUATION_ENDINGS = (
    " of",
    " or",
    " and",
    " the",
    " a",
    " an",
    " to",
    " in",
    " for",
    " with",
    " by",
    " from",
    " at",
    " on",
    " that",
    " which",
    " where",
    " as",
    " is",
    " are",
    " was",
    " were",
    " be",
    " but",
    " than",
    " not",
)


# ─── Classification Helpers ──────────────────────────────────────────────────


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


# ─── Pydantic Structured Output Model ────────────────────────────────────────


class TableStructure(BaseModel):
    """Structured representation of an NEC table extracted from OCR fragments.

    The LLM returns this schema via structured output.  The model_validator
    guarantees that every data_row has exactly len(column_headers) cells,
    eliminating the column-count and boundary-sensitivity ambiguities that
    plagued the earlier procedural approach.

    This same schema is used by the manual correction workflow (Phase 3) and
    the merge script (Phase 4).  See docs/table_cleaning.md for the full
    process and docs/table_ambiguity.md for the ambiguity analysis that
    motivated this design.
    """

    title: str
    column_headers: list[str]
    data_rows: list[list[str]]
    footnotes: list[str]

    @model_validator(mode="after")
    def validate_row_widths(self) -> "TableStructure":
        """Ensure every data row has exactly len(column_headers) cells."""
        n_cols = len(self.column_headers)
        for i, row in enumerate(self.data_rows):
            if len(row) != n_cols:
                raise ValueError(f"Row {i} has {len(row)} cells, expected {n_cols} (matching column_headers)")
        return self


# ─── LLM Client & Cache ──────────────────────────────────────────────────────

# System prompt instructs the LLM on how to reconstruct table structure
_SYSTEM_PROMPT = """\
You are a table-reconstruction expert.  You receive OCR-extracted text fragments
from a table in the NFPA 70 National Electrical Code (NEC) 2023 edition.

The OCR captured individual table cells as separate paragraphs, reading
left-to-right, top-to-bottom.  The first fragment is always the table title.

The remaining fragments are a mix of:
  - Column headers: labels for individual columns.
  - Data values: cell contents, given in reading order (left-to-right across
                 each row, then down to the next row).
  - Footnotes: explanatory text that appears after the last data row.
               Footnotes often start with a number, asterisk, or superscript
               character.

Your task: reconstruct the original table structure and return it as JSON that
matches the provided schema exactly.

Rules:
  1. column_headers must list every column header, left to right.
  2. Each element of data_rows is one row of the table, with exactly
     len(column_headers) cell values, in left-to-right order.
  3. Group headers (spanning headers above column headers) should be
     prepended to the relevant column header in parentheses, e.g.
     "Minimum Clear Distance (Condition 1)".
  4. footnotes is a list of footnote strings.  If there are none, return [].
  5. Do NOT invent data that is not present in the fragments.
  6. If a cell is empty or missing, use "-" as a placeholder.
  7. If multiple OCR fragments clearly belong to the same cell (e.g. a long
     equipment name split across lines), merge them into one cell value.
"""

# Cache file stores previously-formatted tables to avoid redundant LLM calls
CACHE_FILE = ROOT / "data" / "intermediate" / "tables" / "table_llm_cache.json"

# Module-level mutable state (lazy-initialised); not true constants.
_LLM_CLIENT: OpenAI | None = None
_LLM_DEPLOYMENT: str = ""
_LLM_AVAILABLE: bool | None = None  # None = not yet checked
_LLM_CACHE: dict[str, dict] | None = None  # None = not yet loaded


def _init_llm() -> bool:
    """Lazy-initialise the Azure OpenAI client.  Returns True if available."""
    global _LLM_CLIENT, _LLM_DEPLOYMENT, _LLM_AVAILABLE  # pylint: disable=global-statement
    if _LLM_AVAILABLE is not None:
        return _LLM_AVAILABLE

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

    # All three variables must be set for LLM formatting to work
    if not all([endpoint, api_key, deployment]):
        logger.warning("Azure OpenAI credentials not configured — all tables will use text-block fallback")
        _LLM_AVAILABLE = False
        return False

    base_url = f"{endpoint}/openai/v1/"
    logger.info("Connecting to Azure OpenAI at %s  (deployment=%s)", base_url, deployment)
    _LLM_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
    _LLM_DEPLOYMENT = deployment
    _LLM_AVAILABLE = True
    return True


def _load_cache() -> dict[str, dict]:
    """Load the LLM results cache from disk (creates empty cache if file is missing)."""
    global _LLM_CACHE  # pylint: disable=global-statement
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as fopen:
            _LLM_CACHE = json.load(fopen)
        logger.info("Loaded LLM cache with %d entries from %s", len(_LLM_CACHE), CACHE_FILE)
    else:
        _LLM_CACHE = {}
        logger.info("No LLM cache found — will create %s", CACHE_FILE)
    return _LLM_CACHE


def _save_cache() -> None:
    """Persist the current cache to disk."""
    if _LLM_CACHE is None:
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as fopen:
        json.dump(_LLM_CACHE, fopen, indent=2)


def _call_llm(fragments: list[str]) -> TableStructure | None:
    """Send table fragments to the LLM and return a validated TableStructure, or None on failure."""
    if _LLM_CLIENT is None:
        return None

    # Build numbered fragment list for the user message
    numbered = "\n".join(f"  [{i}] {frag}" for i, frag in enumerate(fragments))
    user_msg = f"Here are the OCR fragments (one per line, numbered):\n\n{numbered}"

    t0 = time.time()
    try:
        completion = _LLM_CLIENT.beta.chat.completions.parse(
            model=_LLM_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=TableStructure,
        )
        elapsed = time.time() - t0
        logger.debug("LLM responded in %.1fs", elapsed)

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.warning("LLM returned a refusal or empty parsed result (%.1fs)", elapsed)
            return None
        return parsed

    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - t0
        logger.error("LLM call failed after %.1fs: %s", elapsed, exc)
        return None


# ─── Markdown Rendering ──────────────────────────────────────────────────────


def _render_markdown(result: TableStructure) -> str:
    """Convert a validated TableStructure into a markdown table string."""
    lines: list[str] = [f"**{result.title}**", ""]

    # Column header row + separator
    lines.append("| " + " | ".join(result.column_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(result.column_headers)) + " |")

    # Data rows
    for row in result.data_rows:
        lines.append("| " + " | ".join(row) + " |")

    # Footnotes as blockquotes
    if result.footnotes:
        lines.append("")
        for footnote in result.footnotes:
            lines.append(f"> {footnote}")

    return "\n".join(lines)


def _format_text_block(title: str, content_parts: list[str]) -> str:
    """Fallback: store the table as a clearly-delimited text block preserving all content."""
    lines = [f"**{title}**", ""]
    for part in content_parts[1:]:  # skip title (already used above)
        lines.append(part)
    return "\n".join(lines)


def format_table(title: str, content_parts: list[str]) -> str:
    """Format extracted table content as markdown using the LLM, with text-block fallback.

    Checks the LLM cache first.  On a cache miss, calls the LLM and caches the
    result.  Falls back to a plain text block if the LLM is unavailable, the
    call fails, or Pydantic validation rejects the response.

    Tables that this function produces incorrectly (empty, interleaved, or
    structurally wrong) are caught by the triage and manual correction phases
    described in docs/table_cleaning.md §§ Phase 2–4.
    """
    table_id = get_table_id(title)

    # ── 1. Check cache ────────────────────────────────────────────────────
    cache = _load_cache()
    if table_id in cache:
        logger.debug("Cache hit for %s", table_id)
        try:
            cached = TableStructure(**cache[table_id])
            return _render_markdown(cached)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Cached entry for %s failed validation — re-querying LLM", table_id)

    # ── 2. Attempt LLM formatting ────────────────────────────────────────
    if _init_llm() and len(content_parts) >= 2:
        result = _call_llm(content_parts)
        if result is not None:
            # Cache the successful result and persist immediately
            cache[table_id] = result.model_dump()
            _save_cache()
            return _render_markdown(result)

    # ── 3. Fallback to text block ─────────────────────────────────────────
    logger.debug("Using text-block fallback for %s", table_id)
    return _format_text_block(title, content_parts)


# ─── Dict Utility ────────────────────────────────────────────────────────────


def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    """Re-index a dict with consecutive integer string keys."""
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(i): val for i, (_, val) in enumerate(sorted_items)}


# ─── Main Pipeline Step ──────────────────────────────────────────────────────


def run(paragraphs: dict[str, dict]) -> dict[str, dict]:
    """Detect tables, format them via LLM as markdown, and repair interrupted paragraphs.

    Returns a new paragraphs dict where every table region has been collapsed
    into a single paragraph containing markdown-formatted content.

    This is Phase 1 of the table cleaning process.  The output may still
    contain broken tables (empty rows, interleaved data, structural errors)
    which are addressed in Phases 2–4.  See docs/table_cleaning.md.
    """
    n = len(paragraphs)

    # ── 1. Locate every genuine table start ──────────────────────────────
    table_starts = find_table_starts(paragraphs)
    logger.info("Detected %d table starts", len(table_starts))

    # ── 2. Build table info (end idx, interruption, formatted content) ───
    table_info: list[dict] = []
    for idx, start in enumerate(table_starts):
        next_start = table_starts[idx + 1] if idx + 1 < len(table_starts) else None
        end = find_table_end(paragraphs, start, next_start)
        pre_idx, post_idx = detect_interruption(paragraphs, start, end)
        content_parts = extract_table_content(paragraphs, start, end)

        # Progress logging (LLM calls can be slow on first run)
        logger.info(
            "Formatting table %d/%d: %s (%d fragments)",
            idx + 1,
            len(table_starts),
            get_table_id(content_parts[0]) if content_parts else "unknown",
            len(content_parts),
        )

        table_info.append(
            {
                "table_id": get_table_id(content_parts[0]) if content_parts else "unknown",
                "start": start,
                "end": end,
                "pre_idx": pre_idx,
                "post_idx": post_idx,
                "formatted": format_table(content_parts[0] if content_parts else "", content_parts),
            }
        )

    # ── 3. Build skip-set and merge-map ──────────────────────────────────
    skip_set: set[int] = set()
    merge_map: dict[int, str] = {}  # pre_idx → merged content

    for info in table_info:
        # All paragraphs inside the table region are replaced by the formatted block
        for i in range(info["start"], info["end"] + 1):
            skip_set.add(i)

        # If the table interrupted a paragraph, also skip everything from
        # the table end to the continuation (page markers + the continuation itself)
        if info["post_idx"] is not None:
            for i in range(info["end"] + 1, info["post_idx"] + 1):
                skip_set.add(i)
            pre_content = paragraphs[str(info["pre_idx"])]["content"]
            post_content = paragraphs[str(info["post_idx"])]["content"]
            merge_map[info["pre_idx"]] = pre_content + " " + post_content

    # ── 4. Rebuild the paragraph dict ────────────────────────────────────
    table_start_map = {info["start"]: info for info in table_info}
    emitted_table_ids: set[str] = set()  # Track which table IDs already emitted (dedup)
    output: dict[str, dict] = {}
    out_idx = 0

    for i in range(n):
        if i in skip_set:
            # Emit the formatted table at the start of its region (once per table ID)
            if i in table_start_map:
                info = table_start_map[i]
                tid = info["table_id"]
                if tid not in emitted_table_ids:
                    emitted_table_ids.add(tid)
                    output[str(out_idx)] = {
                        "content": info["formatted"],
                        "page": paragraphs[str(info["start"])]["page"],
                    }
                    out_idx += 1
                else:
                    logger.debug("Dedup: skipping duplicate emission for %s at idx %d", tid, i)
            # All other paragraphs in the region are dropped
            continue

        if i in merge_map:
            # Emit the merged (repaired) paragraph
            output[str(out_idx)] = {
                "content": merge_map[i],
                "page": paragraphs[str(i)]["page"],
            }
            out_idx += 1
        else:
            # Regular paragraph — copy unchanged
            output[str(out_idx)] = paragraphs[str(i)].copy()
            out_idx += 1

    logger.info("Rebuilt paragraph dict: %d -> %d entries", n, out_idx)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Read raw paragraphs
    paragraphs_file = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    with open(paragraphs_file, "r", encoding="utf-8") as fopen:
        raw_paragraphs = json.load(fopen)

    # Run cleaning
    output = run(raw_paragraphs)
    logger.info("Done: %d paragraphs in, %d out", len(raw_paragraphs), len(output))
