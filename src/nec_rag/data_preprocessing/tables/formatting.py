"""LLM-based table formatting, caching, and markdown rendering.

Sends extracted OCR table fragments to Azure OpenAI with a Pydantic-enforced
structured output schema (TableStructure) to reconstruct column headers, data
rows, and footnotes.  Results are cached on disk to avoid redundant LLM calls.
Falls back to a plain text block when the LLM is unavailable or fails.
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from nec_rag.data_preprocessing.tables.classifiers import get_table_id
from nec_rag.data_preprocessing.tables.schema import TableStructure

logger = logging.getLogger(__name__)


# ─── Project Root & Environment ───────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env")


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
