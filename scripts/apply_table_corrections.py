"""Apply table corrections from table_corrections.json to create a corrected LLM cache.

Reads the corrections file produced by review_tables.py and applies automated
fixes to the LLM table cache.  Produces a new corrected cache file and then
re-runs the full cleaning pipeline to regenerate clean.json and clean.txt.

Fix categories handled:
  - stolen_data:      Re-extract raw fragments for the table, re-send to LLM
  - multi_page_merge: Merge raw fragments from all occurrences, re-send to LLM
  - llm_retry:        Re-send original fragments to LLM
  - manual_override:  Copy provided JSON directly into cache, or skip
  - ok:               No action

Usage:
    source .venv/bin/activate
    python scripts/apply_table_corrections.py
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator

# Add project root to path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_PARAGRAPHS_FILE = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
CACHE_FILE = ROOT / "data" / "intermediate" / "tables" / "table_llm_cache.json"
CORRECTIONS_FILE = ROOT / "data" / "intermediate" / "tables" / "table_corrections.json"
CORRECTED_CACHE_FILE = ROOT / "data" / "intermediate" / "tables" / "table_llm_cache_corrected.json"

# ── Pydantic model (must match tables.py) ─────────────────────────────────────


class TableStructure(BaseModel):
    """Structured representation of an NEC table extracted from OCR fragments."""

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
                raise ValueError(f"Row {i} has {len(row)} cells, expected {n_cols}")
        return self


# ── LLM system prompt (same as tables.py) ─────────────────────────────────────

SYSTEM_PROMPT = """\
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

# ── Table-detection regex patterns (from tables.py) ───────────────────────────

TABLE_TITLE_RE = re.compile(r"^Table \d+\.\d+")
TABLE_ID_RE = re.compile(r"(Table \d+\.\d+(?:\s*\([^)]*\))*)")
PAGE_MARKER_PREFIXES = (
    "2023 Edition NATIONAL ELECTRICAL CODE",
    "NATIONAL ELECTRICAL CODE 2023 Edition",
    "Copyright @NFPA",
    "Copyright @ NFPA",
    "EDUFIRE.IR",
    "Telegram: EDUFIRE.IR",
)
SECTION_NUM_ONLY_RE = re.compile(r"^\d{2,}\.\d+$")
PAGE_NUM_RE = re.compile(r"^70-\d+$")


def is_page_marker(content: str) -> bool:
    """Return True if the paragraph is a page marker (header/footer/copyright/etc)."""
    if any(content.startswith(prefix) for prefix in PAGE_MARKER_PREFIXES):
        return True
    if PAGE_NUM_RE.match(content):
        return True
    if content.startswith("ARTICLE ") and content.isupper():
        return True
    if SECTION_NUM_ONLY_RE.match(content):
        return True
    return False


def get_table_id(content: str) -> str:
    """Extract and normalise a table ID from a title string."""
    match = TABLE_ID_RE.match(content)
    raw_id = match.group(1) if match else content
    return raw_id.replace(" ", "")


# ── LLM helpers ───────────────────────────────────────────────────────────────


def init_llm():
    """Initialise the Azure OpenAI client. Returns (client, deployment) or (None, None)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

    if not all([endpoint, api_key, deployment]):
        logger.error("Azure OpenAI credentials not configured")
        return None, None

    base_url = f"{endpoint}/openai/v1/"
    logger.info("Connecting to Azure OpenAI at %s (deployment=%s)", base_url, deployment)
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, deployment


def call_llm(client, deployment, fragments: list[str], extra_instructions: str = "") -> TableStructure | None:
    """Send table fragments to the LLM and return a validated TableStructure.

    If extra_instructions is provided, it is appended to the user message
    as additional guidance for the LLM.
    """
    if client is None:
        return None

    numbered = "\n".join(f"  [{i}] {frag}" for i, frag in enumerate(fragments))
    user_msg = f"Here are the OCR fragments (one per line, numbered):\n\n{numbered}"

    # Append extra instructions if provided (e.g. from llm_retry_with_instructions)
    if extra_instructions:
        user_msg += f"\n\nAdditional instructions:\n{extra_instructions}"

    t0 = time.time()
    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=TableStructure,
        )
        elapsed = time.time() - t0
        logger.info("LLM responded in %.1fs", elapsed)

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.warning("LLM returned empty result (%.1fs)", elapsed)
            return None
        return parsed

    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - t0
        logger.error("LLM call failed after %.1fs: %s", elapsed, exc)
        return None


# ── Fragment extraction from raw paragraphs ───────────────────────────────────


def find_table_region_in_raw(raw_paragraphs, table_id, start_search=0):
    """Find the raw paragraph indices for a table by its ID.

    Returns (start_idx, end_idx) or (None, None) if not found.
    Scans from start_search forward.
    """
    n = len(raw_paragraphs)

    # Find the table title paragraph
    start_idx = None
    for i in range(start_search, n):
        content = raw_paragraphs[str(i)]["content"]
        if TABLE_TITLE_RE.match(content) and get_table_id(content) == table_id:
            start_idx = i
            break

    if start_idx is None:
        return None, None

    # Walk forward to find the end (next section boundary or next different table)
    end_idx = start_idx
    for i in range(start_idx + 1, min(n, start_idx + 300)):
        content = raw_paragraphs[str(i)]["content"]

        # Skip page markers
        if is_page_marker(content):
            continue

        # If we hit a different table title, stop
        if TABLE_TITLE_RE.match(content):
            other_id = get_table_id(content)
            if other_id != table_id:
                break
            # Same table ID (continuation) -- skip continuation headers
            if i + 1 < n and raw_paragraphs[str(i + 1)]["content"].strip() == "Continued":
                continue

        # Section boundary (e.g. "250.50 Uses Permitted.")
        if re.match(r"^\d{2,}\.\d+ [A-Z]", content) and len(content) > 20:
            break
        # Part header
        if re.match(r"^Part [IVX]+\.", content):
            break

        end_idx = i

    return start_idx, end_idx


def extract_fragments(raw_paragraphs, start_idx, end_idx, table_id):
    """Extract clean table fragments from a raw paragraph region."""
    parts = []
    skip_next = False

    for i in range(start_idx, end_idx + 1):
        content = raw_paragraphs[str(i)]["content"]

        # Skip page markers
        if is_page_marker(content):
            continue

        # Skip "(continues)" markers and their continuation headers
        if content.strip().lower() in ("(continues)", "continued"):
            skip_next = True
            continue
        if skip_next:
            if TABLE_TITLE_RE.match(content) and get_table_id(content) == table_id:
                continue
            if content.strip() == "Continued":
                skip_next = False
                continue
            skip_next = False

        parts.append(content)

    return parts


def find_all_table_regions(raw_paragraphs, table_id):
    """Find ALL raw regions for a table ID (for multi-page merge)."""
    regions = []
    search_from = 0
    n = len(raw_paragraphs)

    while search_from < n:
        start, end = find_table_region_in_raw(raw_paragraphs, table_id, start_search=search_from)
        if start is None:
            break
        regions.append((start, end))
        search_from = end + 1

    return regions


# ── Apply corrections ─────────────────────────────────────────────────────────


def apply_corrections(corrections, cache, raw_paragraphs, client, deployment):  # pylint: disable=too-many-branches
    """Apply each correction to the cache. Returns the updated cache dict."""
    corrected_cache = dict(cache)  # shallow copy

    for correction in corrections:
        table_id = correction["table_id"]
        fix = correction["fix_category"]

        logger.info("Processing %s: fix=%s", table_id, fix)

        if fix == "ok":
            # No action needed
            logger.info("  Skipping %s (marked ok)", table_id)
            continue

        if fix == "manual_override":
            # Handle manual override or skip
            if correction.get("override_action") == "skip":
                logger.info("  Skipping %s (manual skip)", table_id)
                continue
            if "override_data" in correction:
                # Validate and store the provided data
                try:
                    validated = TableStructure(**correction["override_data"])
                    corrected_cache[table_id] = validated.model_dump()
                    logger.info("  Applied manual override for %s", table_id)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("  Manual override for %s failed validation: %s", table_id, exc)
            continue

        if fix == "stolen_data":
            # Re-extract this table's own fragments from the raw data and re-send to LLM
            start, end = find_table_region_in_raw(raw_paragraphs, table_id)
            if start is None:
                logger.warning("  Could not find %s in raw paragraphs", table_id)
                continue
            fragments = extract_fragments(raw_paragraphs, start, end, table_id)
            logger.info("  Re-extracting %s: %d fragments from raw[%d:%d]", table_id, len(fragments), start, end)

            if len(fragments) >= 2:
                result = call_llm(client, deployment, fragments)
                if result is not None:
                    corrected_cache[table_id] = result.model_dump()
                    logger.info("  Updated cache for %s (%d rows)", table_id, len(result.data_rows))
                else:
                    logger.warning("  LLM failed for %s", table_id)
            else:
                logger.warning("  Too few fragments for %s (%d)", table_id, len(fragments))
            continue

        if fix == "multi_page_merge":
            # Find all regions for this table, merge fragments, re-send to LLM
            regions = find_all_table_regions(raw_paragraphs, table_id)
            if not regions:
                logger.warning("  Could not find any regions for %s", table_id)
                continue

            all_fragments = []
            for start, end in regions:
                frags = extract_fragments(raw_paragraphs, start, end, table_id)
                # For subsequent regions, skip the title (already in first region)
                if all_fragments and frags and TABLE_TITLE_RE.match(frags[0]):
                    frags = frags[1:]
                all_fragments.extend(frags)

            logger.info("  Merged %d regions -> %d total fragments for %s", len(regions), len(all_fragments), table_id)

            if len(all_fragments) >= 2:
                result = call_llm(client, deployment, all_fragments)
                if result is not None:
                    corrected_cache[table_id] = result.model_dump()
                    logger.info("  Updated cache for %s (%d rows)", table_id, len(result.data_rows))
                else:
                    logger.warning("  LLM failed for %s", table_id)
            else:
                logger.warning("  Too few merged fragments for %s (%d)", table_id, len(all_fragments))
            continue

        if fix in ("llm_retry", "llm_retry_with_instructions"):
            # Re-send the original fragments to the LLM, optionally with extra instructions
            start, end = find_table_region_in_raw(raw_paragraphs, table_id)
            if start is None:
                logger.warning("  Could not find %s in raw paragraphs", table_id)
                continue
            fragments = extract_fragments(raw_paragraphs, start, end, table_id)
            extra = correction.get("extra_instructions", "")
            if extra:
                logger.info("  Retrying LLM for %s with extra instructions: %s", table_id, extra[:80])
            else:
                logger.info("  Retrying LLM for %s: %d fragments", table_id, len(fragments))

            if len(fragments) >= 2:
                result = call_llm(client, deployment, fragments, extra_instructions=extra)
                if result is not None:
                    corrected_cache[table_id] = result.model_dump()
                    logger.info("  Updated cache for %s (%d rows)", table_id, len(result.data_rows))
                else:
                    logger.warning("  LLM retry failed for %s", table_id)
            else:
                logger.warning("  Too few fragments for %s (%d)", table_id, len(fragments))
            continue

        logger.warning("  Unknown fix category '%s' for %s", fix, table_id)

    return corrected_cache


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    """Load corrections, apply fixes, write corrected cache, re-run pipeline."""
    # Load corrections
    if not CORRECTIONS_FILE.exists():
        logger.error("No corrections file found at %s. Run review_tables.py first.", CORRECTIONS_FILE)
        sys.exit(1)

    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as fopen:
        corrections = json.load(fopen)
    logger.info("Loaded %d corrections from %s", len(corrections), CORRECTIONS_FILE)

    # Count fixes needed (skip 'ok' entries)
    fixes_needed = [c for c in corrections if c["fix_category"] != "ok"]
    llm_needed = [c for c in fixes_needed if c["fix_category"] in ("stolen_data", "multi_page_merge", "llm_retry")]
    logger.info("%d fixes to apply (%d require LLM calls)", len(fixes_needed), len(llm_needed))

    # Load existing cache
    with open(CACHE_FILE, "r", encoding="utf-8") as fopen:
        cache = json.load(fopen)
    logger.info("Loaded %d cache entries from %s", len(cache), CACHE_FILE)

    # Load raw paragraphs (needed for fragment extraction)
    with open(RAW_PARAGRAPHS_FILE, "r", encoding="utf-8") as fopen:
        raw_paragraphs = json.load(fopen)
    logger.info("Loaded %d raw paragraphs", len(raw_paragraphs))

    # Initialise LLM if needed
    client, deployment = None, None
    if llm_needed:
        client, deployment = init_llm()
        if client is None:
            logger.error("LLM required for %d fixes but not available", len(llm_needed))
            sys.exit(1)

    # Apply corrections
    corrected_cache = apply_corrections(corrections, cache, raw_paragraphs, client, deployment)

    # Write corrected cache
    with open(CORRECTED_CACHE_FILE, "w", encoding="utf-8") as fopen:
        json.dump(corrected_cache, fopen, indent=2)
    logger.info("Wrote corrected cache to %s (%d entries)", CORRECTED_CACHE_FILE, len(corrected_cache))

    # Copy corrected cache over the original so the pipeline uses it
    with open(CACHE_FILE, "w", encoding="utf-8") as fopen:
        json.dump(corrected_cache, fopen, indent=2)
    logger.info("Updated original cache at %s", CACHE_FILE)

    # Re-run the full cleaning pipeline
    logger.info("Re-running cleaning pipeline...")
    from nec_rag.cleaning.clean import load_paragraphs, run_cleaning_pipeline, save_outputs  # pylint: disable=wrong-import-position,import-outside-toplevel

    output_dir = ROOT / "data" / "intermediate"
    raw = load_paragraphs()
    cleaned = run_cleaning_pipeline(raw)
    save_outputs(cleaned, output_dir)
    logger.info("Pipeline complete. clean.json and clean.txt regenerated.")


if __name__ == "__main__":
    main()
