"""Standalone script to test LLM-based table reconstruction on known NEC tables.

Loads raw OCR paragraphs, uses the existing procedural code to detect and extract
table cell values, then sends those values to Azure OpenAI with a Pydantic-enforced
structured output schema.  The LLM reconstructs column headers, data rows, and
footnotes, and the script renders the result as a markdown table for manual
ground-truth verification.

Usage:
    python scripts/test_llm_tables.py                       # run all target tables
    python scripts/test_llm_tables.py "Table 426.3"         # run one specific table
    python scripts/test_llm_tables.py --list                 # list all detected tables
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator

# ─── Project setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from nec_rag.cleaning import remove_junk_pages, tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)


# ─── Pydantic structured-output model ────────────────────────────────────────


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
                raise ValueError(f"Row {i} has {len(row)} cells, expected {n_cols} (matching column_headers)")
        return self


# ─── LLM client ──────────────────────────────────────────────────────────────

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
"""


def build_client() -> OpenAI:
    """Create an OpenAI client pointed at the Azure endpoint."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")
    # Azure structured-output support uses the /openai/v1/ base-URL pattern
    base_url = f"{endpoint}/openai/v1/"
    logger.info("Connecting to Azure OpenAI at %s", base_url)
    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(client: OpenAI, deployment: str, fragments: list[str]) -> TableStructure | None:
    """Send table fragments to the LLM and return a validated TableStructure, or None on failure."""
    # Build the user message with numbered fragments for clarity
    numbered = "\n".join(f"  [{i}] {frag}" for i, frag in enumerate(fragments))
    user_msg = f"Here are the OCR fragments (one per line, numbered):\n\n{numbered}"

    logger.info("Sending %d fragments to LLM (%s) ...", len(fragments), deployment)
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
            logger.warning("LLM returned a refusal or empty parsed result")
            return None
        return parsed

    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - t0
        logger.error("LLM call failed after %.1fs: %s", elapsed, exc)
        return None


# ─── Markdown rendering ──────────────────────────────────────────────────────


def render_markdown(result: TableStructure) -> str:
    """Convert a TableStructure into a readable markdown table string."""
    lines: list[str] = [f"**{result.title}**", ""]

    # Column header row + separator
    lines.append("| " + " | ".join(result.column_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(result.column_headers)) + " |")

    # Data rows
    for row in result.data_rows:
        lines.append("| " + " | ".join(row) + " |")

    # Footnotes
    if result.footnotes:
        lines.append("")
        for footnote in result.footnotes:
            lines.append(f"> {footnote}")

    return "\n".join(lines)


# ─── Data loading & table detection ──────────────────────────────────────────

# Target tables for ground-truth testing (table ID -> short description)
TARGET_TABLES = {
    "Table426.3": "Simple 2-col table (article/section references)",
    "Table110.26(A)(1)": "3-condition voltage table (failed procedurally)",
    "Table220.103": "Text-first-col + numeric-second-col (failed procedurally)",
    "Table400.44(B)(3)": "Normal 4-column table (baseline)",
    "Table400.5(A)": "Multi-column ampacity table (complex, may be large)",
}


def load_and_detect() -> tuple[dict[str, dict], list[dict]]:
    """Load paragraphs, run junk-page removal, detect tables, and return (paragraphs, regions)."""
    paragraphs_file = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    logger.info("Loading paragraphs from %s", paragraphs_file)
    with open(paragraphs_file, "r", encoding="utf-8") as fopen:
        raw_paragraphs = json.load(fopen)
    logger.info("Loaded %d raw paragraphs", len(raw_paragraphs))

    # Pre-process: remove junk pages
    paragraphs = remove_junk_pages.run(raw_paragraphs)
    logger.info("After remove_junk_pages: %d paragraphs", len(paragraphs))

    # Detect all table start positions
    table_starts = tables.find_table_starts(paragraphs)
    logger.info("Detected %d table starts", len(table_starts))

    # Build region info for each table
    regions: list[dict] = []
    for idx, start in enumerate(table_starts):
        next_start = table_starts[idx + 1] if idx + 1 < len(table_starts) else None
        end = tables.find_table_end(paragraphs, start, next_start)
        parts = tables.extract_table_content(paragraphs, start, end)
        regions.append(
            {
                "table_id": tables.get_table_id(parts[0]) if parts else "unknown",
                "title": parts[0] if parts else "",
                "parts": parts,
            }
        )
    return paragraphs, regions


def select_tables(regions: list[dict], filter_ids: list[str]) -> list[dict]:
    """Filter regions to the requested target tables."""
    if filter_ids:
        # User specified table IDs on the command line (partial match)
        selected = []
        for region in regions:
            for fid in filter_ids:
                normalised = fid.replace(" ", "")
                if normalised in region["table_id"] or region["table_id"] in normalised:
                    selected.append(region)
                    break
        return selected

    # Default: use the predefined target set
    target_ids = set(TARGET_TABLES.keys())
    return [r for r in regions if r["table_id"] in target_ids]


# ─── Per-table processing ────────────────────────────────────────────────────


def process_table(region: dict, client: OpenAI, deployment: str) -> dict:
    """Run LLM formatting on a single table region and print results."""
    tid = region["table_id"]
    parts = region["parts"]
    desc = TARGET_TABLES.get(tid, "")

    print(f"\n{'=' * 80}")
    print(f"  TABLE: {tid}  {desc}")
    print(f"  Fragments: {len(parts)}")
    print("=" * 80)

    # Show raw fragments
    print("\n--- RAW FRAGMENTS ---")
    for i, part in enumerate(parts):
        display = part if len(part) <= 100 else part[:97] + "..."
        print(f"  [{i:3d}] {display}")

    # Call LLM
    result = call_llm(client, deployment, parts)
    if result is None:
        print("\n--- LLM FAILED --- (see log above)")
        return {"table_id": tid, "success": False, "result": None}

    # Show structured output summary
    print("\n--- LLM RESULT ---")
    print(f"  Title:    {result.title}")
    print(f"  Columns:  {result.column_headers}  ({len(result.column_headers)} cols)")
    print(f"  Rows:     {len(result.data_rows)}")
    print(f"  Footnotes: {len(result.footnotes)}")

    # Render markdown
    md = render_markdown(result)
    print("\n--- MARKDOWN OUTPUT ---\n")
    print(md)

    return {"table_id": tid, "success": True, "result": result}


def print_summary(results: list[dict]) -> None:
    """Print a one-line summary for each processed table."""
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {sum(1 for r in results if r['success'])}/{len(results)} tables succeeded")
    print("=" * 80)
    for res in results:
        status = "OK" if res["success"] else "FAIL"
        extra = ""
        if res["result"]:
            r = res["result"]
            extra = f"  {len(r.column_headers)} cols x {len(r.data_rows)} rows"
        print(f"  [{status}]  {res['table_id']}{extra}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: load data, detect tables, run LLM on targets, print results."""
    list_mode = "--list" in sys.argv
    filter_ids = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    # Load paragraphs and detect all tables
    _, regions = load_and_detect()

    # List mode: print all detected tables and exit
    if list_mode:
        print(f"\n{'=' * 80}")
        print(f"  Detected {len(regions)} tables")
        print("=" * 80 + "\n")
        for region in regions:
            n_parts = len(region["parts"])
            print(f"  {region['table_id']:<30s}  ({n_parts:3d} fragments)  {region['title'][:60]}")
        return

    # Select target tables
    selected = select_tables(regions, filter_ids)
    if not selected:
        logger.error("No tables matched filter: %s", filter_ids)
        logger.info("Use --list to see all detected tables")
        return
    logger.info("Selected %d tables to process", len(selected))

    # Set up LLM client
    client = build_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")
    logger.info("Using deployment: %s", deployment)

    # Process each table and collect results
    results = [process_table(region, client, deployment) for region in selected]

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
