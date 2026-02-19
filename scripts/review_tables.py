"""Interactive review tool for problem tables in the NEC cleaning pipeline.

Scans the cleaned output and LLM cache for problem tables (empty/near-empty
cache entries, duplicate table titles, text-block fallbacks), presents each
one with clear diagnostic info, and saves classifications to table_corrections.json.

Usage:
    source .venv/bin/activate
    python scripts/review_tables.py
"""

import json
import logging
import re
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

CLEAN_JSON = ROOT / "data" / "intermediate" / "NFPA 70 NEC 2023_clean.json"
CACHE_FILE = ROOT / "data" / "intermediate" / "tables" / "table_llm_cache.json"
RAW_PARAGRAPHS_FILE = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
CORRECTIONS_FILE = ROOT / "data" / "intermediate" / "tables" / "table_corrections.json"

# ── Fix categories ────────────────────────────────────────────────────────────

VALID_CATEGORIES = ("stolen_data", "multi_page_merge", "llm_retry", "llm_retry_with_instructions", "manual_override", "ok")

# ── Raw fragment helpers (shared logic with apply_table_corrections.py) ───────

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


def _is_page_marker(content: str) -> bool:
    """Return True if the paragraph is a page marker."""
    if any(content.startswith(p) for p in PAGE_MARKER_PREFIXES):
        return True
    if PAGE_NUM_RE.match(content):
        return True
    if content.startswith("ARTICLE ") and content.isupper():
        return True
    return bool(SECTION_NUM_ONLY_RE.match(content))


def _get_table_id(content: str) -> str:
    """Extract and normalise a table ID from a title string."""
    match = TABLE_ID_RE.match(content)
    raw_id = match.group(1) if match else content
    return raw_id.replace(" ", "")


def _find_raw_table_region(raw_paragraphs, cache_id):
    """Locate the raw paragraph index range for a table.

    Returns (start_idx, end_idx) or (None, None) if not found.
    """
    spaced_id = re.sub(r"(Table)(\d)", r"\1 \2", cache_id)
    n = len(raw_paragraphs)

    # Find the table title paragraph in raw data
    start_idx = None
    for i in range(n):
        content = raw_paragraphs[str(i)]["content"]
        if TABLE_TITLE_RE.match(content):
            tid = _get_table_id(content)
            if tid == cache_id or content.startswith(spaced_id):
                start_idx = i
                break
    if start_idx is None:
        return None, None

    # Walk forward to find the end boundary
    end_idx = start_idx
    for i in range(start_idx + 1, min(n, start_idx + 300)):
        content = raw_paragraphs[str(i)]["content"]
        if _is_page_marker(content):
            continue
        if TABLE_TITLE_RE.match(content) and _get_table_id(content) != cache_id:
            break
        if TABLE_TITLE_RE.match(content) and i + 1 < n and raw_paragraphs[str(i + 1)]["content"].strip() == "Continued":
            continue
        if re.match(r"^\d{2,}\.\d+ [A-Z]", content) and len(content) > 20:
            break
        if re.match(r"^Part [IVX]+\.", content):
            break
        end_idx = i

    return start_idx, end_idx


def _extract_clean_fragments(raw_paragraphs, start_idx, end_idx, cache_id):
    """Extract clean table fragments from a raw paragraph region, skipping noise."""
    parts = []
    skip_next = False
    for i in range(start_idx, end_idx + 1):
        content = raw_paragraphs[str(i)]["content"]
        if _is_page_marker(content):
            continue
        if content.strip().lower() in ("(continues)", "continued"):
            skip_next = True
            continue
        if skip_next:
            if TABLE_TITLE_RE.match(content) and _get_table_id(content) == cache_id:
                continue
            if content.strip() == "Continued":
                skip_next = False
                continue
            skip_next = False
        parts.append(content)
    return parts


def get_raw_fragments(raw_paragraphs, cache_id):
    """Extract the clean OCR fragments that would be sent to the LLM for a table.

    Returns (fragments_list, raw_start_idx, raw_end_idx) or ([], None, None).
    """
    start_idx, end_idx = _find_raw_table_region(raw_paragraphs, cache_id)
    if start_idx is None:
        return [], None, None
    parts = _extract_clean_fragments(raw_paragraphs, start_idx, end_idx, cache_id)
    return parts, start_idx, end_idx


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data():
    """Load cleaned paragraphs, LLM cache, and raw paragraphs."""
    logger.info("Loading cleaned paragraphs from %s", CLEAN_JSON)
    with open(CLEAN_JSON, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)
    logger.info("Loaded %d paragraphs", len(paragraphs))

    logger.info("Loading LLM cache from %s", CACHE_FILE)
    with open(CACHE_FILE, "r", encoding="utf-8") as fopen:
        cache = json.load(fopen)
    logger.info("Loaded %d cache entries", len(cache))

    logger.info("Loading raw paragraphs from %s", RAW_PARAGRAPHS_FILE)
    with open(RAW_PARAGRAPHS_FILE, "r", encoding="utf-8") as fopen:
        raw_paragraphs = json.load(fopen)
    logger.info("Loaded %d raw paragraphs", len(raw_paragraphs))

    return paragraphs, cache, raw_paragraphs


# ── Table index builder ──────────────────────────────────────────────────────


def build_table_index(paragraphs, cache):
    """Build an ordered list of all table paragraphs with metadata.

    Returns a list of dicts: {idx, title, cache_id, rows, page, has_markdown}
    """
    table_index = []
    for i in range(len(paragraphs)):
        content = paragraphs[str(i)]["content"]
        if not content.startswith("**Table "):
            continue
        title_match = re.match(r"\*\*(.+?)\*\*", content)
        if not title_match:
            continue

        title = title_match.group(1)
        id_match = re.match(r"(Table \d+\.\d+(?:\s*\([^)]*\))*)", title)
        cache_id = id_match.group(1).replace(" ", "") if id_match else title.replace(" ", "")
        rows = len(cache[cache_id]["data_rows"]) if cache_id in cache else -1
        has_markdown = "|" in content

        table_index.append(
            {
                "idx": i,
                "title": title,
                "cache_id": cache_id,
                "rows": rows,
                "page": paragraphs[str(i)]["page"],
                "has_markdown": has_markdown,
            }
        )
    return table_index


def find_in_table_index(table_index, cache_id):
    """Find the position(s) of a cache_id in the table index."""
    return [j for j, t in enumerate(table_index) if t["cache_id"] == cache_id]


# ── Problem detection ─────────────────────────────────────────────────────────


def find_problem_tables(_paragraphs, cache, table_index):
    """Scan for problem tables across three categories.

    Returns a list of problem dicts with diagnostic info.
    """
    problems = []
    seen_ids = set()

    # Category 1: Cache entries with 0-1 data rows
    for cache_id, entry in sorted(cache.items()):
        if len(entry["data_rows"]) <= 1:
            problem_id = f"empty__{cache_id}"
            if problem_id not in seen_ids:
                seen_ids.add(problem_id)
                problems.append(
                    {
                        "table_id": cache_id,
                        "problem_type": "empty_cache",
                        "cache_entry": entry,
                    }
                )

    # Category 2: Text-block fallback tables (no markdown pipes in output)
    for entry in table_index:
        if not entry["has_markdown"]:
            cache_id = entry["cache_id"]
            problem_id = f"fallback__{cache_id}"
            if problem_id not in seen_ids:
                seen_ids.add(problem_id)
                problems.append(
                    {
                        "table_id": cache_id,
                        "problem_type": "text_block_fallback",
                        "paragraph_idx": entry["idx"],
                        "page": entry["page"],
                    }
                )

    return problems


# ── Display helpers ───────────────────────────────────────────────────────────


def fmt_row_count(rows):
    """Format row count with warning indicator."""
    if rows == 0:
        return "0 rows  <-- EMPTY"
    if rows == 1:
        return "1 row   <-- SUSPICIOUS"
    if rows < 0:
        return "NOT IN CACHE"
    return f"{rows} rows"


def show_neighboring_tables(table_index, cache_id, radius=3):
    """Show the tables before and after this one, with row counts."""
    positions = find_in_table_index(table_index, cache_id)
    if not positions:
        print("    (table not found in cleaned output)")
        return

    # Use the first occurrence
    pos = positions[0]
    start = max(0, pos - radius)
    end = min(len(table_index), pos + radius + 1)

    for j in range(start, end):
        entry = table_index[j]
        marker = ">>>" if j == pos else "   "
        print(f"    {marker} pg {entry['page']:>3}  {entry['cache_id']:35s}  {fmt_row_count(entry['rows'])}")


def show_cache_entry_detail(cache, cache_id):
    """Show the cache entry for a table in a readable format."""
    if cache_id not in cache:
        print("    (no cache entry)")
        return

    entry = cache[cache_id]
    print(f"    Title:    {entry['title']}")
    print(f"    Columns:  {entry['column_headers']}")
    print(f"    Rows:     {len(entry['data_rows'])}")
    if entry["data_rows"]:
        # Show first 5 rows
        for row in entry["data_rows"][:5]:
            print(f"              {row}")
        if len(entry["data_rows"]) > 5:
            print(f"              ... ({len(entry['data_rows']) - 5} more)")
    if entry["footnotes"]:
        print(f"    Notes:    {entry['footnotes'][0][:80]}")
        if len(entry["footnotes"]) > 1:
            print(f"              ... ({len(entry['footnotes']) - 1} more notes)")


def count_raw_fragments(raw_paragraphs, cache_id):
    """Count how many raw OCR fragments exist for this table ID."""
    # Convert cache_id back to a raw table title prefix (e.g. "Table 240.6")
    spaced_id = re.sub(r"(Table)(\d)", r"\1 \2", cache_id)
    count = 0
    found_start = False
    for i in range(len(raw_paragraphs)):
        content = raw_paragraphs[str(i)]["content"]
        if content.startswith(spaced_id) or content.startswith(cache_id):
            found_start = True
            count += 1
            continue
        if found_start:
            # Keep counting short fragments that are likely table cells
            if len(content) < 80 and not re.match(r"^\d{2,}\.\d+ [A-Z]", content):
                count += 1
            else:
                break
    return count


# ── Display per problem type ──────────────────────────────────────────────────


def display_empty_cache(problem, _paragraphs, cache, table_index, raw_paragraphs):
    """Display an empty/near-empty cache entry problem."""
    cache_id = problem["table_id"]
    entry = problem["cache_entry"]
    n_rows = len(entry["data_rows"])
    n_raw = count_raw_fragments(raw_paragraphs, cache_id)

    print()
    print("  WHAT'S WRONG:")
    if n_rows == 0:
        print("    The LLM extracted 0 data rows for this table.")
        print("    Either its data was absorbed by a neighboring table (stolen_data),")
        print("    or the LLM simply failed to parse it (llm_retry).")
    else:
        print(f"    The LLM only extracted {n_rows} data row (suspiciously few).")
        print("    This might be correct for a tiny table, or data may be missing.")

    print()
    print(f"  RAW FRAGMENTS: ~{n_raw} OCR fragments found in raw data")

    print()
    print("  WHAT THE LLM PRODUCED:")
    show_cache_entry_detail(cache, cache_id)

    print()
    print("  NEIGHBORING TABLES (look for one with suspiciously many rows):")
    show_neighboring_tables(table_index, cache_id)


def display_text_block_fallback(problem, paragraphs, cache, table_index):  # pylint: disable=unused-argument
    """Display a text-block fallback problem."""
    cache_id = problem["table_id"]
    pidx = problem["paragraph_idx"]
    content = paragraphs[str(pidx)]["content"]

    print()
    print("  WHAT'S WRONG:")
    print("    This table was output as plain text instead of a markdown table.")
    print("    The LLM either failed to parse it, or it was never sent to the LLM.")

    # Show if there's a cache entry
    print()
    if cache_id in cache:
        print("  CACHE ENTRY EXISTS (but output is still text-block):")
        show_cache_entry_detail(cache, cache_id)
    else:
        print("  NO CACHE ENTRY (table was never successfully parsed by LLM)")

    print()
    print("  TEXT OUTPUT (first 600 chars):")
    preview = content[:600]
    # Wrap long lines for readability
    for line in preview.split("\n"):
        print(f"    {line}")
    if len(content) > 600:
        print(f"    ... ({len(content) - 600} more chars)")

    print()
    print("  NEIGHBORING TABLES:")
    show_neighboring_tables(table_index, cache_id)


# ── Main display router ──────────────────────────────────────────────────────


def display_problem(problem, paragraphs, cache, table_index, raw_paragraphs, index, total):
    """Display a single problem table for interactive review."""
    cache_id = problem["table_id"]
    ptype = problem["problem_type"]

    # Header
    print()
    print("=" * 80)
    print(f"  PROBLEM {index + 1}/{total}")
    print(f"  Table:   {cache_id}")
    if ptype == "empty_cache":
        n_rows = len(problem["cache_entry"]["data_rows"])
        print(f"  Issue:   Cache entry has {n_rows} data rows (expected many more)")
    elif ptype == "text_block_fallback":
        print(f"  Issue:   Output is plain text, not a markdown table (page {problem['page']})")
    print("=" * 80)

    # Detailed display per type
    if ptype == "empty_cache":
        display_empty_cache(problem, paragraphs, cache, table_index, raw_paragraphs)
    elif ptype == "text_block_fallback":
        display_text_block_fallback(problem, paragraphs, cache, table_index)


# ── Classification prompt ─────────────────────────────────────────────────────


def prompt_classification(problem, raw_paragraphs):
    """Prompt the user to classify a problem table. Returns a correction dict."""
    print()
    print("-" * 60)
    print(f"  HOW SHOULD WE FIX  {problem['table_id']}  ?")
    print()
    print("    1. stolen_data      Another table absorbed this table's data.")
    print("                        (We'll re-extract and re-send to LLM.)")
    print("    2. multi_page_merge This table spans multiple pages and got")
    print("                        split into separate detections. Merge all.")
    print("    3. llm_retry        Raw data looks fine; just retry the LLM.")
    print("    4. llm_retry+instr  Retry the LLM with extra instructions you")
    print("                        provide (shows fragment boundaries first).")
    print("    5. manual_override  Provide a manual fix, or skip this table.")
    print("    6. ok               Actually fine -- no fix needed.")
    print()

    name_map = {
        "1": "stolen_data",
        "2": "multi_page_merge",
        "3": "llm_retry",
        "4": "llm_retry_with_instructions",
        "5": "manual_override",
        "6": "ok",
    }

    while True:
        choice = input("  Enter choice (1-6): ").strip().lower()
        choice = name_map.get(choice, choice)
        if choice in VALID_CATEGORIES:
            break
        print("  Invalid. Enter a number 1-6.")

    correction = {
        "table_id": problem["table_id"],
        "problem_type": problem["problem_type"],
        "fix_category": choice,
    }

    # ── stolen_data: ask which table stole the data ───────────────────────
    if choice == "stolen_data":
        thief = input("  Which table absorbed the data? (e.g. Table240.4(G)): ").strip()
        correction["stolen_by"] = thief

    # ── llm_retry_with_instructions: show fragments, ask for instructions ─
    if choice == "llm_retry_with_instructions":
        _show_fragment_boundaries(problem["table_id"], raw_paragraphs)
        print()
        print("  Type your extra instructions for the LLM (single line):")
        print("  Example: 'This table has 9 columns. The first column is Insulation type.'")
        extra = input("  > ").strip()
        correction["extra_instructions"] = extra

    # ── manual_override: ask for JSON or skip ─────────────────────────────
    if choice == "manual_override":
        action = input("  Enter 'skip' to leave as-is, or 'json' to paste TableStructure JSON: ").strip().lower()
        if action == "json":
            print("  Paste JSON (end with empty line):")
            json_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                json_lines.append(line)
            try:
                correction["override_data"] = json.loads("\n".join(json_lines))
            except json.JSONDecodeError as exc:
                print(f"  Invalid JSON: {exc}. Marking as skip.")
                correction["override_action"] = "skip"
        else:
            correction["override_action"] = "skip"

    return correction


def _show_fragment_boundaries(cache_id, raw_paragraphs):
    """Show the first and last few raw fragments for a table, so the user
    can see exactly what context the LLM will receive."""
    fragments, start_idx, end_idx = get_raw_fragments(raw_paragraphs, cache_id)
    if not fragments:
        print(f"\n  Could not find raw fragments for {cache_id}")
        return

    n_frags = len(fragments)
    n_show = 3  # number of lines to show at each end

    print()
    print(f"  RAW FRAGMENTS for {cache_id}  ({n_frags} total, raw idx {start_idx}-{end_idx})")
    print()
    print("  FIRST fragments (start of table):")
    for i, frag in enumerate(fragments[:n_show]):
        print(f"    [{i}] {frag}")

    if n_frags > n_show * 2:
        print(f"    ... ({n_frags - n_show * 2} more fragments) ...")

    print()
    print("  LAST fragments (end of table / boundary):")
    tail_start = max(n_show, n_frags - n_show)
    for i in range(tail_start, n_frags):
        print(f"    [{i}] {fragments[i]}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    """Run the interactive review process."""
    paragraphs, cache, raw_paragraphs = load_data()

    # Build table index for neighbor lookups
    table_index = build_table_index(paragraphs, cache)
    logger.info("Built table index with %d entries", len(table_index))

    # Load existing corrections if resuming
    corrections = []
    already_reviewed = set()
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as fopen:
            corrections = json.load(fopen)
        already_reviewed = {c["table_id"] + "__" + c["problem_type"] for c in corrections}
        print(f"\nResuming: {len(corrections)} tables already reviewed (will be skipped).\n")

    # Find problem tables
    problems = find_problem_tables(paragraphs, cache, table_index)
    print(f"\nFound {len(problems)} problem tables.\n")

    # Filter already-reviewed
    pending = [p for p in problems if (p["table_id"] + "__" + p["problem_type"]) not in already_reviewed]
    if not pending:
        print("All problems reviewed. Run scripts/apply_table_corrections.py next.")
        return

    print(f"{len(pending)} remaining to review.\n")

    # Interactive loop
    for i, problem in enumerate(pending):
        display_problem(problem, paragraphs, cache, table_index, raw_paragraphs, i, len(pending))
        correction = prompt_classification(problem, raw_paragraphs)
        corrections.append(correction)

        # Save after each (resume-safe)
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as fopen:
            json.dump(corrections, fopen, indent=2)
        print(f"\n  Saved. ({i + 1}/{len(pending)} done)\n")

    print(f"\nAll done! Corrections saved to {CORRECTIONS_FILE}")
    print("Next: python scripts/apply_table_corrections.py")


if __name__ == "__main__":
    main()
