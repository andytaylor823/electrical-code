"""Test script: demonstrate table cleaning on specific NEC tables.

Loads the raw paragraphs, runs the table cleaning module, and prints
before/after comparisons for the tables identified in the task:

  - Table 400.5(A)(1)  — does not cross page breaks
  - Table 400.44(B)(1)  — goes right up until a page break
  - Table 400.44(B)(2)  — starts right after a page break
  - Table 400.44(B)(3)  — a normal table
  - Table 400.50         — very long table spanning two pages
  - Table 426.3          — interrupts a surrounding paragraph

Usage:
  python scripts/test_tables.py
"""

import json
import logging
from pathlib import Path

from nec_rag.cleaning import remove_junk_pages, tables

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
PARAGRAPHS_FILE = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"

# Tables to inspect (name, search string to find in content)
TABLES_TO_INSPECT = [
    "Table 400.5(A)(1)",
    "Table 400.44(B)(1)",
    "Table 400.44(B)(2)",
    "Table 400.44(B)(3)",
    "Table 400.50",
    "Table 426.3",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def find_paragraph_by_content(paragraphs, search, *, title_only=True):
    """Find the first paragraph whose content starts with *search*."""
    for key, val in paragraphs.items():
        content = val["content"]
        if title_only and content.startswith(search):
            return int(key)
        if not title_only and search in content:
            return int(key)
    return None


def show_context(paragraphs, center_idx, before=3, after=8, label=""):
    """Print paragraphs around *center_idx* with a marker."""
    n = len(paragraphs)
    lo = max(0, center_idx - before)
    hi = min(n, center_idx + after + 1)
    if label:
        print(f"  [{label}]")
    for j in range(lo, hi):
        p = paragraphs[str(j)]
        marker = " >>> " if j == center_idx else "     "
        content = p["content"][:200]
        # Show first 200 chars; indicate if truncated
        suffix = "..." if len(p["content"]) > 200 else ""
        print(f"{marker}[{j}] (p{p['page']}): {content}{suffix}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    # Load raw paragraphs and apply junk-page removal (matches pipeline position)
    logger.info("Loading paragraphs from %s", PARAGRAPHS_FILE)
    with open(PARAGRAPHS_FILE, "r", encoding="utf-8") as fopen:
        raw = json.load(fopen)
    logger.info("Loaded %d raw paragraphs", len(raw))

    cleaned = remove_junk_pages.run(raw)
    logger.info("After remove_junk_pages: %d paragraphs", len(cleaned))

    # Run table cleaning
    after_tables = tables.run(cleaned)
    logger.info("After tables.run: %d paragraphs", len(after_tables))

    # ── Show before / after for each table ───────────────────────────────
    separator = "=" * 80

    for table_name in TABLES_TO_INSPECT:
        print(f"\n{separator}")
        print(f"  {table_name}")
        print(separator)

        # ── BEFORE ──
        before_idx = find_paragraph_by_content(cleaned, table_name)
        if before_idx is not None:
            print(f"\n  BEFORE (raw paragraphs around index {before_idx}):")
            show_context(cleaned, before_idx, before=3, after=15)
        else:
            print(f"\n  (not found in pre-cleaning paragraphs by title match)")

        # ── AFTER ──
        # The formatted table paragraph starts with "**Table ..."
        search_key = f"**{table_name}"
        after_idx = find_paragraph_by_content(after_tables, search_key)
        if after_idx is not None:
            print(f"  AFTER (formatted table at index {after_idx}):")
            # Print the full formatted content (may be multi-line)
            formatted = after_tables[str(after_idx)]["content"]
            for line in formatted.split("\n"):
                print(f"     {line}")
            print()
            # Also show surrounding context
            print(f"  AFTER context:")
            show_context(after_tables, after_idx, before=2, after=3)
        else:
            print(f"  (formatted table not found — may have been merged or missed)")
            print()

    # ── Special: show paragraph-interruption repair for Table 426.3 ──────
    print(f"\n{separator}")
    print("  PARAGRAPH INTERRUPTION REPAIR — Table 426.3")
    print(separator)

    # Before: show the interrupted paragraph
    before_idx = find_paragraph_by_content(cleaned, "(2) They shall be permitted", title_only=False)
    if before_idx is not None:
        print(f"\n  BEFORE — paragraph cut mid-sentence at index {before_idx}:")
        show_context(cleaned, before_idx, before=1, after=12)

    # After: show the merged paragraph
    after_idx = find_paragraph_by_content(after_tables, "(2) They shall be permitted", title_only=False)
    if after_idx is not None:
        print(f"  AFTER — merged paragraph at index {after_idx}:")
        print(f"     Content: {after_tables[str(after_idx)]['content'][:300]}")
        print()
        show_context(after_tables, after_idx, before=1, after=4)

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  SUMMARY")
    print(separator)
    print(f"  Paragraphs before table cleaning: {len(cleaned)}")
    print(f"  Paragraphs after table cleaning:  {len(after_tables)}")
    print(f"  Paragraphs removed (collapsed into table blocks): {len(cleaned) - len(after_tables)}")

    # Count how many formatted tables were produced
    table_count = sum(1 for v in after_tables.values() if v["content"].startswith("**Table "))
    md_table_count = sum(1 for v in after_tables.values() if "| --- |" in v["content"] or "|---|" in v["content"])
    print(f"  Total formatted table paragraphs: {table_count}")
    print(f"  Tables with markdown grid:        {md_table_count}")
    print(f"  Tables with text-block fallback:  {table_count - md_table_count}")
    print()


if __name__ == "__main__":
    main()
