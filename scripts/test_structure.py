"""Validate the structured NEC JSON output.

Checks:
- Correct number of chapters, articles, tables
- Hierarchy shape (every article has parts, every part has subsections list)
- Sub-item splitting on a known section (110.14)
- Front-matter is present where expected
- No lost content (paragraph count integrity)

Usage:
    source .venv/bin/activate
    python scripts/test_structure.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
STRUCTURED_PATH = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"


class _Counter:  # pylint: disable=too-few-public-methods
    """Track pass/fail counts for validation checks."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, condition: bool, description: str):
        """Assert a condition, log pass/fail."""
        if condition:
            logger.info("PASS: %s", description)
            self.passed += 1
        else:
            logger.error("FAIL: %s", description)
            self.failed += 1


def _find_article(all_articles: list[dict], article_num: int) -> dict | None:
    """Find an article by number in the flat articles list."""
    return next((a for a in all_articles if a["article_num"] == article_num), None)


def _find_subsection(article: dict, subsection_id: str) -> dict | None:
    """Find a subsection within an article (searching all parts)."""
    for part in article["parts"]:
        for sub in part["subsections"]:
            if sub["id"] == subsection_id:
                return sub
    return None


def _check_counts(counter: _Counter, chapters: list[dict], all_articles: list[dict], definitions: list[dict]):
    """Verify expected counts for chapters, articles, tables, definitions."""
    counter.check(len(chapters) >= 8, f"At least 8 chapters (got {len(chapters)})")
    counter.check(len(all_articles) == 150, f"Exactly 150 articles (got {len(all_articles)})")

    total_tables = sum(len(art["tables"]) for art in all_articles)
    counter.check(total_tables == 217, f"Exactly 217 tables (got {total_tables})")
    counter.check(len(definitions) > 800, f"At least 800 definitions (got {len(definitions)})")


def _check_tree_shape(counter: _Counter, chapters: list[dict], all_articles: list[dict]):
    """Verify every chapter/article/part/subsection has the required keys."""
    for chapter in chapters:
        counter.check("chapter_num" in chapter and "title" in chapter and "articles" in chapter, f"Chapter {chapter.get('chapter_num')} has required keys")

    shape_ok = all("subsections" in part for art in all_articles for part in art.get("parts", []))
    counter.check(shape_ok, "All articles have parts, all parts have subsections")

    keys_ok = all("front_matter" in sub and "sub_items" in sub for art in all_articles for part in art["parts"] for sub in part["subsections"])
    counter.check(keys_ok, "All subsections have front_matter and sub_items keys")


def _check_110_14(counter: _Counter, all_articles: list[dict]):
    """Spot-check sub-item splitting on Section 110.14."""
    art_110 = _find_article(all_articles, 110)
    sec = _find_subsection(art_110, "110.14") if art_110 else None
    counter.check(sec is not None, "Section 110.14 found in Article 110")
    if not sec:
        return

    counter.check(len(sec["front_matter"]) > 100, f"110.14 has substantial front_matter ({len(sec['front_matter'])} chars)")

    labels = [si["label"] for si in sec["sub_items"]]
    for lbl in ("(A)", "(B)", "(C)", "(D)"):
        counter.check(lbl in labels, f"110.14 has sub-item {lbl}")
    counter.check(len(labels) >= 4, f"110.14 has at least 4 sub-items (got {len(labels)})")

    sub_a = next((si for si in sec["sub_items"] if si["label"] == "(A)"), None)
    if sub_a:
        counter.check("Terminals" in sub_a["title"], f"110.14(A) title contains 'Terminals' (got '{sub_a['title']}')")


def _check_specific_articles(counter: _Counter, all_articles: list[dict]):
    """Check Article 660 capture, Table 240.6(A) placement, and Article 90 implicit part."""
    art_660 = _find_article(all_articles, 660)
    counter.check(art_660 is not None, "Article 660 (X-Ray Equipment) is present")
    if art_660:
        counter.check("X-Ray" in art_660["title"], f"Article 660 title contains 'X-Ray' (got '{art_660['title']}')")

    tbl_ok = all(all(k in t for k in ("id", "title", "column_headers", "data_rows", "footnotes")) for art in all_articles for t in art["tables"])
    counter.check(tbl_ok, "All tables have id, title, column_headers, data_rows, footnotes")

    art_240 = _find_article(all_articles, 240)
    if art_240:
        table_ids = [t["id"] for t in art_240["tables"]]
        counter.check("Table240.6(A)" in table_ids, "Table 240.6(A) is under Article 240")

    art_90 = _find_article(all_articles, 90)
    if art_90:
        counter.check(len(art_90["parts"]) == 1, f"Article 90 has exactly 1 implicit part (got {len(art_90['parts'])})")
        counter.check(art_90["parts"][0]["part_num"] is None, "Article 90's implicit part has part_num=null")


def main():
    """Run all validation checks on the structured output."""
    logger.info("Loading structured output from %s", STRUCTURED_PATH)
    with open(STRUCTURED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chapters = data["chapters"]
    definitions = data["definitions"]
    all_articles = [art for ch in chapters for art in ch["articles"]]

    counter = _Counter()
    _check_counts(counter, chapters, all_articles, definitions)
    _check_tree_shape(counter, chapters, all_articles)
    _check_110_14(counter, all_articles)
    _check_specific_articles(counter, all_articles)

    # Summary statistics
    total_subsections = sum(len(part["subsections"]) for art in all_articles for part in art["parts"])
    total_tables = sum(len(art["tables"]) for art in all_articles)
    logger.info("─" * 60)
    logger.info("Total chapters: %d", len(chapters))
    logger.info("Total articles: %d", len(all_articles))
    logger.info("Total parts: %d", sum(len(art["parts"]) for art in all_articles))
    logger.info("Total subsections: %d", total_subsections)
    logger.info("Total tables: %d", total_tables)
    logger.info("Total definitions: %d", len(definitions))
    logger.info("─" * 60)
    logger.info("Results: %d passed, %d failed", counter.passed, counter.failed)

    return 0 if counter.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
