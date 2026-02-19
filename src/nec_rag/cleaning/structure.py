"""Parse cleaned NEC paragraphs into a nested hierarchical document tree.

Takes the flat paragraph dict produced by the cleaning pipeline and organises
it into a four-level tree that mirrors the NEC's own structure:

    Chapter
      └─ Article
           ├─ Tables  (belong to the article, not to any subsection)
           └─ Part    (implicit if the article has no explicit parts)
                └─ Subsection  (e.g. 250.50)
                     ├─ front_matter  (text before the first (A)/(B)/(C))
                     └─ sub_items     (each lettered/numbered item)

Definitions (Article 100) are collected into a separate top-level list.

Output: a single JSON file with two top-level keys:
    "chapters"    - nested list of chapter > article > part > subsection
    "definitions" - flat list of definition dicts

Usage:
    python -m nec_rag.cleaning.structure
"""

import json
import logging
import re
from pathlib import Path

from nec_rag.cleaning import remove_page_furniture

logger = logging.getLogger(__name__)

# ── Project root ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# ── Regex patterns for structural boundaries ─────────────────────────────────

# "ARTICLE 90 Introduction" or "ARTICLE 660 X-Ray Equipment"
ARTICLE_TITLE_RE = re.compile(r"^ARTICLE (\d+) ([A-Z].+)")

# "Part III. Grounding Electrode System" -- may have trailing section text
PART_HEADER_RE = re.compile(r"^Part ([IVX]+)\.\s*(.*)")

# "250.50 Grounding Electrode." -- subsection boundary.
# NEC article numbers are always 2+ digits, so require >=2 digits before the
# dot to avoid false-positives on measurements like "1.0 m" or "3.05".
SECTION_RE = re.compile(r"^(\d{2,}\.\d+)\s+(.+)")

# Markdown table paragraph as rendered by the cleaning pipeline:
# "**Table 400.5(A)(1) Ampacity ...**" (bold title line)
MD_TABLE_RE = re.compile(r"^\*\*Table \d+")

# Table-ID extractor from the bold title line
TABLE_ID_FROM_BOLD_RE = re.compile(r"\*\*(Table \d+\.\d+(?:\s*\([^)]*\))*)\s*(.*?)\*\*")

# Sub-item marker: "(A) Practical Safeguarding" or "(1) Public and private..."
SUB_ITEM_RE = re.compile(r"^\(([A-Za-z0-9]+)\)\s*(.*)")

# Informational Note
INFO_NOTE_RE = re.compile(r"^Informational Note")

# ── Chapter-to-article mapping ───────────────────────────────────────────────

CHAPTER_MAP = {
    1: {"title": "General", "min_article": 90, "max_article": 199},
    2: {"title": "Wiring and Protection", "min_article": 200, "max_article": 299},
    3: {"title": "Wiring Methods and Materials", "min_article": 300, "max_article": 399},
    4: {"title": "Equipment for General Use", "min_article": 400, "max_article": 499},
    5: {"title": "Special Occupancies", "min_article": 500, "max_article": 599},
    6: {"title": "Special Equipment", "min_article": 600, "max_article": 699},
    7: {"title": "Special Conditions", "min_article": 700, "max_article": 799},
    8: {"title": "Communications Systems", "min_article": 800, "max_article": 899},
    9: {"title": "Tables", "min_article": 900, "max_article": 999},
}


def _get_chapter_for_article(article_num: int) -> tuple[int, str] | None:
    """Return (chapter_num, chapter_title) for a given article number."""
    for chapter_num, info in CHAPTER_MAP.items():
        if info["min_article"] <= article_num <= info["max_article"]:
            return chapter_num, info["title"]
    return None


def _normalise_table_id(bold_title: str) -> str:
    """Extract and normalise a table ID from the bold markdown title line.

    '**Table 240.6(A) Standard Ampere Ratings...**' -> 'Table240.6(A)'
    """
    match = TABLE_ID_FROM_BOLD_RE.match(bold_title)
    if match:
        return match.group(1).replace(" ", "")
    stripped = bold_title.replace("**", "").strip()
    token_match = re.match(r"(Table \d+\.\d+(?:\s*\([^)]*\))*)", stripped)
    if token_match:
        return token_match.group(1).replace(" ", "")
    return stripped


# ── Markdown table parser ────────────────────────────────────────────────────

SEPARATOR_RE = re.compile(r"^\|[\s\-|]+\|$")


def parse_markdown_table(paragraph: str) -> dict:
    """Parse a markdown-formatted table paragraph into structured form.

    Expected format (from the cleaning pipeline):
        **Title**

        | col1 | col2 |
        | --- | --- |
        | val1 | val2 |

        > footnote text
    """
    lines = paragraph.split("\n")
    title, column_headers, data_rows, footnotes = "", [], [], []

    idx = _skip_blank(lines, 0)
    if idx < len(lines):
        title = lines[idx].replace("**", "").strip()
        idx += 1

    idx = _skip_blank(lines, idx)

    # Parse header row
    if idx < len(lines) and "|" in lines[idx]:
        column_headers = _parse_pipe_row(lines[idx])
        idx += 1

    # Skip separator row (| --- | --- |)
    if idx < len(lines) and SEPARATOR_RE.match(lines[idx].strip()):
        idx += 1

    # Parse data rows
    idx, data_rows = _parse_data_rows(lines, idx)

    # Collect footnotes
    footnotes = _parse_footnotes(lines, idx)

    return {
        "id": _normalise_table_id(paragraph),
        "title": title,
        "column_headers": column_headers,
        "data_rows": data_rows,
        "footnotes": footnotes,
    }


def _skip_blank(lines: list[str], idx: int) -> int:
    """Advance idx past any blank lines."""
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _parse_pipe_row(line: str) -> list[str]:
    """Split a pipe-delimited row into cell strings."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_data_rows(lines: list[str], idx: int) -> tuple[int, list[list[str]]]:
    """Parse table data rows, stopping at footnotes or end of lines."""
    rows = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith(">"):
            break
        if "|" in line:
            rows.append(_parse_pipe_row(line))
        idx += 1
    return idx, rows


def _parse_footnotes(lines: list[str], idx: int) -> list[str]:
    """Collect blockquote footnotes from the remaining lines."""
    footnotes = []
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith(">"):
            footnotes.append(line.lstrip(">").strip())
        idx += 1
    return footnotes


# ── Sub-item splitter ────────────────────────────────────────────────────────


def _split_sub_items(paragraphs: list[str]) -> tuple[str, list[dict]]:
    """Split a subsection's paragraph list into front_matter and sub_items.

    Returns (front_matter_text, list_of_sub_item_dicts).
    """
    first_sub_idx = next((i for i, p in enumerate(paragraphs) if SUB_ITEM_RE.match(p)), None)

    if first_sub_idx is None:
        return "\n".join(paragraphs), []

    front_matter = "\n".join(paragraphs[:first_sub_idx])
    sub_items = _group_sub_items(paragraphs[first_sub_idx:])
    return front_matter, sub_items


def _group_sub_items(paragraphs: list[str]) -> list[dict]:
    """Group paragraphs starting from the first sub-item marker into sub-item dicts."""
    sub_items: list[dict] = []
    current_label = None
    current_title = None
    current_content: list[str] = []

    for para in paragraphs:
        match = SUB_ITEM_RE.match(para)
        if match:
            if current_label is not None:
                sub_items.append(_build_sub_item(current_label, current_title, current_content))
            current_label = match.group(1)
            raw_title = match.group(2).strip()
            dot_pos = raw_title.find(".")
            current_title = raw_title[: dot_pos + 1].strip() if dot_pos != -1 else raw_title
            current_content = [para]
        else:
            current_content.append(para)

    if current_label is not None:
        sub_items.append(_build_sub_item(current_label, current_title, current_content))

    return sub_items


def _build_sub_item(label: str, title: str, content_parts: list[str]) -> dict:
    """Construct a sub-item dict."""
    return {
        "label": f"({label})",
        "title": title,
        "content": "\n".join(content_parts),
    }


# ── Parser state container ───────────────────────────────────────────────────


class _ParserState:  # pylint: disable=too-many-instance-attributes
    """Mutable state bag for the paragraph parser."""

    def __init__(self):
        self.articles: dict[int, dict] = {}
        self.definitions: list[dict] = []

        # Current hierarchy context
        self.article_num: int | None = None
        self.article_title: str | None = None
        self.part_num: str | None = None
        self.part_title: str | None = None
        self.subsection_id: str | None = None
        self.subsection_title: str | None = None
        self.subsection_page: int | None = None
        self.subsection_paragraphs: list[str] = []

        # Phase tracking
        self.in_definitions = False
        self.in_main = False

        # Definition accumulation
        self.def_term: str | None = None
        self.def_content: list[str] = []
        self.def_page: int | None = None

    def ensure_article(self, art_num: int, art_title: str):
        """Create the article bucket if it doesn't exist yet."""
        if art_num not in self.articles:
            self.articles[art_num] = {
                "article_num": art_num,
                "title": art_title,
                "tables": [],
                "parts_ordered": [],
                "parts_subsections": {},
            }

    def flush_subsection(self):
        """Save the accumulated subsection to the current article/part."""
        if self.subsection_id is None or not self.subsection_paragraphs:
            self.subsection_paragraphs = []
            return

        front_matter, sub_items = _split_sub_items(self.subsection_paragraphs)
        subsection_dict = {
            "id": self.subsection_id,
            "title": self.subsection_title,
            "page": self.subsection_page,
            "front_matter": front_matter,
            "sub_items": sub_items,
        }

        if self.article_num is not None:
            self.ensure_article(self.article_num, self.article_title or "")
            art = self.articles[self.article_num]
            art["parts_subsections"].setdefault(self.part_num, []).append(subsection_dict)

        self.subsection_paragraphs = []

    def flush_definition(self):
        """Save the accumulated definition."""
        if self.def_term and self.def_content:
            self.definitions.append(
                {
                    "term": self.def_term,
                    "page": self.def_page,
                    "content": "\n".join(self.def_content),
                }
            )
        self.def_term = None
        self.def_content = []
        self.def_page = None


# ── Paragraph handlers ───────────────────────────────────────────────────────


def _handle_article(state: _ParserState, content: str):
    """Process an ARTICLE title paragraph and update state phases."""
    match = ARTICLE_TITLE_RE.match(content)
    state.flush_subsection()
    if state.in_definitions:
        state.flush_definition()

    state.article_num = int(match.group(1))
    state.article_title = match.group(2).strip()
    state.part_num = None
    state.part_title = None
    state.subsection_id = None
    state.ensure_article(state.article_num, state.article_title)
    logger.debug("Entered Article %d: %s", state.article_num, state.article_title)

    # Article 100 starts the definitions zone; other articles are main content
    if state.article_num == 100:
        state.in_definitions = True
        state.in_main = False
    else:
        state.in_definitions = False
        state.in_main = True


def _handle_definition(state: _ParserState, content: str, page: int):
    """Process a paragraph in the definitions zone."""
    # Informational notes and sub-items belong to the current definition
    if INFO_NOTE_RE.match(content) or content.startswith("("):
        if state.def_content:
            state.def_content.append(content)
        return

    # Bracketed references like "[101:3.3.198.1] (517) (CMP-15)"
    if content.startswith("[") or (len(content) < 30 and content.startswith("(")):
        if state.def_content:
            state.def_content.append(content)
        return

    # New definition term
    state.flush_definition()
    term_match = re.match(r"^([^.]+)\.", content)
    state.def_term = term_match.group(1).strip() if term_match else content[:60]
    state.def_content = [content]
    state.def_page = page


def _handle_part(state: _ParserState, content: str, page: int):
    """Process a Part header and register it on the current article."""
    match = PART_HEADER_RE.match(content)
    state.flush_subsection()
    state.part_num = match.group(1)
    state.part_title = match.group(2).strip() if match.group(2) else None

    if state.article_num is not None:
        state.ensure_article(state.article_num, state.article_title or "")
        part_entry = (state.part_num, state.part_title)
        if part_entry not in state.articles[state.article_num]["parts_ordered"]:
            state.articles[state.article_num]["parts_ordered"].append(part_entry)

    # Some Part headers embed a section, e.g. "Part IV. ... 110.51 General."
    section_in_part = SECTION_RE.search(match.group(2)) if match.group(2) else None
    if section_in_part:
        state.subsection_id = section_in_part.group(1)
        state.subsection_title = section_in_part.group(2).strip()
        state.subsection_page = page
        state.subsection_paragraphs = [content]

    logger.debug("Entered Part %s: %s", state.part_num, state.part_title)


def _handle_subsection(state: _ParserState, content: str, page: int):
    """Process a subsection boundary (e.g. '250.50 Grounding Electrode.')."""
    match = SECTION_RE.match(content)
    state.flush_subsection()
    state.subsection_id = match.group(1)
    state.subsection_title = match.group(2).strip()
    state.subsection_page = page
    state.subsection_paragraphs = [content]


def _handle_table(state: _ParserState, content: str):
    """Parse a markdown table paragraph and attach it to the current article."""
    table_data = parse_markdown_table(content)
    if state.article_num is not None:
        state.ensure_article(state.article_num, state.article_title or "")
        state.articles[state.article_num]["tables"].append(table_data)
    else:
        logger.warning("Table found outside of any article context: %s", table_data.get("id", "unknown"))


def _is_subsection_start(content: str) -> bool:
    """Return True if this paragraph starts a new numbered subsection."""
    return bool(SECTION_RE.match(content)) and not content.startswith("Table ") and not MD_TABLE_RE.match(content)


# ── Main structuring logic ───────────────────────────────────────────────────


def structure_paragraphs(paragraphs: dict[str, dict]) -> dict:
    """Parse flat paragraphs into a nested Chapter > Article > Part > Subsection tree.

    Returns a dict with:
        "chapters"    - list of nested chapter dicts
        "definitions" - list of definition dicts
    """
    state = _ParserState()
    n = len(paragraphs)

    for i in range(n):
        content = paragraphs[str(i)]["content"]
        page = paragraphs[str(i)]["page"]

        # Article titles always take priority and drive phase transitions
        if ARTICLE_TITLE_RE.match(content):
            _handle_article(state, content)
            continue

        # Skip preamble (content before the first article)
        if not state.in_main and not state.in_definitions:
            continue

        # Definitions zone (Article 100)
        if state.in_definitions and not state.in_main:
            _handle_definition(state, content, page)
            continue

        # Main content zone -- check structural markers in priority order
        if not state.in_main:
            continue

        if PART_HEADER_RE.match(content):
            _handle_part(state, content, page)
        elif _is_subsection_start(content):
            _handle_subsection(state, content, page)
        elif MD_TABLE_RE.match(content):
            _handle_table(state, content)
        elif state.subsection_id is not None:
            state.subsection_paragraphs.append(content)

    # Flush final subsection and definition
    state.flush_subsection()
    state.flush_definition()

    # Assemble the nested tree
    chapters = _assemble_chapters(state.articles)

    total_subsections = sum(len(subs) for art in state.articles.values() for subs in art["parts_subsections"].values())
    total_tables = sum(len(art["tables"]) for art in state.articles.values())
    logger.info(
        "Structured %d definitions, %d articles, %d subsections, %d tables across %d chapters",
        len(state.definitions),
        len(state.articles),
        total_subsections,
        total_tables,
        len(chapters),
    )

    return {
        "chapters": chapters,
        "definitions": state.definitions,
    }


# ── Tree assembly ────────────────────────────────────────────────────────────


def _assemble_chapters(articles: dict[int, dict]) -> list[dict]:
    """Group article dicts into chapter dicts with nested parts and subsections."""
    chapter_articles: dict[int, list[dict]] = {}

    for art_num in sorted(articles.keys()):
        art = articles[art_num]
        chapter_info = _get_chapter_for_article(art_num)
        if chapter_info is None:
            logger.warning("Article %d does not map to any chapter", art_num)
            continue

        ch_num, _ = chapter_info
        chapter_articles.setdefault(ch_num, []).append(
            {
                "article_num": art["article_num"],
                "title": art["title"],
                "tables": art["tables"],
                "parts": _build_parts_list(art),
            }
        )

    return [{"chapter_num": ch_num, "title": CHAPTER_MAP[ch_num]["title"], "articles": chapter_articles[ch_num]} for ch_num in sorted(chapter_articles.keys())]


def _build_parts_list(article: dict) -> list[dict]:
    """Build the parts list for a single article.

    Articles with explicit Part headers get one entry per part.
    Articles without explicit parts get a single implicit part (part_num=null).
    """
    parts_ordered = article["parts_ordered"]
    parts_subsections = article["parts_subsections"]

    if not parts_ordered:
        return [{"part_num": None, "title": None, "subsections": parts_subsections.get(None, [])}]

    parts = [{"part_num": pn, "title": pt, "subsections": parts_subsections.get(pn, [])} for pn, pt in parts_ordered]

    # Subsections filed under None appeared before the first Part header
    leading = parts_subsections.get(None, [])
    if leading:
        parts.insert(0, {"part_num": None, "title": None, "subsections": leading})

    return parts


# ── CLI entry point ──────────────────────────────────────────────────────────


def run_from_clean_json(input_path: Path | None = None, output_path: Path | None = None) -> dict:
    """Load cleaned paragraphs, structure them, and write the output JSON."""
    if input_path is None:
        input_path = ROOT / "data" / "intermediate" / "NFPA 70 NEC 2023_clean.json"
    if output_path is None:
        output_path = ROOT / "data" / "prepared" / "NFPA 70 NEC 2023_structured.json"

    logger.info("Loading cleaned paragraphs from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as fopen:
        paragraphs = json.load(fopen)
    logger.info("Loaded %d paragraphs", len(paragraphs))

    # Apply page-furniture removal if not already done
    has_furniture = any(remove_page_furniture.is_page_furniture(paragraphs[str(i)]["content"]) for i in range(min(100, len(paragraphs))))
    if has_furniture:
        logger.info("Detected page furniture in input -- applying removal first")
        paragraphs = remove_page_furniture.run(paragraphs)

    result = structure_paragraphs(paragraphs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fopen:
        json.dump(result, fopen, indent=2)
    logger.info(
        "Wrote structured output to %s (%d chapters, %d definitions)",
        output_path,
        len(result["chapters"]),
        len(result["definitions"]),
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_from_clean_json()
