"""Unit tests for the structure module.

Tests cover the pure parsing functions that transform flat cleaned paragraphs
into the nested Chapter > Article > Part > Subsection hierarchy.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from nec_rag.data_preprocessing.text_cleaning.structure import (
    _build_part_dict,
    _build_parts_list,
    _build_sub_item,
    _extract_table_refs,
    _get_chapter_for_article,
    _group_sub_items,
    _is_subsection_start,
    _normalise_table_id,
    _parse_data_rows,
    _parse_footnotes,
    _parse_pipe_row,
    _skip_blank,
    _split_sub_items,
    parse_markdown_table,
    structure_paragraphs,
)


def make_paragraphs(items: list[tuple[str, int]]) -> dict[str, dict]:
    """Build a paragraph dict from a list of (content, page) tuples."""
    return {str(i): {"content": content, "page": page} for i, (content, page) in enumerate(items)}


# ===========================================================================
# _get_chapter_for_article tests
# ===========================================================================


class TestGetChapterForArticle:

    def test_chapter_1_boundaries(self):
        assert _get_chapter_for_article(90) == (1, "General")
        assert _get_chapter_for_article(100) == (1, "General")
        assert _get_chapter_for_article(199) == (1, "General")

    def test_chapter_2(self):
        assert _get_chapter_for_article(200) == (2, "Wiring and Protection")
        assert _get_chapter_for_article(250) == (2, "Wiring and Protection")

    def test_chapter_9(self):
        assert _get_chapter_for_article(900) == (9, "Tables")

    def test_out_of_range(self):
        assert _get_chapter_for_article(50) is None
        assert _get_chapter_for_article(1000) is None

    def test_all_chapters_covered(self):
        """Spot-check one article from each chapter."""
        expected = {
            90: 1,
            200: 2,
            300: 3,
            400: 4,
            500: 5,
            600: 6,
            700: 7,
            800: 8,
            900: 9,
        }
        for article, chapter in expected.items():
            result = _get_chapter_for_article(article)
            assert result is not None
            assert result[0] == chapter


# ===========================================================================
# _normalise_table_id tests
# ===========================================================================


class TestNormaliseTableId:

    def test_standard_bold_title(self):
        result = _normalise_table_id("**Table 240.6(A) Standard Ampere Ratings**")
        assert result == "Table240.6(A)"

    def test_bold_title_with_description(self):
        result = _normalise_table_id("**Table 310.16 Ampacities of Insulated Conductors**")
        assert result == "Table310.16"

    def test_bold_title_multiple_parenthetical(self):
        result = _normalise_table_id("**Table 400.5(A)(1) Ampacity for Flexible Cords**")
        assert result == "Table400.5(A)(1)"

    def test_non_bold_fallback(self):
        """When there are no bold markers, the fallback extractor should still work."""
        result = _normalise_table_id("Table 220.55 Demand Factors")
        assert "Table220.55" in result


# ===========================================================================
# _extract_table_refs tests
# ===========================================================================


class TestExtractTableRefs:

    def test_single_ref(self):
        text = "See Table 220.55 for demand factors."
        refs = _extract_table_refs(text)
        assert refs == ["Table220.55"]

    def test_multiple_refs(self):
        text = "Refer to Table 310.16 and Table 220.55 for details."
        refs = _extract_table_refs(text)
        assert refs == ["Table220.55", "Table310.16"]

    def test_parenthetical_ref(self):
        text = "Based on Table 110.26(A)(1) requirements."
        refs = _extract_table_refs(text)
        assert refs == ["Table110.26(A)(1)"]

    def test_no_refs(self):
        text = "No table references here."
        refs = _extract_table_refs(text)
        assert refs == []

    def test_deduplication(self):
        text = "See Table 310.16 and also Table 310.16."
        refs = _extract_table_refs(text)
        assert refs == ["Table310.16"]


# ===========================================================================
# Markdown table parser helpers
# ===========================================================================


class TestSkipBlank:

    def test_skips_blanks(self):
        lines = ["", "", "content", "more"]
        assert _skip_blank(lines, 0) == 2

    def test_no_blanks(self):
        lines = ["content", "more"]
        assert _skip_blank(lines, 0) == 0

    def test_all_blanks(self):
        lines = ["", "", ""]
        assert _skip_blank(lines, 0) == 3

    def test_start_past_blanks(self):
        lines = ["", "content"]
        assert _skip_blank(lines, 1) == 1


class TestParsePipeRow:

    def test_basic_row(self):
        result = _parse_pipe_row("| AWG | 60°C | 75°C |")
        assert result == ["AWG", "60°C", "75°C"]

    def test_single_cell(self):
        result = _parse_pipe_row("| value |")
        assert result == ["value"]

    def test_cells_with_spaces(self):
        result = _parse_pipe_row("|  spaced  |  values  |")
        assert result == ["spaced", "values"]


class TestParseDataRows:

    def test_basic_data_rows(self):
        lines = ["| 14 | 15 | 15 |", "| 12 | 20 | 20 |", "", "> footnote"]
        _, rows = _parse_data_rows(lines, 0)
        assert len(rows) == 2
        assert rows[0] == ["14", "15", "15"]
        assert rows[1] == ["12", "20", "20"]

    def test_stops_at_footnote(self):
        lines = ["| 14 | 15 |", "> footnote text"]
        end_idx, rows = _parse_data_rows(lines, 0)
        assert len(rows) == 1
        assert end_idx == 1

    def test_skips_blank_lines(self):
        lines = ["| 14 | 15 |", "", "| 12 | 20 |"]
        _, rows = _parse_data_rows(lines, 0)
        assert len(rows) == 2

    def test_empty_input(self):
        _, rows = _parse_data_rows([], 0)
        assert not rows


class TestParseFootnotes:

    def test_basic_footnotes(self):
        lines = ["> Note 1", "> Note 2"]
        result = _parse_footnotes(lines, 0)
        assert result == ["Note 1", "Note 2"]

    def test_no_footnotes(self):
        lines = ["regular text"]
        result = _parse_footnotes(lines, 0)
        assert not result

    def test_mixed_content(self):
        lines = ["> Note 1", "not a footnote", "> Note 2"]
        result = _parse_footnotes(lines, 0)
        assert result == ["Note 1", "Note 2"]


# ===========================================================================
# parse_markdown_table tests
# ===========================================================================


class TestParseMarkdownTable:

    def test_full_table(self):
        md = "**Table 310.16 Ampacities**\n\n| AWG | 60°C | 75°C |\n| --- | --- | --- |\n| 14 | 15 | 15 |\n| 12 | 20 | 20 |\n\n> Based on ambient 30°C."
        result = parse_markdown_table(md)
        assert result["title"] == "Table 310.16 Ampacities"
        assert result["column_headers"] == ["AWG", "60°C", "75°C"]
        assert len(result["data_rows"]) == 2
        assert result["data_rows"][0] == ["14", "15", "15"]
        assert result["footnotes"] == ["Based on ambient 30°C."]

    def test_table_without_footnotes(self):
        md = "**Table 240.6(A) Standard Ampere Ratings**\n\n| Rating |\n| --- |\n| 15 |\n| 20 |\n"
        result = parse_markdown_table(md)
        assert result["title"] == "Table 240.6(A) Standard Ampere Ratings"
        assert not result["footnotes"]
        assert len(result["data_rows"]) == 2

    def test_table_id_is_normalised(self):
        md = "**Table 400.5(A)(1) Ampacity**\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        result = parse_markdown_table(md)
        assert "Table400.5(A)(1)" in result["id"]


# ===========================================================================
# Sub-item splitting tests
# ===========================================================================


class TestSplitSubItems:

    def test_no_sub_items(self):
        paragraphs = ["All conductors shall be copper.", "They must be properly grounded."]
        front, subs = _split_sub_items(paragraphs)
        assert front == "All conductors shall be copper.\nThey must be properly grounded."
        assert not subs

    def test_with_sub_items(self):
        paragraphs = [
            "250.50 Grounding Electrode.",
            "(A) Metal Underground Water Pipe. Shall be used.",
            "(B) Metal Frame of Building. Shall be used.",
        ]
        front, subs = _split_sub_items(paragraphs)
        assert front == "250.50 Grounding Electrode."
        assert len(subs) == 2
        assert subs[0]["label"] == "(A)"
        assert subs[1]["label"] == "(B)"

    def test_front_matter_before_sub_items(self):
        paragraphs = [
            "General requirements apply.",
            "Additional text before items.",
            "(1) First item here.",
        ]
        front, subs = _split_sub_items(paragraphs)
        assert "General requirements apply." in front
        assert "Additional text before items." in front
        assert len(subs) == 1

    def test_empty_paragraphs(self):
        front, subs = _split_sub_items([])
        assert front == ""
        assert not subs


class TestGroupSubItems:

    def test_single_item(self):
        paragraphs = ["(A) General. All conductors shall be installed properly."]
        result = _group_sub_items(paragraphs)
        assert len(result) == 1
        assert result[0]["label"] == "(A)"
        assert "General." in result[0]["title"]

    def test_multiple_items(self):
        paragraphs = [
            "(A) First Item. Description of first.",
            "(B) Second Item. Description of second.",
            "(C) Third Item. Description of third.",
        ]
        result = _group_sub_items(paragraphs)
        assert len(result) == 3
        assert result[0]["label"] == "(A)"
        assert result[1]["label"] == "(B)"
        assert result[2]["label"] == "(C)"

    def test_multi_paragraph_item(self):
        paragraphs = [
            "(A) General. Main requirements.",
            "Additional detail for item A.",
            "(B) Specific. Requirements.",
        ]
        result = _group_sub_items(paragraphs)
        assert len(result) == 2
        assert "Additional detail" in result[0]["content"]


class TestBuildSubItem:

    def test_basic(self):
        result = _build_sub_item("A", "General.", ["(A) General. Text here."])
        assert result["label"] == "(A)"
        assert result["title"] == "General."
        assert result["content"] == "(A) General. Text here."

    def test_multi_content(self):
        result = _build_sub_item("1", "First.", ["(1) First. Part one.", "Continuation text."])
        assert result["content"] == "(1) First. Part one.\nContinuation text."


# ===========================================================================
# _is_subsection_start tests
# ===========================================================================


class TestIsSubsectionStart:

    def test_standard_subsection(self):
        assert _is_subsection_start("250.50 Grounding Electrode.") is True

    def test_subsection_with_long_title(self):
        assert _is_subsection_start("110.26 Spaces About Electrical Equipment.") is True

    def test_table_title_not_subsection(self):
        assert _is_subsection_start("Table 310.16 Ampacities of Insulated Conductors") is False

    def test_markdown_table_not_subsection(self):
        assert _is_subsection_start("**Table 310.16 Ampacities**") is False

    def test_single_digit_before_dot(self):
        """'1.0 m' has only 1 digit before dot -- should not match (requires 2+)."""
        assert _is_subsection_start("1.0 m clearance required") is False

    def test_regular_text(self):
        assert _is_subsection_start("Conductors shall be properly installed.") is False


# ===========================================================================
# _build_part_dict / _build_parts_list tests
# ===========================================================================


class TestBuildPartDict:

    def test_basic(self):
        subs = [
            {"id": "250.50", "title": "Grounding", "referenced_tables": ["Table250.52(A)(1)"]},
            {"id": "250.52", "title": "Electrodes", "referenced_tables": []},
        ]
        result = _build_part_dict("III", "Grounding Electrode System", subs)
        assert result["part_num"] == "III"
        assert result["title"] == "Grounding Electrode System"
        assert len(result["subsections"]) == 2
        assert "Table250.52(A)(1)" in result["referenced_tables"]

    def test_implicit_part(self):
        subs = [{"id": "90.1", "title": "Scope", "referenced_tables": []}]
        result = _build_part_dict(None, None, subs)
        assert result["part_num"] is None
        assert result["title"] is None
        assert len(result["subsections"]) == 1

    def test_empty_subsections(self):
        result = _build_part_dict("I", "General", [])
        assert result["subsections"] == []
        assert result["referenced_tables"] == []


class TestBuildPartsList:

    def test_no_explicit_parts(self):
        """Article with no Part headers gets a single implicit part."""
        article = {
            "parts_ordered": [],
            "parts_subsections": {None: [{"id": "90.1", "title": "Scope", "referenced_tables": []}]},
        }
        parts = _build_parts_list(article)
        assert len(parts) == 1
        assert parts[0]["part_num"] is None

    def test_explicit_parts(self):
        article = {
            "parts_ordered": [("I", "General"), ("II", "Specific")],
            "parts_subsections": {
                "I": [{"id": "250.1", "title": "Scope", "referenced_tables": []}],
                "II": [{"id": "250.20", "title": "Requirements", "referenced_tables": []}],
            },
        }
        parts = _build_parts_list(article)
        assert len(parts) == 2
        assert parts[0]["part_num"] == "I"
        assert parts[1]["part_num"] == "II"

    def test_leading_subsections_before_first_part(self):
        """Subsections filed under None that appear before any Part header get an implicit leading part."""
        article = {
            "parts_ordered": [("I", "Main")],
            "parts_subsections": {
                None: [{"id": "250.1", "title": "Scope", "referenced_tables": []}],
                "I": [{"id": "250.20", "title": "General", "referenced_tables": []}],
            },
        }
        parts = _build_parts_list(article)
        assert len(parts) == 2
        assert parts[0]["part_num"] is None
        assert parts[1]["part_num"] == "I"


# ===========================================================================
# structure_paragraphs end-to-end tests
# ===========================================================================


class TestStructureParagraphs:

    def test_single_article_with_subsection(self):
        """A minimal article with one subsection should produce one chapter with one article."""
        paras = make_paragraphs(
            [
                ("ARTICLE 110 Requirements for Electrical Installations", 30),
                ("110.1 Scope. This article covers general requirements.", 30),
            ]
        )
        result = structure_paragraphs(paras)
        assert len(result["chapters"]) == 1
        chapter = result["chapters"][0]
        assert chapter["chapter_num"] == 1
        assert len(chapter["articles"]) == 1
        article = chapter["articles"][0]
        assert article["article_num"] == 110
        assert len(article["parts"]) >= 1

    def test_multiple_articles_in_same_chapter(self):
        paras = make_paragraphs(
            [
                ("ARTICLE 110 Requirements for Electrical Installations", 30),
                ("110.1 Scope. This article covers general requirements.", 30),
                ("ARTICLE 120 General Wiring", 50),
                ("120.1 Scope. Wiring requirements.", 50),
            ]
        )
        result = structure_paragraphs(paras)
        assert len(result["chapters"]) == 1
        assert len(result["chapters"][0]["articles"]) == 2

    def test_articles_across_chapters(self):
        paras = make_paragraphs(
            [
                ("ARTICLE 110 Requirements for Electrical Installations", 30),
                ("110.1 Scope. General requirements.", 30),
                ("ARTICLE 210 Branch Circuits", 60),
                ("210.1 Scope. Branch circuit requirements.", 60),
            ]
        )
        result = structure_paragraphs(paras)
        assert len(result["chapters"]) == 2
        assert result["chapters"][0]["chapter_num"] == 1
        assert result["chapters"][1]["chapter_num"] == 2

    def test_definitions_collected(self):
        """Article 100 content should be collected as definitions."""
        paras = make_paragraphs(
            [
                ("ARTICLE 100 Definitions", 30),
                ("Accessible (as applied to equipment). Admitting close approach.", 30),
                ("Ampacity. The maximum current a conductor can carry.", 31),
                ("ARTICLE 110 Requirements for Electrical Installations", 40),
                ("110.1 Scope. General requirements.", 40),
            ]
        )
        result = structure_paragraphs(paras)
        assert len(result["definitions"]) == 2
        assert result["definitions"][0]["term"] == "Accessible (as applied to equipment)"

    def test_subsection_with_sub_items(self):
        paras = make_paragraphs(
            [
                ("ARTICLE 250 Grounding and Bonding", 100),
                ("250.50 Grounding Electrode. The following shall be used.", 100),
                ("(A) Metal Underground Water Pipe. Shall be in contact with earth.", 100),
                ("(B) Metal Frame of Building. Where connected to ground.", 101),
            ]
        )
        result = structure_paragraphs(paras)
        article = result["chapters"][0]["articles"][0]
        subsection = article["parts"][0]["subsections"][0]
        assert len(subsection["sub_items"]) == 2
        assert subsection["sub_items"][0]["label"] == "(A)"

    def test_part_headers(self):
        paras = make_paragraphs(
            [
                ("ARTICLE 250 Grounding and Bonding", 100),
                ("Part I. General", 100),
                ("250.1 Scope. This article covers.", 100),
                ("Part II. System Grounding", 105),
                ("250.20 Alternating-Current Systems. Shall be grounded.", 105),
            ]
        )
        result = structure_paragraphs(paras)
        article = result["chapters"][0]["articles"][0]
        assert len(article["parts"]) == 2
        assert article["parts"][0]["part_num"] == "I"
        assert article["parts"][1]["part_num"] == "II"

    def test_table_attached_to_article(self):
        md_table = "**Table 250.66 Grounding Electrode Conductor**\n\n| Size | Conductor |\n| --- | --- |\n| 2 | 8 |\n"
        paras = make_paragraphs(
            [
                ("ARTICLE 250 Grounding and Bonding", 100),
                ("250.1 Scope. This article covers.", 100),
                (md_table, 102),
            ]
        )
        result = structure_paragraphs(paras)
        article = result["chapters"][0]["articles"][0]
        assert len(article["tables"]) == 1
        assert "Table250.66" in article["tables"][0]["id"]

    def test_preamble_skipped(self):
        """Content before the first ARTICLE should be ignored."""
        paras = make_paragraphs(
            [
                ("This is preamble text.", 26),
                ("More preamble.", 27),
                ("ARTICLE 110 Requirements for Electrical Installations", 30),
                ("110.1 Scope. General requirements.", 30),
            ]
        )
        result = structure_paragraphs(paras)
        assert len(result["chapters"]) == 1
        assert result["chapters"][0]["articles"][0]["article_num"] == 110

    def test_empty_input(self):
        result = structure_paragraphs({})
        assert not result["chapters"]
        assert not result["definitions"]

    def test_referenced_tables_propagated(self):
        """Table references in subsection text should propagate up to part, article, and chapter."""
        paras = make_paragraphs(
            [
                ("ARTICLE 250 Grounding and Bonding", 100),
                ("250.50 Grounding Electrode. See Table 250.66 for sizing.", 100),
            ]
        )
        result = structure_paragraphs(paras)
        chapter = result["chapters"][0]
        article = chapter["articles"][0]
        part = article["parts"][0]
        subsection = part["subsections"][0]

        assert "Table250.66" in subsection["referenced_tables"]
        assert "Table250.66" in part["referenced_tables"]
        assert "Table250.66" in article["referenced_tables"]
        assert "Table250.66" in chapter["referenced_tables"]
