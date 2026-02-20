"""Unit tests for pure helper functions in the tables module.

Covers classification helpers, table boundary detection, content extraction,
interruption detection, TableStructure validation, and markdown rendering.

Skips all LLM-related functions (_init_llm, _load_cache, _save_cache,
_call_llm, format_table, run) since those require external services or
global mutable state.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
from pydantic import ValidationError

from nec_rag.data_preprocessing.tables.tables import (
    TableStructure,
    _find_post_paragraph,
    _find_pre_paragraph,
    _format_text_block,
    _is_real_table_start,
    _render_markdown,
    detect_interruption,
    extract_table_content,
    find_table_end,
    find_table_starts,
    get_table_id,
    is_continuation_marker,
    is_data_like,
    is_footnote,
    is_page_marker,
    is_section_boundary,
    is_table_title,
    resort_dict,
)


def make_paragraphs(items: list[tuple[str, int]]) -> dict[str, dict]:
    """Build a paragraph dict from a list of (content, page) tuples."""
    return {str(i): {"content": content, "page": page} for i, (content, page) in enumerate(items)}


# ===========================================================================
# is_page_marker tests
# ===========================================================================


class TestIsPageMarker:

    def test_edition_footer_forward(self):
        assert is_page_marker("2023 Edition NATIONAL ELECTRICAL CODE") is True

    def test_edition_footer_reverse(self):
        assert is_page_marker("NATIONAL ELECTRICAL CODE 2023 Edition") is True

    def test_copyright(self):
        assert is_page_marker("Copyright @NFPA. All rights reserved.") is True

    def test_copyright_space(self):
        assert is_page_marker("Copyright @ NFPA. For individual use only.") is True

    def test_watermark(self):
        assert is_page_marker("EDUFIRE.IR") is True

    def test_telegram_watermark(self):
        assert is_page_marker("Telegram: EDUFIRE.IR") is True

    def test_page_number(self):
        assert is_page_marker("70-284") is True

    def test_article_header_caps(self):
        assert is_page_marker("ARTICLE 250 GROUNDING AND BONDING") is True

    def test_section_number_alone(self):
        assert is_page_marker("400.48") is True

    def test_regular_content(self):
        assert is_page_marker("Conductors shall be copper or aluminum.") is False

    def test_table_title(self):
        assert is_page_marker("Table 310.16 Ampacities") is False

    def test_short_decimal_not_marker(self):
        """Single-digit before dot (like table data '2.79') should not match."""
        assert is_page_marker("2.79") is False


# ===========================================================================
# is_table_title tests
# ===========================================================================


class TestIsTableTitle:

    def test_basic_table_title(self):
        assert is_table_title("Table 310.16 Ampacities of Insulated Conductors") is True

    def test_table_with_parenthetical(self):
        assert is_table_title("Table 400.5(A)(1) Ampacity for Flexible Cords") is True

    def test_not_a_table_title(self):
        assert is_table_title("See Table 310.16 for details.") is False

    def test_section_number(self):
        assert is_table_title("310.16 Ampacities.") is False

    def test_bare_table_word(self):
        assert is_table_title("Table of Contents") is False


# ===========================================================================
# get_table_id tests
# ===========================================================================


class TestGetTableId:

    def test_basic_id(self):
        assert get_table_id("Table 310.16 Ampacities") == "Table310.16"

    def test_parenthetical_id(self):
        assert get_table_id("Table 400.5(A)(1) Ampacity") == "Table400.5(A)(1)"

    def test_spaces_in_parenthetical(self):
        """Spaces between parentheticals should be stripped."""
        assert get_table_id("Table 400.5(A) (1) Ampacity") == "Table400.5(A)(1)"

    def test_no_match_fallback(self):
        """When the regex doesn't match, the raw content (stripped) is returned."""
        result = get_table_id("Not a table")
        assert result == "Nota table" or result == "Not a table".replace(" ", "")


# ===========================================================================
# is_section_boundary tests
# ===========================================================================


class TestIsSectionBoundary:

    def test_part_header(self):
        assert is_section_boundary("Part III. Grounding Electrode System") is True

    def test_section_with_title(self):
        assert is_section_boundary("400.10 Uses Permitted. Flexible cords and cables shall be used only...") is True

    def test_short_section_not_boundary(self):
        """Short section references are not boundaries."""
        assert is_section_boundary("400.10 Uses.") is False

    def test_lettered_subsection_long(self):
        """A lettered subsection with substantial text is a boundary."""
        text = "(A) General Requirements. All conductors shall be installed in a manner that provides adequate clearance and protection."
        assert is_section_boundary(text) is True

    def test_lettered_subsection_short(self):
        """Short lettered references are not boundaries."""
        assert is_section_boundary("(A) General") is False

    def test_regular_text(self):
        assert is_section_boundary("Some normal paragraph text.") is False


# ===========================================================================
# is_footnote tests
# ===========================================================================


class TestIsFootnote:

    def test_digit_start(self):
        assert is_footnote("1 For copper conductors only.") is True

    def test_asterisk_start(self):
        assert is_footnote("*Based on ambient temperature of 30°C.") is True

    def test_plus_start(self):
        assert is_footnote("+See 110.14(C)(1).") is True

    def test_table_reference_words(self):
        assert is_footnote("Column A applies to conditions described in 310.15(B)(1).") is True

    def test_subheading_reference(self):
        assert is_footnote("Where the subheading D is applicable.") is True

    def test_regular_text(self):
        assert is_footnote("Conductors shall be installed in raceways.") is False

    def test_long_text_with_column_not_footnote(self):
        """Long text (>400 chars) with 'Column' should not be treated as a footnote."""
        long_text = "Column " + "x" * 400
        assert is_footnote(long_text) is False


# ===========================================================================
# is_continuation_marker tests
# ===========================================================================


class TestIsContinuationMarker:

    def test_continues(self):
        assert is_continuation_marker("(continues)") is True

    def test_continued(self):
        assert is_continuation_marker("Continued") is True

    def test_continued_parenthetical(self):
        assert is_continuation_marker("(continued)") is True

    def test_continues_with_whitespace(self):
        assert is_continuation_marker("  (continues)  ") is True

    def test_not_continuation(self):
        assert is_continuation_marker("Table 310.16") is False

    def test_partial_match(self):
        assert is_continuation_marker("Continued from previous") is False


# ===========================================================================
# is_data_like tests
# ===========================================================================


class TestIsDataLike:

    def test_dash(self):
        assert is_data_like("-") is True

    def test_double_dash(self):
        assert is_data_like("--") is True

    def test_na(self):
        assert is_data_like("N/A") is True

    def test_integer(self):
        assert is_data_like("250") is True

    def test_decimal(self):
        assert is_data_like("3.14") is True

    def test_fraction(self):
        assert is_data_like("1/0") is True

    def test_negative_integer(self):
        assert is_data_like("-40") is True

    def test_short_text(self):
        assert is_data_like("THWN-2") is True

    def test_long_text_not_data(self):
        long_text = "This is a full paragraph of text that describes grounding requirements."
        assert is_data_like(long_text) is False

    def test_yes(self):
        assert is_data_like("Yes") is True

    def test_no(self):
        assert is_data_like("No") is True


# ===========================================================================
# _is_real_table_start tests
# ===========================================================================


class TestIsRealTableStart:

    def test_real_table_with_short_followers(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("kcmil", 100),
                ("14", 100),
                ("12", 100),
                ("10", 100),
            ]
        )
        assert _is_real_table_start(paras, 0) is True

    def test_textual_reference_not_real(self):
        """A 'Table X.Y' followed by long paragraphs is a textual reference, not a real table."""
        paras = make_paragraphs(
            [
                ("Table 220.55 Demand Factors and Calculations", 100),
                ("This long paragraph discusses the requirements for calculating demand factors for household cooking equipment." * 3, 100),
                ("Another long paragraph with detailed instructions for applying the demand factors in various conditions." * 3, 100),
                ("Yet another detailed paragraph about special conditions that apply to certain equipment types." * 3, 101),
                ("Final paragraph with references to other code sections and additional requirements." * 3, 101),
                ("More text", 101),
            ]
        )
        assert _is_real_table_start(paras, 0) is False

    def test_too_few_followers(self):
        """With fewer than 2 followers, should return False."""
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("Only one follower", 100),
            ]
        )
        assert _is_real_table_start(paras, 0) is False

    def test_skips_page_markers(self):
        """Page markers between the title and table data should be skipped."""
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("70-200", 100),
                ("AWG", 101),
                ("14", 101),
                ("12", 101),
            ]
        )
        assert _is_real_table_start(paras, 0) is True


# ===========================================================================
# find_table_starts tests
# ===========================================================================


class TestFindTableStarts:

    def test_finds_single_table(self):
        paras = make_paragraphs(
            [
                ("Some content", 100),
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("12", 100),
                ("More content", 101),
            ]
        )
        starts = find_table_starts(paras)
        assert starts == [1]

    def test_finds_multiple_tables(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("12", 100),
                ("Table 310.17 Allowable", 101),
                ("Size", 101),
                ("14", 101),
                ("12", 101),
            ]
        )
        starts = find_table_starts(paras)
        assert len(starts) == 2
        assert starts[0] == 0
        assert starts[1] == 4

    def test_skips_continuation_headers(self):
        """A 'Table X.Y' + 'Continued' pair should not be counted as a new start."""
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("12", 100),
                ("Table 310.16 Ampacities", 101),
                ("Continued", 101),
                ("10", 101),
            ]
        )
        starts = find_table_starts(paras)
        assert starts == [0]

    def test_no_tables_found(self):
        paras = make_paragraphs(
            [
                ("Regular content", 100),
                ("More content", 100),
            ]
        )
        starts = find_table_starts(paras)
        assert not starts


# ===========================================================================
# find_table_end tests
# ===========================================================================


class TestFindTableEnd:

    def test_ends_at_section_boundary(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("Part III. Grounding Electrode System", 100),
                ("250.50 Grounding Electrode.", 100),
            ]
        )
        end = find_table_end(paras, 0, next_table_start=None)
        assert end == 2

    def test_ends_at_next_table_start(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("Table 310.17 Allowable", 101),
                ("Size", 101),
            ]
        )
        end = find_table_end(paras, 0, next_table_start=3)
        assert end == 2

    def test_handles_continuation_markers(self):
        """Multi-page table with continuation should be treated as a single region."""
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("(continues)", 100),
                ("Table 310.16 Ampacities", 101),
                ("Continued", 101),
                ("12", 101),
                ("10", 101),
            ]
        )
        end = find_table_end(paras, 0, next_table_start=None)
        assert end >= 7


# ===========================================================================
# extract_table_content tests
# ===========================================================================


class TestExtractTableContent:

    def test_basic_extraction(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("14", 100),
                ("12", 100),
            ]
        )
        content = extract_table_content(paras, 0, 3)
        assert content == ["Table 310.16 Ampacities", "AWG", "14", "12"]

    def test_skips_page_markers(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("AWG", 100),
                ("70-200", 100),
                ("14", 101),
            ]
        )
        content = extract_table_content(paras, 0, 3)
        assert "70-200" not in content
        assert content == ["Table 310.16 Ampacities", "AWG", "14"]

    def test_skips_continuation_noise(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("(continues)", 100),
                ("Table 310.16 Ampacities", 101),
                ("Continued", 101),
                ("12", 101),
            ]
        )
        content = extract_table_content(paras, 0, 5)
        assert "(continues)" not in content
        assert "Continued" not in content
        assert content[0] == "Table 310.16 Ampacities"
        assert "14" in content
        assert "12" in content


# ===========================================================================
# _find_pre_paragraph / _find_post_paragraph tests
# ===========================================================================


class TestFindPrePostParagraph:

    def test_find_pre_skips_page_markers(self):
        paras = make_paragraphs(
            [
                ("Real content before table", 100),
                ("70-200", 100),
                ("EDUFIRE.IR", 100),
                ("Table 310.16 Ampacities", 101),
            ]
        )
        pre = _find_pre_paragraph(paras, 3)
        assert pre == 0

    def test_find_pre_at_start(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
            ]
        )
        pre = _find_pre_paragraph(paras, 0)
        assert pre is None

    def test_find_post_skips_page_markers(self):
        paras = make_paragraphs(
            [
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("70-200", 100),
                ("EDUFIRE.IR", 101),
                ("Real content after table", 101),
            ]
        )
        post = _find_post_paragraph(paras, 1)
        assert post == 4

    def test_find_post_at_end(self):
        paras = make_paragraphs(
            [
                ("14", 100),
                ("70-200", 100),
            ]
        )
        post = _find_post_paragraph(paras, 0)
        assert post is None


# ===========================================================================
# detect_interruption tests
# ===========================================================================


class TestDetectInterruption:

    def test_detects_interrupted_sentence(self):
        """A sentence ending mid-word before a table should be detected."""
        paras = make_paragraphs(
            [
                ("Conductors shall be installed in accordance with", 100),
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("the requirements of this section.", 100),
            ]
        )
        pre, post = detect_interruption(paras, 1, 2)
        assert pre == 0
        assert post == 3

    def test_no_interruption_when_sentence_complete(self):
        paras = make_paragraphs(
            [
                ("Conductors shall be copper.", 100),
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("New section starts here.", 100),
            ]
        )
        pre, post = detect_interruption(paras, 1, 2)
        assert pre is None
        assert post is None

    def test_no_interruption_when_post_is_section_boundary(self):
        paras = make_paragraphs(
            [
                ("Conductors shall be", 100),
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("Part III. Grounding Electrode System", 100),
            ]
        )
        pre, post = detect_interruption(paras, 1, 2)
        assert pre is None
        assert post is None

    def test_continuation_with_lowercase_start(self):
        """A post-table paragraph starting with lowercase signals continuation."""
        paras = make_paragraphs(
            [
                ("The conductor shall be rated for", 100),
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("not less than 60 degrees.", 100),
            ]
        )
        pre, post = detect_interruption(paras, 1, 2)
        assert pre == 0
        assert post == 3

    def test_continuation_with_conjunction_ending(self):
        """Pre-text ending with a conjunction signals interruption."""
        paras = make_paragraphs(
            [
                ("Conductors shall be installed in accordance with the", 100),
                ("Table 310.16 Ampacities", 100),
                ("14", 100),
                ("Requirements of Article 300.", 100),
            ]
        )
        pre, post = detect_interruption(paras, 1, 2)
        assert pre == 0
        assert post == 3


# ===========================================================================
# TableStructure validation tests
# ===========================================================================


class TestTableStructure:

    def test_valid_structure(self):
        ts = TableStructure(
            title="Table 310.16",
            column_headers=["AWG", "60°C", "75°C"],
            data_rows=[["14", "15", "15"], ["12", "20", "20"]],
            footnotes=["Based on ambient 30°C."],
        )
        assert ts.title == "Table 310.16"
        assert len(ts.data_rows) == 2
        assert len(ts.footnotes) == 1

    def test_empty_data_rows(self):
        ts = TableStructure(
            title="Table 310.16",
            column_headers=["AWG", "60°C"],
            data_rows=[],
            footnotes=[],
        )
        assert ts.data_rows == []

    def test_row_width_mismatch_raises(self):
        with pytest.raises(ValidationError):
            TableStructure(
                title="Table 310.16",
                column_headers=["AWG", "60°C", "75°C"],
                data_rows=[["14", "15"]],
                footnotes=[],
            )

    def test_extra_cells_raises(self):
        with pytest.raises(ValidationError):
            TableStructure(
                title="Table 310.16",
                column_headers=["AWG", "60°C"],
                data_rows=[["14", "15", "20"]],
                footnotes=[],
            )


# ===========================================================================
# _render_markdown tests
# ===========================================================================


class TestRenderMarkdown:

    def test_basic_render(self):
        ts = TableStructure(
            title="Table 310.16 Ampacities",
            column_headers=["AWG", "60°C", "75°C"],
            data_rows=[["14", "15", "15"], ["12", "20", "20"]],
            footnotes=[],
        )
        md = _render_markdown(ts)
        assert "**Table 310.16 Ampacities**" in md
        assert "| AWG | 60°C | 75°C |" in md
        assert "| --- | --- | --- |" in md
        assert "| 14 | 15 | 15 |" in md
        assert "| 12 | 20 | 20 |" in md

    def test_render_with_footnotes(self):
        ts = TableStructure(
            title="Test Table",
            column_headers=["A", "B"],
            data_rows=[["1", "2"]],
            footnotes=["Note 1", "Note 2"],
        )
        md = _render_markdown(ts)
        assert "> Note 1" in md
        assert "> Note 2" in md

    def test_render_no_data_rows(self):
        ts = TableStructure(
            title="Empty Table",
            column_headers=["A", "B"],
            data_rows=[],
            footnotes=[],
        )
        md = _render_markdown(ts)
        assert "**Empty Table**" in md
        assert "| A | B |" in md


# ===========================================================================
# _format_text_block tests
# ===========================================================================


class TestFormatTextBlock:

    def test_basic_format(self):
        parts = ["Table 310.16 Ampacities", "AWG", "14", "12"]
        result = _format_text_block("Table 310.16 Ampacities", parts)
        assert result.startswith("**Table 310.16 Ampacities**")
        assert "AWG" in result
        assert "14" in result

    def test_title_not_duplicated(self):
        """The title from content_parts[0] should be skipped since it's used in the bold header."""
        parts = ["Table 310.16 Ampacities", "AWG", "14"]
        result = _format_text_block("Table 310.16 Ampacities", parts)
        lines = result.split("\n")
        assert lines[0] == "**Table 310.16 Ampacities**"
        # "Table 310.16 Ampacities" should NOT appear again as a data line
        non_empty = [line for line in lines if line.strip()]
        assert non_empty[1] == "AWG"


# ===========================================================================
# resort_dict tests (tables version)
# ===========================================================================


class TestTablesResortDict:

    def test_consecutive_keys(self):
        d = {"0": {"content": "a"}, "1": {"content": "b"}}
        result = resort_dict(d)
        assert list(result.keys()) == ["0", "1"]

    def test_gap_in_keys(self):
        d = {"0": {"content": "a"}, "5": {"content": "b"}, "10": {"content": "c"}}
        result = resort_dict(d)
        assert list(result.keys()) == ["0", "1", "2"]
        assert result["1"]["content"] == "b"

    def test_empty_dict(self):
        assert resort_dict({}) == {}
