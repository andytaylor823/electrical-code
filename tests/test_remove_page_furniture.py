"""Unit tests for remove_page_furniture module.

Tests cover:
  - is_page_furniture: classification of page headers, footers, watermarks, etc.
  - run: full removal pipeline with re-indexing
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from nec_rag.data_preprocessing.text_cleaning.remove_page_furniture import is_page_furniture, run


def make_paragraphs(items: list[tuple[str, int]]) -> dict[str, dict]:
    """Build a paragraph dict from a list of (content, page) tuples."""
    return {str(i): {"content": content, "page": page} for i, (content, page) in enumerate(items)}


def contents(paragraphs: dict[str, dict]) -> list[str]:
    """Extract content strings in key order."""
    return [paragraphs[str(i)]["content"] for i in range(len(paragraphs))]


# ===========================================================================
# is_page_furniture tests
# ===========================================================================


class TestIsPageFurniturePositive:
    """Tests for content that IS page furniture (should return True)."""

    def test_edition_marker_forward(self):
        assert is_page_furniture("2023 Edition NATIONAL ELECTRICAL CODE") is True

    def test_edition_marker_reverse(self):
        assert is_page_furniture("NATIONAL ELECTRICAL CODE 2023 Edition") is True

    def test_watermark_edufire(self):
        assert is_page_furniture("EDUFIRE.IR") is True

    def test_watermark_telegram(self):
        assert is_page_furniture("Telegram: EDUFIRE.IR") is True

    def test_copyright_at_nfpa(self):
        assert is_page_furniture("Copyright @NFPA. All rights reserved.") is True

    def test_copyright_at_space_nfpa(self):
        assert is_page_furniture("Copyright @ NFPA. All rights reserved.") is True

    def test_page_number_short(self):
        assert is_page_furniture("70-23") is True

    def test_page_number_long(self):
        assert is_page_furniture("70-284") is True

    def test_article_header_caps(self):
        assert is_page_furniture("ARTICLE 100 - DEFINITIONS") is True

    def test_article_header_caps_no_dash(self):
        assert is_page_furniture("ARTICLE 250 GROUNDING AND BONDING") is True

    def test_chapter_caps_standalone(self):
        assert is_page_furniture("CHAPTER 1") is True

    def test_chapter_caps_two_digit(self):
        assert is_page_furniture("CHAPTER 9") is True

    def test_chapter_title_mixed_case(self):
        assert is_page_furniture("Chapter 1 General") is True

    def test_chapter_title_longer(self):
        assert is_page_furniture("Chapter 2 Wiring and Protection") is True

    def test_bare_section_number(self):
        assert is_page_furniture("110.26") is True

    def test_bare_section_number_long(self):
        assert is_page_furniture("250.50") is True


class TestIsPageFurnitureNegative:
    """Tests for content that is NOT page furniture (should return False)."""

    def test_page_number_with_text_not_furniture(self):
        """'70-284 Some text' should NOT match the page-number regex (anchored)."""
        assert is_page_furniture("70-284 Some text") is False

    def test_article_mixed_case_not_furniture(self):
        """Mixed-case article text is real content, not a page header."""
        assert is_page_furniture("Article 100 defines key terms used throughout the code.") is False

    def test_chapter_caps_with_title_not_standalone(self):
        """'CHAPTER 1 GENERAL' has text after the number, so it does NOT match the standalone regex."""
        assert is_page_furniture("CHAPTER 1 GENERAL") is False

    def test_chapter_lowercase_start_not_furniture(self):
        """'chapter 1 general' starts with lowercase, should NOT match."""
        assert is_page_furniture("chapter 1 general") is False

    def test_section_number_with_text_not_furniture(self):
        """'110.26 Spaces About Electrical Equipment.' is a subsection title, not furniture."""
        assert is_page_furniture("110.26 Spaces About Electrical Equipment.") is False

    def test_short_decimal_not_furniture(self):
        """'1.0' has only 1 digit before the dot, so it should NOT match (requires 2+)."""
        assert is_page_furniture("1.0") is False

    def test_regular_paragraph(self):
        assert is_page_furniture("Conductors shall be installed in accordance with Table 310.16.") is False

    def test_table_title(self):
        assert is_page_furniture("Table 310.16 Ampacities of Insulated Conductors") is False

    def test_informational_note(self):
        assert is_page_furniture("Informational Note: See 300.5 for information on underground installations.") is False

    def test_sub_item(self):
        assert is_page_furniture("(A) Practical Safeguarding.") is False

    def test_empty_string(self):
        assert is_page_furniture("") is False


# ===========================================================================
# run tests
# ===========================================================================


class TestRemovePageFurnitureRun:
    """Tests for the full removal pipeline."""

    def test_removes_edition_markers(self):
        paras = make_paragraphs(
            [
                ("Real content", 100),
                ("2023 Edition NATIONAL ELECTRICAL CODE", 100),
                ("More content", 101),
            ]
        )
        result = run(paras)
        assert len(result) == 2
        assert contents(result) == ["Real content", "More content"]

    def test_removes_watermarks(self):
        paras = make_paragraphs(
            [
                ("EDUFIRE.IR", 100),
                ("Real content", 100),
                ("Telegram: EDUFIRE.IR", 101),
            ]
        )
        result = run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "Real content"

    def test_removes_page_numbers(self):
        paras = make_paragraphs(
            [
                ("70-50", 100),
                ("Real content", 100),
                ("70-284", 101),
            ]
        )
        result = run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "Real content"

    def test_removes_copyright(self):
        paras = make_paragraphs(
            [
                ("Copyright @NFPA. Not to be reproduced.", 100),
                ("Real content", 100),
            ]
        )
        result = run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "Real content"

    def test_removes_article_headers(self):
        paras = make_paragraphs(
            [
                ("ARTICLE 250 GROUNDING AND BONDING", 100),
                ("250.1 Scope.", 100),
            ]
        )
        result = run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "250.1 Scope."

    def test_removes_bare_section_numbers(self):
        paras = make_paragraphs(
            [
                ("110.26", 100),
                ("Real content about clearances", 100),
            ]
        )
        result = run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "Real content about clearances"

    def test_reindexes_after_removal(self):
        paras = make_paragraphs(
            [
                ("EDUFIRE.IR", 100),
                ("First real", 100),
                ("70-50", 101),
                ("Second real", 101),
                ("CHAPTER 3", 102),
                ("Third real", 102),
            ]
        )
        result = run(paras)
        assert list(result.keys()) == ["0", "1", "2"]
        assert contents(result) == ["First real", "Second real", "Third real"]

    def test_preserves_page_numbers_in_output(self):
        paras = make_paragraphs(
            [
                ("EDUFIRE.IR", 100),
                ("Real content", 200),
            ]
        )
        result = run(paras)
        assert result["0"]["page"] == 200

    def test_empty_input(self):
        result = run({})
        assert result == {}

    def test_all_furniture_returns_empty(self):
        paras = make_paragraphs(
            [
                ("70-50", 100),
                ("EDUFIRE.IR", 100),
                ("2023 Edition NATIONAL ELECTRICAL CODE", 101),
            ]
        )
        result = run(paras)
        assert len(result) == 0

    def test_no_furniture_passthrough(self):
        paras = make_paragraphs(
            [
                ("Real content one", 100),
                ("Real content two", 101),
            ]
        )
        result = run(paras)
        assert len(result) == 2
        assert contents(result) == ["Real content one", "Real content two"]
