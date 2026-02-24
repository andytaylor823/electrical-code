"""Unit tests for nec_lookup and browse_nec_structure tools.

Tests exercise the real structured JSON so that assertions verify actual
NEC content.  The rag_search and explain_image tools are excluded because
they require embedding models / LLM calls respectively.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from nec_rag.agent.tools import browse_nec_structure, nec_lookup

# ===========================================================================
# Section lookup tests
# ===========================================================================


class TestSectionLookup:
    """Test nec_lookup when called with section_ids only."""

    def test_known_section_returns_header_and_content(self):
        """Section 90.1 (Scope) should return a header and front-matter text."""
        result = nec_lookup.invoke({"section_ids": ["90.1"]})
        assert "[Section 90.1, Article 90, page 26]" in result
        assert "Scope" in result

    def test_section_content_matches_expected_text(self):
        """Section 250.50 (page 148) should contain grounding electrode language."""
        result = nec_lookup.invoke({"section_ids": ["250.50"]})
        assert "[Section 250.50, Article 250, page 148]" in result
        assert "grounding electrode" in result.lower()

    def test_section_with_sub_items_includes_all(self):
        """Section 250.52 has sub-items (A), (1), (2), etc.  All should appear."""
        result = nec_lookup.invoke({"section_ids": ["250.52"]})
        assert "(A)" in result
        assert "Metal Underground Water Pipe" in result
        assert "Metal In-ground Support Structure" in result

    def test_invalid_section_returns_error_with_suggestions(self):
        """A bogus section ID should produce an error and similar-ID suggestions."""
        result = nec_lookup.invoke({"section_ids": ["999.99"]})
        assert "Error" in result
        assert "not found" in result

    def test_close_typo_section_returns_suggestions(self):
        """A near-miss like '250.5' (instead of 250.50) should suggest real IDs."""
        result = nec_lookup.invoke({"section_ids": ["250.5"]})
        assert "Error" in result
        assert "Similar section IDs" in result
        assert "250.50" in result or "250.52" in result

    def test_section_110_26_clearances(self):
        """Section 110.26 (Spaces About Electrical Equipment) is a frequently
        referenced section -- verify it returns substantive content."""
        result = nec_lookup.invoke({"section_ids": ["110.26"]})
        assert "110.26" in result
        assert "Article 110" in result
        assert len(result) > 100

    def test_multiple_sections_in_single_call(self):
        """Requesting multiple section IDs should return content for all of them."""
        result = nec_lookup.invoke({"section_ids": ["90.1", "250.50"]})
        assert "[Section 90.1" in result
        assert "[Section 250.50" in result


# ===========================================================================
# Table lookup tests
# ===========================================================================


class TestTableLookup:
    """Test nec_lookup when called with table_ids only."""

    def test_known_table_returns_markdown(self):
        """Table 310.16 (Ampacities) should come back as a markdown table."""
        result = nec_lookup.invoke({"table_ids": ["Table 310.16"]})
        assert "Table 310.16" in result
        assert "Ampacities" in result
        assert "| " in result
        assert "| --- |" in result

    def test_table_has_data_rows(self):
        """Table 310.16 should contain recognisable conductor sizes."""
        result = nec_lookup.invoke({"table_ids": ["Table 310.16"]})
        assert "14" in result or "12" in result

    def test_table_with_footnotes(self):
        """Table 220.55 has footnotes -- they should appear as blockquotes."""
        result = nec_lookup.invoke({"table_ids": ["Table 220.55"]})
        assert "> " in result

    def test_table_id_normalisation_spaces(self):
        """'Table 220.55' and 'Table220.55' should resolve to the same table."""
        with_space = nec_lookup.invoke({"table_ids": ["Table 220.55"]})
        without_space = nec_lookup.invoke({"table_ids": ["Table220.55"]})
        assert with_space == without_space

    def test_table_id_normalisation_case(self):
        """Case-insensitive inputs like 'table 310.16' should still resolve."""
        result = nec_lookup.invoke({"table_ids": ["table 310.16"]})
        assert "Table 310.16" in result
        assert "| " in result

    def test_table_id_normalisation_extra_spaces(self):
        """Extra whitespace like 'TABLE  220.55' should still resolve."""
        result = nec_lookup.invoke({"table_ids": ["TABLE  220.55"]})
        assert "Table 220.55" in result

    def test_invalid_table_returns_error_with_suggestions(self):
        """A bogus table ID should produce an error and similar-ID suggestions."""
        result = nec_lookup.invoke({"table_ids": ["Table 999.99"]})
        assert "Error" in result
        assert "not found" in result

    def test_parenthetical_table_id(self):
        """Parenthetical table IDs like 'Table 110.26(A)(1)' should resolve."""
        result = nec_lookup.invoke({"table_ids": ["Table 110.26(A)(1)"]})
        assert "110.26" in result
        assert "| " in result

    def test_multiple_tables_in_single_call(self):
        """Requesting multiple table IDs should return content for all of them."""
        result = nec_lookup.invoke({"table_ids": ["Table 310.16", "Table 220.55"]})
        assert "Table 310.16" in result
        assert "Table 220.55" in result


# ===========================================================================
# Combined lookup tests
# ===========================================================================


class TestCombinedLookup:
    """Test nec_lookup with both section_ids and table_ids supplied."""

    def test_both_section_and_table_returns_both(self):
        """Providing both IDs should return section content AND table content."""
        result = nec_lookup.invoke({"section_ids": ["250.50"], "table_ids": ["Table 310.16"]})
        assert "Section 250.50" in result
        assert "grounding electrode" in result.lower()
        assert "Table 310.16" in result
        assert "| " in result


# ===========================================================================
# Error / edge-case tests
# ===========================================================================


class TestLookupErrors:

    def test_no_args_returns_error(self):
        """Calling with neither section_ids nor table_ids should return an error."""
        result = nec_lookup.invoke({})
        assert "Error" in result
        assert "at least one" in result

    def test_empty_lists_returns_error(self):
        """Explicitly empty lists should be treated as missing."""
        result = nec_lookup.invoke({"section_ids": [], "table_ids": []})
        assert "Error" in result

    def test_empty_strings_in_lists_returns_error(self):
        """Lists containing only empty strings should be treated as missing."""
        result = nec_lookup.invoke({"section_ids": ["", "  "], "table_ids": [""]})
        assert "Error" in result

    def test_exceeding_max_ids_returns_error(self):
        """Requesting more than 10 total IDs should return an error."""
        many_sections = [f"250.{i}" for i in range(8)]
        many_tables = ["Table 310.16", "Table 220.55", "Table 110.26(A)(1)"]
        result = nec_lookup.invoke({"section_ids": many_sections, "table_ids": many_tables})
        assert "Error" in result
        assert "maximum" in result.lower()

    def test_exactly_ten_ids_succeeds(self):
        """Requesting exactly 10 total IDs should NOT return an error."""
        sections = ["90.1", "250.50", "250.52", "110.26", "110.14"]
        tables = ["Table 310.16", "Table 220.55", "Table 110.26(A)(1)", "Table 310.16", "Table 220.55"]
        result = nec_lookup.invoke({"section_ids": sections, "table_ids": tables})
        assert "Split the request into multiple calls" not in result
        assert "[Section 90.1" in result
        assert "Table 310.16" in result


# ===========================================================================
# browse_nec_structure -- top-level (no args)
# ===========================================================================


class TestBrowseNoArgs:
    """Calling browse_nec_structure with no arguments lists all chapters."""

    def test_lists_all_eight_chapters(self):
        result = browse_nec_structure.invoke({})
        for ch_num in range(1, 9):
            assert f"Chapter {ch_num}:" in result

    def test_includes_well_known_articles(self):
        result = browse_nec_structure.invoke({})
        assert "Article 90:" in result
        assert "Article 250:" in result
        assert "Article 310:" in result
        assert "Article 680:" in result

    def test_chapter_titles_present(self):
        result = browse_nec_structure.invoke({})
        assert "General" in result
        assert "Wiring and Protection" in result
        assert "Communications Systems" in result


# ===========================================================================
# browse_nec_structure -- chapter level
# ===========================================================================


class TestBrowseChapter:
    """Calling with chapter=N lists the articles in that chapter."""

    def test_chapter_1_header(self):
        result = browse_nec_structure.invoke({"chapter": 1})
        assert result.startswith("Chapter 1:")

    def test_chapter_1_articles(self):
        result = browse_nec_structure.invoke({"chapter": 1})
        assert "Article 90:" in result
        assert "Article 100:" in result
        assert "Article 110:" in result

    def test_chapter_2_contains_article_250(self):
        result = browse_nec_structure.invoke({"chapter": 2})
        assert "Article 250:" in result
        assert "Grounding and Bonding" in result

    def test_chapter_does_not_leak_other_chapters(self):
        """Requesting chapter 1 should NOT include chapter 2 articles."""
        result = browse_nec_structure.invoke({"chapter": 1})
        assert "Article 250:" not in result
        assert "Article 310:" not in result

    def test_invalid_chapter_returns_error(self):
        result = browse_nec_structure.invoke({"chapter": 99})
        assert "Error" in result
        assert "not found" in result


# ===========================================================================
# browse_nec_structure -- article level
# ===========================================================================


class TestBrowseArticle:
    """Calling with article=N returns parts, subsections, and scope text."""

    def test_article_250_header(self):
        result = browse_nec_structure.invoke({"article": 250})
        assert result.startswith("Article 250:")
        assert "Grounding and Bonding" in result

    def test_article_250_has_parts(self):
        result = browse_nec_structure.invoke({"article": 250})
        assert "Part I:" in result
        assert "Part II:" in result
        assert "Part III:" in result

    def test_article_250_lists_subsections(self):
        result = browse_nec_structure.invoke({"article": 250})
        assert "250.50" in result
        assert "250.52" in result

    def test_article_includes_page_numbers(self):
        result = browse_nec_structure.invoke({"article": 250})
        assert "(page " in result

    def test_article_includes_scope_text(self):
        """The output should include the full text of the article's Scope (XXX.1) subsection."""
        result = browse_nec_structure.invoke({"article": 250})
        assert "--- Scope ---" in result
        assert "250.1" in result

    def test_article_without_explicit_parts(self):
        """Article 90 has no named parts -- subsections should still be listed."""
        result = browse_nec_structure.invoke({"article": 90})
        assert "90.1" in result
        assert "90.2" in result
        assert "Part I:" not in result

    def test_article_90_scope_text(self):
        """Article 90's scope text should mention 'use and application'."""
        result = browse_nec_structure.invoke({"article": 90})
        assert "--- Scope ---" in result
        assert "use and application" in result.lower()

    def test_invalid_article_returns_error(self):
        result = browse_nec_structure.invoke({"article": 999})
        assert "Error" in result
        assert "not found" in result

    def test_invalid_article_lists_valid_articles(self):
        """The error message should include the list of valid article numbers."""
        result = browse_nec_structure.invoke({"article": 999})
        assert "90" in result
        assert "250" in result


# ===========================================================================
# browse_nec_structure -- article + part level
# ===========================================================================


class TestBrowseArticlePart:
    """Calling with article=N, part=M narrows the outline to a single part."""

    def test_article_250_part_3_header(self):
        result = browse_nec_structure.invoke({"article": 250, "part": 3})
        assert "Part III:" in result
        assert "Grounding Electrode" in result

    def test_part_filter_excludes_other_parts(self):
        result = browse_nec_structure.invoke({"article": 250, "part": 3})
        assert "Part I:" not in result
        assert "Part II:" not in result

    def test_part_3_includes_relevant_sections(self):
        result = browse_nec_structure.invoke({"article": 250, "part": 3})
        assert "250.50" in result
        assert "250.52" in result
        assert "250.66" in result

    def test_part_still_includes_scope(self):
        """Even when filtered to a single part, the article scope should appear."""
        result = browse_nec_structure.invoke({"article": 250, "part": 3})
        assert "--- Scope ---" in result

    def test_invalid_part_returns_error(self):
        result = browse_nec_structure.invoke({"article": 250, "part": 99})
        assert "Error" in result
        assert "not found" in result

    def test_invalid_part_lists_valid_parts(self):
        result = browse_nec_structure.invoke({"article": 250, "part": 99})
        assert "Valid parts:" in result

    def test_part_without_article_falls_through(self):
        """Supplying part= without article= should fall through to the full chapter listing."""
        result = browse_nec_structure.invoke({"part": 3})
        assert "Chapter 1:" in result
        assert "Chapter 8:" in result
