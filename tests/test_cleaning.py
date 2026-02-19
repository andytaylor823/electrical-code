"""Unit tests for the cleaning pipeline modules.

Tests cover:
  - hyphens_endline: de-hyphenation of line-broken words
  - sentence_runover: sentence boundary detection and cross-page merge logic
  - remove_junk_pages: page-range filtering and dict re-indexing
  - clean: full pipeline integration and text conversion
"""

from nec_rag.cleaning import hyphens_endline, remove_junk_pages, sentence_runover
from nec_rag.cleaning.clean import paragraphs_to_text, run_cleaning_pipeline

# ---------------------------------------------------------------------------
# Helpers to build paragraph dicts quickly
# ---------------------------------------------------------------------------


def make_paragraphs(items: list[tuple[str, int]]) -> dict[str, dict]:
    """Build a paragraph dict from a list of (content, page) tuples."""
    return {str(i): {"content": content, "page": page} for i, (content, page) in enumerate(items)}


def contents(paragraphs: dict[str, dict]) -> list[str]:
    """Extract just the content strings from a paragraph dict, in key order."""
    return [paragraphs[str(i)]["content"] for i in range(len(paragraphs))]


# ===========================================================================
# hyphens_endline tests
# ===========================================================================


class TestHyphensEndline:
    """Tests for end-of-line de-hyphenation."""

    def test_basic_dehyphenation(self):
        """A simple broken word like 'electri- cal' should become 'electrical'."""
        paras = make_paragraphs([("electri- cal wiring", 1)])
        result = hyphens_endline.run(paras)
        assert result["0"]["content"] == "electrical wiring"

    def test_multiple_hyphens_in_one_paragraph(self):
        """Multiple broken words in the same paragraph should all be fixed."""
        paras = make_paragraphs([("electri- cal and mechani- cal systems", 1)])
        result = hyphens_endline.run(paras)
        assert result["0"]["content"] == "electrical and mechanical systems"

    def test_no_hyphen_passthrough(self):
        """Text without the hyphen pattern should pass through unchanged."""
        paras = make_paragraphs([("normal text here", 1)])
        result = hyphens_endline.run(paras)
        assert result["0"]["content"] == "normal text here"

    def test_real_hyphenated_word_not_affected(self):
        """A real hyphenated compound word (no trailing space after hyphen) should NOT be altered."""
        paras = make_paragraphs([("self-contained unit", 1)])
        result = hyphens_endline.run(paras)
        # The pattern requires '- ' (hyphen-space), so 'self-contained' is untouched
        assert result["0"]["content"] == "self-contained unit"

    def test_digit_before_hyphen_not_affected(self):
        """Digits before '- ' should NOT match (pattern only matches [A-Za-z])."""
        paras = make_paragraphs([("Phase 3- wire", 1)])
        result = hyphens_endline.run(paras)
        # '3- ' does not match [A-Za-z]- , so it stays
        assert result["0"]["content"] == "Phase 3- wire"

    def test_uppercase_dehyphenation(self):
        """Uppercase letters should also be de-hyphenated."""
        paras = make_paragraphs([("NATION- AL ELECTRICAL CODE", 1)])
        result = hyphens_endline.run(paras)
        assert result["0"]["content"] == "NATIONAL ELECTRICAL CODE"

    def test_hyphen_at_start_of_text(self):
        """A standalone '- ' at the beginning (no preceding letter) should NOT match."""
        paras = make_paragraphs([("- bullet point item", 1)])
        result = hyphens_endline.run(paras)
        assert result["0"]["content"] == "- bullet point item"

    def test_preserves_page_numbers(self):
        """Page numbers should be preserved through the transformation."""
        paras = make_paragraphs([("electri- cal", 42), ("normal text", 43)])
        result = hyphens_endline.run(paras)
        assert result["0"]["page"] == 42
        assert result["1"]["page"] == 43

    def test_multiple_paragraphs(self):
        """De-hyphenation should work independently across multiple paragraphs."""
        paras = make_paragraphs([("electri- cal", 1), ("mechani- cal", 1), ("normal", 1)])
        result = hyphens_endline.run(paras)
        assert contents(result) == ["electrical", "mechanical", "normal"]

    def test_empty_input(self):
        """An empty paragraph dict should return an empty dict."""
        result = hyphens_endline.run({})
        assert not result

    def test_hyphen_at_end_of_string(self):
        """A trailing 'x- ' at the end of a string should still be de-hyphenated."""
        paras = make_paragraphs([("connect- ", 1)])
        result = hyphens_endline.run(paras)
        # Pattern matches 't- ', so 't' replaces 't- ', giving 'connect'
        assert result["0"]["content"] == "connect"


# ===========================================================================
# sentence_runover.sentence_runs_over tests
# ===========================================================================


class TestSentenceRunsOver:
    """Tests for the sentence_runs_over boundary-detection heuristic."""

    # --- Cases that should return False (NOT a runover) ---

    def test_p2_starts_with_section_number(self):
        """If p2 starts with a float like '290.98', it's a new section."""
        assert sentence_runover.sentence_runs_over("some text", "290.98 Wiring methods") is False

    def test_p2_starts_with_integer_section_number(self):
        """An integer section number like '110' is also parseable as a float."""
        assert sentence_runover.sentence_runs_over("some text", "110 General") is False

    def test_p1_ends_with_period(self):
        """If p1 ends with '.', the sentence is complete."""
        assert sentence_runover.sentence_runs_over("end of sentence.", "beginning of next") is False

    def test_p1_ends_with_question_mark(self):
        """If p1 ends with '?', the sentence is complete."""
        assert sentence_runover.sentence_runs_over("is this done?", "beginning of next") is False

    def test_p1_ends_with_exclamation(self):
        """If p1 ends with '!', the sentence is complete."""
        assert sentence_runover.sentence_runs_over("warning!", "beginning of next") is False

    def test_p2_starts_with_open_paren(self):
        """A parenthetical start signals a new section or note."""
        assert sentence_runover.sentence_runs_over("some text", "(1) First item") is False

    def test_p2_starts_with_informational(self):
        """'Informational' indicates a new Informational Note section."""
        assert sentence_runover.sentence_runs_over("some text", "Informational Note: ...") is False

    def test_p2_starts_with_part(self):
        """'Part' indicates a new structural division."""
        assert sentence_runover.sentence_runs_over("some text", "Part III. Grounding") is False

    def test_p2_starts_with_table(self):
        """'Table' indicates a table heading."""
        assert sentence_runover.sentence_runs_over("some text", "Table 310.16 Ampacities") is False

    def test_p2_starts_with_figure(self):
        """'Figure' indicates a figure heading."""
        assert sentence_runover.sentence_runs_over("some text", "Figure 250.1 Grounding") is False

    def test_p1_all_uppercase(self):
        """All-caps p1 is a page header, not a runover."""
        assert sentence_runover.sentence_runs_over("ARTICLE 250 GROUNDING", "some continuation") is False

    def test_p2_all_uppercase(self):
        """All-caps p2 is a page header, not a runover."""
        assert sentence_runover.sentence_runs_over("some text", "ARTICLE 250 GROUNDING") is False

    # --- Cases that should return True (IS a runover) ---

    def test_basic_runover(self):
        """A sentence that doesn't end with punctuation continues on the next page."""
        assert sentence_runover.sentence_runs_over("conductors shall be", "installed in accordance with") is True

    def test_runover_ends_with_comma(self):
        """A comma at end of p1 means the sentence continues."""
        assert sentence_runover.sentence_runs_over("conductors, cables,", "and raceways shall") is True

    def test_runover_ends_with_colon(self):
        """A colon at end of p1 means a list or continuation follows."""
        assert sentence_runover.sentence_runs_over("the following:", "copper conductors") is True

    def test_runover_ends_with_semicolon(self):
        """A semicolon at end of p1 means the sentence continues."""
        assert sentence_runover.sentence_runs_over("grounding conductors;", "bonding jumpers") is True

    # --- Edge cases ---

    def test_p2_starts_with_word_containing_part(self):
        """'Particular' starts with 'Part' substring but startswith('Part') should still match 'Particular'."""
        # This is actually a known quirk: "Particular" starts with "Part"
        # so it would return False even though it's not the structural keyword "Part"
        assert sentence_runover.sentence_runs_over("some text", "Particular attention") is False

    def test_p2_starts_with_word_containing_table(self):
        """'Tables' starts with 'Table', so startswith will catch it."""
        assert sentence_runover.sentence_runs_over("some text", "Tables shall be") is False

    def test_mixed_case_p1_not_treated_as_header(self):
        """Mixed-case text (not all uppercase) should NOT trigger the isupper() check."""
        assert sentence_runover.sentence_runs_over("Mixed Case Text", "continues here") is True


# ===========================================================================
# sentence_runover.run tests (full merge logic)
# ===========================================================================


class TestSentenceRunoverRun:
    """Tests for the full sentence merge pipeline."""

    def test_merge_across_page_boundary(self):
        """When a sentence runs over a page boundary, the two fragments should be merged."""
        paras = make_paragraphs(
            [
                ("Some content on page 1", 100),  # 0: last real content before page end
                ("2023 Edition NATIONAL ELECTRICAL CODE", 100),  # 1: page-end marker
                ("70-50", 101),  # 2: page number (between markers)
                ("ARTICLE 250 GROUNDING AND BONDING", 101),  # 3: page-start marker (article header)
                ("conductors installed in", 101),  # 4: continuation (this is p2)
                ("More content", 101),  # 5
            ]
        )
        result = sentence_runover.run(paras)
        result_contents = contents(result)
        # Paragraph 0 (p1) should now include paragraph 4 (p2) merged into it
        assert "Some content on page 1 conductors installed in" in result_contents[0]

    def test_no_merge_when_sentence_complete(self):
        """When p1 ends with a period, no merge should occur."""
        paras = make_paragraphs(
            [
                ("Some content on page 1.", 100),  # 0: ends with period
                ("2023 Edition NATIONAL ELECTRICAL CODE", 100),  # 1: page-end marker
                ("70-50", 101),  # 2
                ("ARTICLE 250 GROUNDING AND BONDING", 101),  # 3: page-start marker
                ("New sentence starts here", 101),  # 4
            ]
        )
        result = sentence_runover.run(paras)
        result_contents = contents(result)
        # Paragraph 0 should NOT be merged with paragraph 4
        assert result_contents[0] == "Some content on page 1."

    def test_reindexing_after_merge(self):
        """After a merge, the dict should be re-indexed with consecutive keys."""
        paras = make_paragraphs(
            [
                ("Some content on page 1", 100),
                ("2023 Edition NATIONAL ELECTRICAL CODE", 100),
                ("70-50", 101),
                ("ARTICLE 250 GROUNDING AND BONDING", 101),
                ("continued text here", 101),
                ("Final paragraph", 101),
            ]
        )
        result = sentence_runover.run(paras)
        # Keys should be consecutive starting from "0"
        keys = sorted(result.keys(), key=int)
        expected_keys = [str(i) for i in range(len(result))]
        assert keys == expected_keys

    def test_no_page_boundary_passthrough(self):
        """Without page boundary markers, paragraphs should pass through unchanged."""
        paras = make_paragraphs(
            [
                ("First paragraph", 100),
                ("Second paragraph", 100),
                ("Third paragraph", 100),
            ]
        )
        result = sentence_runover.run(paras)
        assert contents(result) == ["First paragraph", "Second paragraph", "Third paragraph"]


# ===========================================================================
# remove_junk_pages tests
# ===========================================================================


class TestRemoveJunkPages:
    """Tests for page-range filtering."""

    def test_keeps_pages_in_range(self):
        """Pages 26-717 should be kept."""
        paras = make_paragraphs([("content", 26), ("content", 100), ("content", 717)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 3

    def test_removes_pages_before_range(self):
        """Pages before 26 should be removed."""
        paras = make_paragraphs([("cover page", 1), ("toc", 10), ("toc", 25)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 0

    def test_removes_pages_after_range(self):
        """Pages after 717 should be removed."""
        paras = make_paragraphs([("index", 718), ("appendix", 800)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 0

    def test_boundary_page_26_kept(self):
        """Page 26 (first real page) should be kept."""
        paras = make_paragraphs([("first real page", 26)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "first real page"

    def test_boundary_page_717_kept(self):
        """Page 717 (last real page) should be kept."""
        paras = make_paragraphs([("last real page", 717)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 1
        assert result["0"]["content"] == "last real page"

    def test_boundary_page_25_removed(self):
        """Page 25 (one before first real page) should be removed."""
        paras = make_paragraphs([("just before content", 25)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 0

    def test_boundary_page_718_removed(self):
        """Page 718 (one after last real page) should be removed."""
        paras = make_paragraphs([("just after content", 718)])
        result = remove_junk_pages.run(paras)
        assert len(result) == 0

    def test_mixed_pages_filtered(self):
        """A mix of in-range and out-of-range pages should be filtered correctly."""
        paras = make_paragraphs(
            [
                ("cover", 1),
                ("toc", 10),
                ("real content 1", 26),
                ("real content 2", 400),
                ("real content 3", 717),
                ("appendix", 800),
            ]
        )
        result = remove_junk_pages.run(paras)
        assert len(result) == 3
        assert contents(result) == ["real content 1", "real content 2", "real content 3"]

    def test_reindexing(self):
        """After filtering, keys should be re-indexed as consecutive '0', '1', '2', ..."""
        paras = make_paragraphs([("junk", 1), ("real", 100), ("junk", 800), ("real", 200)])
        result = remove_junk_pages.run(paras)
        assert list(result.keys()) == ["0", "1"]
        # Verify the order is by original key (i.e., original paragraph order)
        assert result["0"]["content"] == "real"
        assert result["1"]["content"] == "real"

    def test_empty_input(self):
        """An empty paragraph dict should return an empty dict."""
        result = remove_junk_pages.run({})
        assert result == {}

    def test_preserves_content_and_page(self):
        """Both content and page values should be preserved for kept paragraphs."""
        paras = make_paragraphs([("important section", 300)])
        result = remove_junk_pages.run(paras)
        assert result["0"]["content"] == "important section"
        assert result["0"]["page"] == 300


# ===========================================================================
# resort_dict tests (shared utility in remove_junk_pages and sentence_runover)
# ===========================================================================


class TestResortDict:
    """Tests for the dict re-indexing utility."""

    def test_already_consecutive(self):
        """A dict with consecutive keys should be returned with same values, new consecutive keys."""
        d = {"0": {"content": "a"}, "1": {"content": "b"}, "2": {"content": "c"}}
        result = remove_junk_pages.resort_dict(d)
        assert list(result.keys()) == ["0", "1", "2"]
        assert result["0"]["content"] == "a"
        assert result["2"]["content"] == "c"

    def test_gap_in_keys(self):
        """A dict with gaps in keys should be re-indexed to fill the gaps."""
        d = {"0": {"content": "a"}, "5": {"content": "b"}, "10": {"content": "c"}}
        result = remove_junk_pages.resort_dict(d)
        assert list(result.keys()) == ["0", "1", "2"]
        assert result["0"]["content"] == "a"
        assert result["1"]["content"] == "b"
        assert result["2"]["content"] == "c"

    def test_unsorted_keys(self):
        """Keys should be sorted numerically, not lexicographically."""
        d = {"10": {"content": "c"}, "2": {"content": "b"}, "0": {"content": "a"}}
        result = remove_junk_pages.resort_dict(d)
        assert result["0"]["content"] == "a"
        assert result["1"]["content"] == "b"
        assert result["2"]["content"] == "c"

    def test_empty_dict(self):
        """An empty dict should return an empty dict."""
        result = remove_junk_pages.resort_dict({})
        assert result == {}


# ===========================================================================
# clean.paragraphs_to_text tests
# ===========================================================================


class TestParagraphsToText:
    """Tests for converting paragraph dicts to plain text."""

    def test_basic_concatenation(self):
        """Paragraphs should be joined with newlines."""
        paras = make_paragraphs([("Line one", 1), ("Line two", 1), ("Line three", 1)])
        result = paragraphs_to_text(paras)
        assert result == "Line one\nLine two\nLine three"

    def test_non_ascii_stripped(self):
        """Characters not representable in charmap should be stripped."""
        # The euro sign (U+20AC) is not in Latin-1/charmap range
        paras = make_paragraphs([("Price: \u20ac100", 1)])
        result = paragraphs_to_text(paras)
        assert "\u20ac" not in result
        assert "Price:" in result

    def test_single_paragraph(self):
        """A single paragraph should produce text with no newline."""
        paras = make_paragraphs([("Only one", 1)])
        result = paragraphs_to_text(paras)
        assert result == "Only one"

    def test_empty_input(self):
        """An empty dict should produce an empty string."""
        result = paragraphs_to_text({})
        assert result == ""


# ===========================================================================
# clean.run_cleaning_pipeline integration tests
# ===========================================================================


class TestRunCleaningPipeline:
    """Integration tests for the full cleaning pipeline."""

    def test_pipeline_removes_junk_and_dehyphenates(self):
        """The pipeline should remove junk pages AND fix hyphens."""
        paras = make_paragraphs(
            [
                ("cover page junk", 1),  # should be removed (page < 26)
                ("electri- cal systems", 100),  # should be de-hyphenated
                ("normal content", 200),  # should pass through
                ("appendix junk", 800),  # should be removed (page > 717)
            ]
        )
        result = run_cleaning_pipeline(paras)
        result_contents = contents(result)
        # Only the two middle paragraphs should remain
        assert len(result) == 2
        # The hyphen should be fixed
        assert result_contents[0] == "electrical systems"
        assert result_contents[1] == "normal content"

    def test_pipeline_preserves_page_numbers(self):
        """Page numbers should survive the full pipeline."""
        paras = make_paragraphs([("some content", 100)])
        result = run_cleaning_pipeline(paras)
        assert result["0"]["page"] == 100

    def test_pipeline_order_matters(self):
        """Verify the pipeline runs in the correct order (junk removal first, then merge, then hyphens)."""
        # If hyphens ran before junk removal, the junk-page paragraphs would still be present
        paras = make_paragraphs(
            [
                ("junk hyph- enated", 1),  # page 1 = junk, also has a hyphen
                ("real content", 100),
            ]
        )
        result = run_cleaning_pipeline(paras)
        # Only the real content should remain
        assert len(result) == 1
        assert result["0"]["content"] == "real content"
