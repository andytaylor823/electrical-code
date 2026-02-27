"""Unit tests for rag_search deduplication across multiple calls.

Mocks the embedding resources and retrieval layer so we can control
exactly which section IDs are returned, then verify that duplicate
sections are filtered out on subsequent calls.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from unittest.mock import patch

from nec_rag.agent.tools import rag_search, reset_seen_sections


def _fake_result(section_id: str, text: str = "") -> dict:
    """Build a minimal retrieved-result dict matching _retrieve()'s output format."""
    return {
        "document": text or f"Content of section {section_id}.",
        "metadata": {
            "section_id": section_id,
            "article_num": int(section_id.split(".")[0]),
            "page": 100,
            "referenced_tables": "",
        },
        "distance": 0.1,
    }


# Shared mock targets
_PATCH_EMBED = "nec_rag.agent.tools.load_embedding_resources"
_PATCH_RETRIEVE = "nec_rag.agent.tools._retrieve"


class TestRagSearchDeduplication:
    """Verify that rag_search filters out sections already returned by prior calls."""

    def setup_method(self):
        """Reset the seen-section set before every test."""
        reset_seen_sections()

    def test_ab_then_bc_yields_abc_not_abbc(self):
        """Core scenario: (A+B) then (B+C) should produce (A+B) + (C), not (A+B) + (B+C)."""
        result_a = _fake_result("250.50")
        result_b = _fake_result("250.52")
        result_c = _fake_result("110.26")

        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE) as mock_retrieve:
            # First call returns A + B
            mock_retrieve.return_value = [result_a, result_b]
            context_1 = rag_search.invoke({"user_request": "grounding electrodes"})

            # Second call returns B + C (B is a duplicate)
            mock_retrieve.return_value = [result_b, result_c]
            context_2 = rag_search.invoke({"user_request": "clearance requirements"})

        # First call should contain both A and B
        assert "250.50" in context_1
        assert "250.52" in context_1

        # Second call should contain only C (B was already seen)
        assert "110.26" in context_2
        assert "250.52" not in context_2

    def test_first_call_returns_all_results(self):
        """The very first rag_search call should never filter anything."""
        results = [_fake_result("250.50"), _fake_result("250.52"), _fake_result("110.26")]

        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE, return_value=results):
            context = rag_search.invoke({"user_request": "grounding electrodes"})

        assert "250.50" in context
        assert "250.52" in context
        assert "110.26" in context

    def test_all_duplicates_returns_empty_context(self):
        """If a second call returns only sections already seen, context should be empty."""
        result_a = _fake_result("250.50")
        result_b = _fake_result("250.52")

        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE) as mock_retrieve:
            # First call seeds the seen set
            mock_retrieve.return_value = [result_a, result_b]
            rag_search.invoke({"user_request": "first query"})

            # Second call returns the exact same sections
            mock_retrieve.return_value = [result_a, result_b]
            context_2 = rag_search.invoke({"user_request": "second query"})

        # No new sections, so no section headers should appear
        assert "250.50" not in context_2
        assert "250.52" not in context_2

    def test_reset_clears_seen_set(self):
        """After reset_seen_sections(), previously seen IDs should be returned again."""
        result_a = _fake_result("250.50")

        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE, return_value=[result_a]):
            rag_search.invoke({"user_request": "first invocation"})

            # Reset simulates the start of a new agent invocation
            reset_seen_sections()

            context = rag_search.invoke({"user_request": "new invocation"})

        assert "250.50" in context

    def test_three_calls_cumulative_dedup(self):
        """Three successive calls should accumulate the seen set correctly."""
        result_a = _fake_result("250.50")
        result_b = _fake_result("250.52")
        result_c = _fake_result("110.26")
        result_d = _fake_result("110.14")

        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE) as mock_retrieve:
            # Call 1: A + B
            mock_retrieve.return_value = [result_a, result_b]
            ctx1 = rag_search.invoke({"user_request": "query 1"})

            # Call 2: B + C (B is duplicate from call 1)
            mock_retrieve.return_value = [result_b, result_c]
            ctx2 = rag_search.invoke({"user_request": "query 2"})

            # Call 3: A + C + D (A and C are duplicates from calls 1 and 2)
            mock_retrieve.return_value = [result_a, result_c, result_d]
            ctx3 = rag_search.invoke({"user_request": "query 3"})

        # Call 1: A and B both present
        assert "250.50" in ctx1
        assert "250.52" in ctx1

        # Call 2: only C is new
        assert "110.26" in ctx2
        assert "250.52" not in ctx2

        # Call 3: only D is new (A seen in call 1, C seen in call 2)
        assert "110.14" in ctx3
        assert "250.50" not in ctx3
        assert "110.26" not in ctx3

    def test_nec_lookup_does_not_affect_seen_set(self):
        """nec_lookup should not add to the seen set — duplicate filtering is rag_search only."""
        from nec_rag.agent.tools import nec_lookup  # pylint: disable=import-outside-toplevel

        # Use nec_lookup to fetch section 90.1 (hits the real structured JSON)
        nec_lookup.invoke({"section_ids": ["90.1"]})

        # Now rag_search returns a result for 90.1 — it should NOT be filtered
        result_901 = _fake_result("90.1")
        with patch(_PATCH_EMBED, return_value=(None, None)), patch(_PATCH_RETRIEVE, return_value=[result_901]):
            context = rag_search.invoke({"user_request": "scope of the NEC"})

        assert "90.1" in context
