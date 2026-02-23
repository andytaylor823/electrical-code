"""Unit tests for the agent tools module.

Only _build_context() is testable without mocking -- all other functions
require Azure OpenAI, ChromaDB, or embedding model access.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from nec_rag.agent.utils import _build_context as build_context


class TestBuildContext:

    def test_single_result(self):
        retrieved = [
            {
                "document": "250.50 Grounding Electrode. Shall be used as grounding electrode.",
                "metadata": {"section_id": "250.50", "article_num": 250, "page": 150},
                "distance": 0.15,
            }
        ]
        result = build_context(retrieved)
        assert "[Section 250.50, Article 250, page 150]" in result
        assert "250.50 Grounding Electrode" in result

    def test_multiple_results(self):
        retrieved = [
            {
                "document": "First subsection text.",
                "metadata": {"section_id": "110.1", "article_num": 110, "page": 30},
                "distance": 0.1,
            },
            {
                "document": "Second subsection text.",
                "metadata": {"section_id": "210.1", "article_num": 210, "page": 60},
                "distance": 0.2,
            },
        ]
        result = build_context(retrieved)
        assert "[Section 110.1, Article 110, page 30]" in result
        assert "[Section 210.1, Article 210, page 60]" in result
        assert "First subsection text." in result
        assert "Second subsection text." in result

    def test_results_separated_by_double_newline(self):
        retrieved = [
            {
                "document": "Text A.",
                "metadata": {"section_id": "110.1", "article_num": 110, "page": 30},
                "distance": 0.1,
            },
            {
                "document": "Text B.",
                "metadata": {"section_id": "210.1", "article_num": 210, "page": 60},
                "distance": 0.2,
            },
        ]
        result = build_context(retrieved)
        assert "\n\n" in result

    def test_empty_results(self):
        result = build_context([])
        assert result == ""
