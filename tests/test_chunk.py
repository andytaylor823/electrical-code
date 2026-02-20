"""Unit tests for the chunking module.

Tests cover the pure functions that split the structured NEC JSON into
subsection-level chunks with full parent metadata.

Skips load_and_chunk() since it requires reading from disk.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from nec_rag.data_preprocessing.embedding.chunk import (
    _build_subsection_text,
    _deduplicated_id,
    _iter_subsections,
    chunk_subsections,
)


def _make_structured_data(chapters: list[dict] | None = None) -> dict:
    """Build a minimal structured data dict matching the pipeline output schema."""
    if chapters is None:
        chapters = [
            {
                "chapter_num": 1,
                "title": "General",
                "articles": [
                    {
                        "article_num": 110,
                        "title": "Requirements for Electrical Installations",
                        "tables": [],
                        "parts": [
                            {
                                "part_num": None,
                                "title": None,
                                "subsections": [
                                    {
                                        "id": "110.1",
                                        "title": "Scope.",
                                        "page": 30,
                                        "front_matter": "110.1 Scope. This article covers general requirements.",
                                        "sub_items": [],
                                        "referenced_tables": [],
                                    }
                                ],
                                "referenced_tables": [],
                            }
                        ],
                        "referenced_tables": [],
                    }
                ],
                "referenced_tables": [],
            }
        ]
    return {"chapters": chapters, "definitions": []}


# ===========================================================================
# _build_subsection_text tests
# ===========================================================================


class TestBuildSubsectionText:

    def test_front_matter_only(self):
        sub = {
            "front_matter": "250.50 Grounding Electrode. Shall be used.",
            "sub_items": [],
        }
        result = _build_subsection_text(sub)
        assert result == "250.50 Grounding Electrode. Shall be used."

    def test_with_sub_items(self):
        sub = {
            "front_matter": "250.50 Grounding Electrode.",
            "sub_items": [
                {"content": "(A) Metal Underground Water Pipe."},
                {"content": "(B) Metal Frame of Building."},
            ],
        }
        result = _build_subsection_text(sub)
        assert "250.50 Grounding Electrode." in result
        assert "(A) Metal Underground Water Pipe." in result
        assert "(B) Metal Frame of Building." in result

    def test_empty_front_matter(self):
        sub = {
            "front_matter": "",
            "sub_items": [{"content": "(1) First item."}],
        }
        result = _build_subsection_text(sub)
        assert "(1) First item." in result

    def test_parts_joined_with_newlines(self):
        sub = {
            "front_matter": "Line 1",
            "sub_items": [{"content": "Line 2"}, {"content": "Line 3"}],
        }
        result = _build_subsection_text(sub)
        assert result == "Line 1\nLine 2\nLine 3"


# ===========================================================================
# _iter_subsections tests
# ===========================================================================


class TestIterSubsections:

    def test_yields_subsections_with_metadata(self):
        data = _make_structured_data()
        results = list(_iter_subsections(data))
        assert len(results) == 1
        sub, meta = results[0]
        assert sub["id"] == "110.1"
        assert meta["article_num"] == 110
        assert meta["chapter_num"] == 1
        assert meta["chapter_title"] == "General"

    def test_multiple_subsections(self):
        data = _make_structured_data(
            [
                {
                    "chapter_num": 2,
                    "title": "Wiring and Protection",
                    "articles": [
                        {
                            "article_num": 210,
                            "title": "Branch Circuits",
                            "tables": [],
                            "parts": [
                                {
                                    "part_num": "I",
                                    "title": "General Provisions",
                                    "subsections": [
                                        {"id": "210.1", "title": "Scope.", "page": 60, "front_matter": "text", "sub_items": [], "referenced_tables": []},
                                        {"id": "210.2", "title": "Other.", "page": 60, "front_matter": "text2", "sub_items": [], "referenced_tables": []},
                                    ],
                                    "referenced_tables": [],
                                }
                            ],
                            "referenced_tables": [],
                        }
                    ],
                    "referenced_tables": [],
                }
            ]
        )
        results = list(_iter_subsections(data))
        assert len(results) == 2
        assert results[0][0]["id"] == "210.1"
        assert results[1][0]["id"] == "210.2"
        assert results[0][1]["part_num"] == "I"

    def test_null_part_num_converted(self):
        """A None part_num should be converted to -1 for ChromaDB compatibility."""
        data = _make_structured_data()
        results = list(_iter_subsections(data))
        meta = results[0][1]
        assert meta["part_num"] == -1

    def test_empty_chapters(self):
        data = {"chapters": [], "definitions": []}
        results = list(_iter_subsections(data))
        assert not results


# ===========================================================================
# _deduplicated_id tests
# ===========================================================================


class TestDeduplicatedId:

    def test_unique_id(self):
        seen = set()
        result = _deduplicated_id(110, "110.1", seen)
        assert result == "110_110.1"
        assert "110_110.1" in seen

    def test_duplicate_id_gets_suffix(self):
        seen = {"110_110.1"}
        result = _deduplicated_id(110, "110.1", seen)
        assert result == "110_110.1_2"
        assert "110_110.1_2" in seen

    def test_triple_duplicate(self):
        seen = {"110_110.1", "110_110.1_2"}
        result = _deduplicated_id(110, "110.1", seen)
        assert result == "110_110.1_3"

    def test_different_articles(self):
        seen = set()
        r1 = _deduplicated_id(110, "110.1", seen)
        r2 = _deduplicated_id(210, "210.1", seen)
        assert r1 == "110_110.1"
        assert r2 == "210_210.1"


# ===========================================================================
# chunk_subsections end-to-end tests
# ===========================================================================


class TestChunkSubsections:

    def test_basic_chunking(self):
        data = _make_structured_data()
        chunks = chunk_subsections(data)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["id"] == "110_110.1"
        assert "110.1 Scope" in chunk["text"]
        assert chunk["metadata"]["section_id"] == "110.1"
        assert chunk["metadata"]["article_num"] == 110
        assert chunk["metadata"]["chapter_num"] == 1
        assert chunk["metadata"]["page"] == 30

    def test_multiple_chunks(self):
        data = _make_structured_data(
            [
                {
                    "chapter_num": 1,
                    "title": "General",
                    "articles": [
                        {
                            "article_num": 110,
                            "title": "Requirements",
                            "tables": [],
                            "parts": [
                                {
                                    "part_num": None,
                                    "title": None,
                                    "subsections": [
                                        {"id": "110.1", "title": "Scope.", "page": 30, "front_matter": "Scope text.", "sub_items": [], "referenced_tables": []},
                                        {"id": "110.2", "title": "Approval.", "page": 31, "front_matter": "Approval text.", "sub_items": [], "referenced_tables": []},
                                    ],
                                    "referenced_tables": [],
                                }
                            ],
                            "referenced_tables": [],
                        }
                    ],
                    "referenced_tables": [],
                }
            ]
        )
        chunks = chunk_subsections(data)
        assert len(chunks) == 2
        assert chunks[0]["id"] == "110_110.1"
        assert chunks[1]["id"] == "110_110.2"

    def test_empty_data(self):
        data = {"chapters": [], "definitions": []}
        chunks = chunk_subsections(data)
        assert not chunks

    def test_title_truncated(self):
        """Very long titles should be truncated to 500 chars for ChromaDB metadata limits."""
        data = _make_structured_data(
            [
                {
                    "chapter_num": 1,
                    "title": "General",
                    "articles": [
                        {
                            "article_num": 110,
                            "title": "Requirements",
                            "tables": [],
                            "parts": [
                                {
                                    "part_num": None,
                                    "title": None,
                                    "subsections": [
                                        {
                                            "id": "110.1",
                                            "title": "A" * 600,
                                            "page": 30,
                                            "front_matter": "text",
                                            "sub_items": [],
                                            "referenced_tables": [],
                                        }
                                    ],
                                    "referenced_tables": [],
                                }
                            ],
                            "referenced_tables": [],
                        }
                    ],
                    "referenced_tables": [],
                }
            ]
        )
        chunks = chunk_subsections(data)
        assert len(chunks[0]["metadata"]["title"]) == 500
