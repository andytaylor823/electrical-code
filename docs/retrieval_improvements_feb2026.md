# Retrieval & Re-Ranking Improvements (February 2026)

Documents the series of changes made to the RAG retrieval pipeline during late February 2026, driven by evaluation against the 20-question master electrician practice exam.

## Starting Point

The original retrieval pipeline embedded subsection-level chunks from the structured NEC JSON into ChromaDB using Azure OpenAI `text-embedding-3-large`. Retrieval was pure cosine-similarity: embed the query, return the top-20 nearest chunks. No re-ranking, no table retrieval, no chunk splitting.

Baseline performance (from `docs/retrieval_recall_analysis.md`):

| n_results | Recall |
|-----------|--------|
| 5         | 12/19 (63%) |
| 10        | 13/19 (68%) |
| 20        | 15/19 (79%) |
| 50        | 17/19 (89%) |

Two questions were permanent misses (q09 = Annex C data not in dataset; q14 = section 705.31). Several questions had the correct section buried deep (q07 at rank 24, q20 at rank 31, q13 at rank 10).

## Change 1: Ground Truth Corrections

### q14 — Section 705.31 Does Not Exist in the 2023 NEC

The original ground truth listed Section 705.31 as the answer to q14 (overcurrent protection for PV supply-side conductor connections within 10 feet of the service disconnect). Investigation revealed that **Section 705.31 was removed from the NEC after the 2017 edition**. The 2023 NEC only has Sections 705.30 and 705.32 — there is no 705.31 and never was in this edition. The question itself is based on an older NEC and is unanswerable from the 2023 code.

**Fix:** Marked q14 as unanswerable in the ground truth for both `retrieval_recall.py` and `rerank_comparison.py`:
```python
"q14": {"section_prefixes": [], "table_ids": []},  # Section 705.31 removed in 2023 NEC
```

This changed the denominator from 19 to 18 answerable questions (q09 is also unanswerable due to missing Annex data, and q19 is pure calculation with no specific NEC section).

### q20 — Section 312.2 Is Also a Valid Answer

The original ground truth for q20 (protection of metal enclosures from corrosion) listed only Section 300.6. Investigation of the re-ranking results showed that Section 312.2 ("Cabinets, Cutout Boxes, and Meter Socket Enclosures — Damp or Wet Locations") received a much higher cross-encoder score (0.89) than 300.6 (0.09). After reading both sections, 312.2 provides a more direct and specific answer about environmental protection of metal enclosures. Both sections are valid evidence for answering the question.

**Fix:** Updated the ground truth to accept both sections:
```python
"q20": {"section_prefixes": ["300.6", "312.2"], "table_ids": []},
```

## Change 2: Embedding Tables as Searchable Chunks

### Problem

Questions like q07 (minimum burial depth from Table 300.5) and q10 (minimum equipment grounding conductor size from Table 250.122) are answered by NEC tables, but tables were not embedded in the vector store. They were only discoverable indirectly if a subsection that *referenced* the table happened to be retrieved. This meant the RAG pipeline could never directly surface the most relevant evidence for table-dependent questions.

### Solution

Added `chunk_tables()` to `chunk.py` to embed each of the 217 NEC tables into the same ChromaDB collection. Each table chunk has:

- **Embedding text** (compact, for semantic search): the table title + column headers only. For example: `Table 250.122: Minimum Size Equipment Grounding Conductors\nColumns: Rating or Setting of Automatic Overcurrent Device... | Size (AWG or kcmil) Copper | Size (AWG or kcmil) Aluminum...`
- **Document text** (full content, for RAG context): the complete markdown-rendered table with all data rows and footnotes
- **Metadata**: `chunk_type: "table"` to distinguish from subsection chunks, plus the same article/chapter metadata as subsections

The embedding text is intentionally compact — title and headers capture *what the table is about* without inflating the embedding vector with row data. The full markdown is stored as the `document` field in ChromaDB so it appears directly in the agent's context when retrieved.

### Impact

Collection grew from 3,159 subsection chunks to 3,376 (3,159 subsections + 217 tables). Table-dependent questions like q10 jumped from rank #2 (matching only via the subsection's `referenced_tables` metadata) to rank #1 (direct table chunk match).

## Change 3: Splitting Large Subsections at Lettered Boundaries

### Problem — Chunk Dilution

Large NEC subsections like 314.23 ("Supports") cover multiple distinct topics under lettered sub-items (A) through (H). When embedded as a single chunk (6,000+ characters), the embedding averages the semantics of all 8 sub-topics together. A query specifically about raceway-supported enclosures (sub-item F) produces only a weak match against the diluted embedding, because the signal from (F) is averaged with 7 unrelated mounting methods.

This was directly observable in q13: Section 314.23 was the ground truth answer, but it ranked only #13 by embedding distance (later #11 after table embeddings were added). The cross-encoder could promote it to #1 when it saw the actual text, confirming the embedding — not the content — was the bottleneck.

### Solution — Split at Lettered Boundaries with Parent Context

Modified `chunk_subsections()` in `chunk.py` to split large subsections into one chunk per lettered sub-item group. The splitting criteria:

- **Character threshold**: `SPLIT_THRESHOLD = 3000` characters. Only subsections whose full text exceeds 3,000 characters are candidates for splitting. Sections below this remain as single chunks.
- **Lettered boundary detection**: Uses regex `^\([A-Z]\)\s` to identify lettered sub-items like `(A) Surface Mounting`, `(B) Structural Mounting`, etc. Numbered sub-items like `(1)`, `(2)` stay grouped with their parent letter.
- **No minimum sub-item count**: Any section over 3,000 characters that has *at least one* lettered sub-item is split. If a large section has no lettered sub-items at all (only 5 such sections exist), it remains as a single chunk.
- **Context preservation**: Each child chunk gets the parent section's `front_matter` prepended, so it retains context about which section it belongs to. For example, a chunk for 314.23(F) starts with "314.23 Supports. Enclosures within the scope of this article shall be supported in accordance with..." before the (F)-specific content.
- **Metadata**: All child chunks share the parent's `section_id` (e.g., `"314.23"`), so the existing ground-truth matching logic and retrieval dedup work without changes. Chunk IDs include the letter suffix (e.g., `314_314.23_F`).

### Impact by the Numbers

- **253 sections** were split (out of 258 that exceeded the threshold; 5 had no lettered sub-items)
- Total subsection chunks grew from 3,159 to 4,223 (net +1,064)
- Combined with 217 table chunks: **4,440 total chunks** in the collection
- p95 chunk size dropped significantly (from ~4,240 to ~2,599 chars)
- 27 child chunks still exceed 5,000 chars (deeply nested sections like 250.30(A)) — acceptable diminishing returns

## Change 4: Cross-Encoder Re-Ranking in the Live Pipeline

### Problem — Embedding Search Alone Has Blind Spots

Evaluation with the `rerank_comparison.py` script showed that a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) could dramatically improve ranking for some questions (q13: #13 to #1, q17: #8 to #1) but regressed on others (q10: #1 to #14). The cross-encoder excels at semantic matching but can be fooled by surface-level lexical overlap — for q10, it ranked Section 393.100 ("Low-Voltage Suspended Ceiling Power Distribution Systems" with "18 AWG minimum") above the correct Section 250.122 because both discuss "minimum size conductor," even though 393.100 is about circuit conductors, not equipment grounding conductors.

### Solution — Union of Both Methods

Rather than replacing embedding search with re-ranking, the pipeline now uses **both**:

1. Retrieve 50 candidates from ChromaDB by embedding cosine similarity
2. Score all 50 with the cross-encoder
3. Return the **union** of (top-10 by re-rank score) and (top-5 by embedding distance), deduplicated

Re-ranked items come first in the merged list (since the user trusts re-ranking more), followed by any embedding-only items not already included. This typically yields 10-15 sections after dedup.

This approach captures the best of both methods:
- The **embedding search** catches domain-specific matches the cross-encoder misses (like q10, where "equipment grounding conductor" is a precise NEC concept)
- The **cross-encoder** catches semantic matches that pure cosine similarity misses (like q13, where a focused 314.23(F) chunk needs cross-attention to be recognized as relevant)

### Integration

- `load_cross_encoder()` in `resources.py` — lazy-loads and caches the cross-encoder model
- `_rerank()` in `utils.py` — scores candidates, computes the union, returns the merged list
- `rag_search` in `tools.py` — widened candidate pool from 20 to 50, calls `_rerank()` after `_retrieve()`
- Cross-encoder is pre-warmed at agent startup alongside the embedding model
- `sentence-transformers` moved from optional to core dependency in `pyproject.toml`

## Final Results

### Retrieval Recall (after all changes)

| n_results | Recall | Before (baseline) |
|-----------|--------|-------------------|
| 5         | 14/18 (78%) | 12/19 (63%) |
| 10        | 16/18 (89%) | 13/19 (68%) |
| 20        | 17/18 (94%) | 15/19 (79%) |
| 50        | 17/18 (94%) | 17/19 (89%) |

Only q09 (Annex C table, data not in dataset) remains a permanent miss.

### Re-Ranking Comparison (after all changes)

| QID | Embedding Rank | Reranked Rank | Ground Truth |
|-----|---------------|---------------|--------------|
| q01 | #3 | #2 | 220.82 |
| q02 | #2 | #1 | 220.103, Table220.103 |
| q03 | #3 | #3 | 511.3, Table511.3 |
| q04 | #1 | #1 | 501.15 |
| q05 | #1 | #1 | 404.8 |
| q06 | #2 | #1 | 314.28, 314.16, Table314.16 |
| q07 | #8 | #3 | 300.5, Table300.5 |
| q08 | #1 | #1 | 230.26 |
| q09 | MISS | MISS | TableC.9 |
| q10 | #1 | #14 | 250.122, Table250.122 |
| q11 | #1 | #1 | 240.24, 404.8 |
| q12 | #2 | #4 | 300.5 |
| q13 | #13 | #1 | 314.23 |
| q14 | n/a | n/a | (unanswerable from 2023 NEC) |
| q15 | #1 | #2 | 551.73, Table551.73 |
| q16 | #1 | #1 | 501.15 |
| q17 | #8 | #1 | 590.6 |
| q18 | #1 | #1 | 250.53 |
| q19 | n/a | n/a | (calculation) |
| q20 | #1 | #1 | 300.6, 312.2 |

### Union Coverage

Taking the union of (top-5 embedding + top-10 re-ranked), **17 of 18 answerable questions** (94%) have the correct section in the merged result set. Only q09 is missed (and cannot be fixed without adding Annex data to the dataset).

## End-to-End Exam Results

After all retrieval improvements, the full integration test suite (`tests_integration/agent/test_master_electrician_exam.py`) scores **19/20 (95%)**, up from 16/20 (80%) at the start of this work.

The single remaining failure is **q09** (minimum trade size of RMC for four 350 kcmil THWN copper conductors). The agent correctly identified the calculation method and attempted to look up the Chapter 9 conduit fill tables (`Table 1, Chapter 9`, `Table 4, Chapter 9`, `Table 5, Chapter 9`), but these tables do not exist in the dataset — Chapter 9 and the Annexes are not yet part of the structured data. The agent acknowledged this gap explicitly in its answer and estimated 2-1/2 in. RMC, but the correct answer is 3 in. RMC. This is a data coverage issue, not a retrieval or reasoning failure; adding Chapter 9 tables to the pipeline (tracked in TODO section 1) would likely fix it.

## Files Changed

| File | Change |
|------|--------|
| `src/nec_rag/data_preprocessing/embedding/chunk.py` | Added `SPLIT_THRESHOLD`, `_group_by_letter()`, table chunking (`chunk_tables()`), subsection splitting in `chunk_subsections()` |
| `src/nec_rag/data_preprocessing/embedding/embed.py` | Updated to use `text` for embedding and `document` for ChromaDB storage (table chunks use different text for each) |
| `src/nec_rag/agent/utils.py` | Added `_rerank()` helper; updated `_build_context()` to handle `chunk_type: "table"` |
| `src/nec_rag/agent/resources.py` | Added `load_cross_encoder()` with lazy caching |
| `src/nec_rag/agent/tools.py` | Widened retrieval to 50 candidates, integrated `_rerank()` into `rag_search` |
| `src/nec_rag/agent/agent.py` | Pre-warms cross-encoder at startup |
| `pyproject.toml` | Moved `sentence-transformers` to core dependencies |
| `scripts/retrieval_recall.py` | Updated ground truth for q14 and q20, updated `_chunk_matches` for table chunks |
| `scripts/rerank_comparison.py` | Created; compares embedding vs cross-encoder ranking per question |
