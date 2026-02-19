# Table Cleaning: End-to-End Process

This document describes the complete process used to extract, reconstruct, and
correct tables from the NFPA 70 NEC 2023 PDF. The raw OCR output contained no
table structure metadata — every cell was captured as a standalone paragraph.
Turning these flat streams of cell values back into properly structured tables
required a three-phase approach: automated detection and LLM reconstruction,
automated triage of failures, and manual interactive correction.

---

## Phase 1: Automated LLM Reconstruction

**Module:** `src/nec_rag/cleaning/tables.py`  
**Pipeline position:** Step 2 of the cleaning pipeline (`clean.py`), runs
after `remove_junk_pages` and before `sentence_runover`.

### The OCR Problem

Azure Document Intelligence extracted table content as a flat stream of
individual cell values — one paragraph per cell, in reading order (left to
right, top to bottom). The OCR produced **no metadata** about column count,
row boundaries, header vs. data cells, merged cells, or footnotes.

A single NEC table like Table 310.4(1) could produce 200+ fragments that
needed to be reassembled into a structured grid.

### Why Procedural Approaches Failed

An initial procedural approach attempted to reconstruct column structure by
counting headers, finding divisors of the data count, and choosing the most
likely column count. An ambiguity analysis (documented separately in
`docs/table_ambiguity.md`) found that **only ~5% of tables** could be
reconstructed unambiguously by this method:

| Metric | Count | Percentage |
|--------|------:|----------:|
| Unambiguously correct | 6 | 4.7% |
| Ambiguous column count | 39 | 30.5% |
| Boundary-sensitive | 122 | 95.3% |

The core issues were **column count ambiguity** (multiple valid column counts
for the same set of data cells) and **header/data boundary sensitivity**
(shifting the boundary between headers and data by 1–2 positions still
yielded valid table dimensions).

### The LLM Solution

The procedural code was replaced with an LLM-based reconstruction step that
kept the reliable parts (detection, extraction, interruption repair) and
delegated the ambiguous formatting to Azure OpenAI GPT with structured output.

#### What the procedural code still handles

The existing heuristic functions in `tables.py` proved reliable for:

1. **Table detection** (`find_table_starts`) — Identifies paragraphs matching
   `Table X.Y...` and validates that they are real tables (not textual
   references) by checking whether the next several paragraphs contain short,
   data-like content.

2. **Multi-page tracking** (`find_table_end`) — Follows `(continues)` /
   `Continued` markers to capture tables spanning multiple pages. Tracks page
   boundaries and distinguishes natural page breaks (table ended) from
   continuation markers (table continues).

3. **Content extraction** (`extract_table_content`) — Strips page markers
   (headers, footers, copyright, watermarks), continuation noise, and
   repeated table titles to produce a clean list of cell values.

4. **Paragraph interruption repair** (`detect_interruption`) — Detects when
   a table appears mid-sentence in the page layout, splitting a paragraph in
   two. The pre- and post-table fragments are re-joined.

#### What the LLM handles

The extracted cell fragments are sent to Azure OpenAI with a system prompt
that describes the OCR structure and a Pydantic-enforced output schema:

```python
class TableStructure(BaseModel):
    title: str
    column_headers: list[str]
    data_rows: list[list[str]]
    footnotes: list[str]

    @model_validator(mode="after")
    def validate_row_widths(self) -> "TableStructure":
        n_cols = len(self.column_headers)
        for i, row in enumerate(self.data_rows):
            if len(row) != n_cols:
                raise ValueError(...)
        return self
```

The `model_validator` guarantees that every data row has exactly
`len(column_headers)` cells, eliminating column-count ambiguity.

#### Caching

LLM results are cached to `data/intermediate/tables/table_llm_cache.json`,
keyed by normalized table ID (e.g. `Table240.6(A)`). On subsequent runs, the
cache is consulted first and the LLM is only called on cache misses. This
makes re-running the pipeline fast and deterministic.

#### Fallback

If the LLM is unavailable, the API call fails, or Pydantic validation
rejects the response, the table content is stored as a plain text block
(all fragments concatenated under the bold title). This preserves all
content but loses structural formatting.

#### Result

The automated LLM pass successfully reconstructed **217 tables** from the
NEC 2023 PDF. Each table was stored in the LLM cache and rendered as a
markdown table in the cleaned output. The vast majority were correct.

---

## Phase 2: Automated Triage

**Script:** `scripts/review_tables.py`

After the LLM pass, an interactive review script scanned the cache and
cleaned output for problem tables across two categories:

### Problem detection

1. **Empty or near-empty cache entries** (0–1 data rows) — The LLM either
   failed to parse the table or produced an implausibly small result.

2. **Text-block fallback** — The cleaned output contained the table as plain
   text (no markdown pipes), indicating the LLM call failed or was never
   attempted.

### Triage process

For each problem table, the review script displayed:

- The LLM cache entry (title, columns, row count, first few rows, footnotes)
- The raw OCR fragment count for the table
- Neighboring tables in the cleaned output (to spot "stolen data" patterns
  where a neighbor had suspiciously many rows)
- The cleaned output content (for fallback cases)

The user classified each problem into one of six categories:

| Category | Description |
|----------|-------------|
| `stolen_data` | Another table absorbed this table's data due to OCR interleaving |
| `multi_page_merge` | Table spans multiple pages and was split into separate detections |
| `llm_retry` | Raw data looks fine; retry the LLM |
| `llm_retry_with_instructions` | Retry LLM with user-provided hints about structure |
| `manual_override` | Provide a manual fix (or skip) |
| `ok` | Actually fine, no fix needed |

Results were saved to `data/intermediate/tables/table_corrections.json`.

### Applying triage fixes

**Script:** `scripts/apply_table_corrections.py`

This script read the triage classifications and applied automated fixes:

- **stolen_data:** Re-extracted raw fragments for the affected table and
  re-sent to the LLM.
- **multi_page_merge:** Found all raw regions for the table, merged their
  fragments, and re-sent the combined set to the LLM.
- **llm_retry / llm_retry_with_instructions:** Re-sent fragments to the LLM,
  optionally with extra instructions.
- **manual_override:** Injected user-provided JSON directly into the cache.

After applying fixes, the script re-ran the full cleaning pipeline to
regenerate the cleaned output files.

---

## Phase 3: Manual Interactive Correction

**Location:** `data/intermediate/tables/README.md` (workflow documentation)

The automated triage resolved some issues, but 12 tables required careful
human review that could not be handled by simple LLM retries. These were
corrected through interactive chat sessions with an AI assistant, one table
at a time.

### Workflow

1. The assistant read the raw OCR fragments and the broken LLM cache entry.
2. The assistant reconstructed the table and presented it as a markdown
   rendering.
3. The human validated the markdown against the original PDF, identified
   errors, and iterated until the table was correct.
4. The corrected table was saved as both JSON and markdown:
   - `data/intermediate/tables/table_<ID>.json` — structured data matching
     the `TableStructure` schema
   - `data/intermediate/tables/table_<ID>.md` — human-readable rendering

### Types of problems encountered

#### Empty tables (0 data rows)

Three tables had zero data rows in the LLM cache. In each case, the root
cause was OCR interleaving from the NEC's two-column page layout.

**Stolen data** was the most common pattern: when two tables appeared
side-by-side on a page, the OCR read them as interleaved fragments. The LLM
absorbed all fragments into whichever table it encountered first (the
"thief"), leaving the second table empty.

Fixing stolen data required correcting **both** tables — the empty one
(reconstructing its actual content) and the thief (removing the stolen data
from its footnotes or extra rows and restoring any of its own data that was
displaced).

**Examples:**
- Table 240.6(A) (page 127) — Its 38 ampere ratings were stolen by
  Table 240.4(G). Required reconstructing 240.6(A) from NEC knowledge
  and cleaning spurious data out of 240.4(G).
- Table 430.249 (page 366) — Severely interleaved with Table 430.248 in
  the two-column layout. Required reconstructing both tables: 430.249
  (21 rows) and 430.248 (12 rows, with corrected voltage values).
- Table 722.135(B) (page 658) — Title was split across OCR fragments
  and table data spanned 2 pages with heavily garbled OCR. Required
  full manual reconstruction (31 rows).

#### Suspicious single-row tables

Five tables had exactly 1 data row, which could indicate either a
legitimately small table or a truncated extraction.

- **Confirmed correct (1 row):** Table 424.3, Table 426.3, Table 427.3 —
  These are genuinely single-row "Other Articles" reference tables.
- **Missing rows:** Table 315.10(A) — Should have had 2 rows (MV-90 and
  MV-105*) but OCR missed the MV-90 row entirely.
- **Structural misinterpretation:** Table 610.14(D) — The LLM produced
  1 row × 3 columns, but the actual table is 3 rows × 2 columns. The OCR
  had merged separate cell values ("6", "4", "2") into a single string
  ("642").

#### Tables with formula content

Table 240.92(B) (page 135) was initially skipped during triage as
"formula-only" content. On closer inspection it contained a legitimate
two-column data table (conductor type vs. T₂ temperature) with equations
and variable definitions as footnotes. It was reconstructed with 8 data rows
and 4 footnotes containing the full equation text.

#### Large table with pervasive row-level errors

Table 310.4(1) (page 182) — "Conductor Applications and Insulations Rated
600 Volts" — is the largest table in the NEC with 233 rows across 9 columns
and 27 conductor types. The LLM's initial reconstruction contained 216 rows
but had 17 distinct correction types needed across individual conductor
entries.

This table was too complex for simple LLM retry. Instead, it was
reconstructed from scratch using a data-driven Python script
(`scripts/reconstruct_310_4_1.py`) that:

1. Defined each conductor type with its temperature/application entries,
   AWG/mm/mils triplets, and outer covering.
2. Cross-joined entries × triplets to produce flattened rows automatically.
3. Validated that every row had exactly 9 cells.

The most complex correction was for the MTW conductor type, which required a
full cross-join of 2 temperatures × 2 covering types × 8 AWG size triplets
= 32 rows. Detailed correction notes are in
`data/intermediate/tables/table_310_4_1_corrections.md`.

### Corrected tables summary

| Table ID | Page | Rows | Problem |
|----------|------|------|---------|
| Table 240.6(A) | 127 | 38 | Data stolen by Table 240.4(G) |
| Table 240.4(G) | 127 | 10 | Had stolen ampere ratings; missing own rows |
| Table 240.92(B) | 135 | 8 | Originally skipped as formula-only |
| Table 310.4(1) | 182 | 233 | 17 row-level correction types; MTW cross-join |
| Table 315.10(A) | 206 | 2 | OCR missed MV-90 row |
| Table 424.3 | 325 | 1 | Confirmed correct |
| Table 426.3 | 337 | 1 | Confirmed correct |
| Table 427.3 | 339 | 1 | Confirmed correct |
| Table 430.248 | 366 | 12 | Interleaved with 430.249; missing row, wrong voltages |
| Table 430.249 | 366 | 21 | Interleaved with 430.248 |
| Table 610.14(D) | 547 | 3 | OCR merged cell values; wrong dimensions |
| Table 722.135(B) | 658 | 31 | Split title; garbled multi-page OCR |

---

## Phase 4: Final Merge

**Script:** `scripts/merge_corrected_tables.py`

After all 12 tables were corrected and validated, the corrections were merged
back into the cleaned output files:

1. Each corrected `table_<ID>.json` file was loaded and rendered into the
   same markdown format used by the cleaning pipeline (`**Title**` + markdown
   table + blockquote footnotes).
2. The matching paragraph in `NFPA 70 NEC 2023_clean.json` was found by
   table ID and its content replaced with the corrected rendering.
3. `NFPA 70 NEC 2023_clean.txt` was regenerated from the updated JSON.

The merge script also handles Table 310.4(1) specially, loading it from
`table_310_4_1_reconstructed.json` (produced by the dedicated reconstruction
script) rather than from a `table_<ID>.json` file.

---

## File Reference

### Source modules

| File | Role |
|------|------|
| `src/nec_rag/cleaning/tables.py` | Core table detection, extraction, LLM formatting, and pipeline step |
| `src/nec_rag/cleaning/clean.py` | Pipeline orchestrator that calls `tables.run()` as step 2 |

### Scripts

| File | Role |
|------|------|
| `scripts/review_tables.py` | Interactive triage of problem tables (Phase 2) |
| `scripts/apply_table_corrections.py` | Applies triage fixes and re-runs pipeline (Phase 2) |
| `scripts/reconstruct_310_4_1.py` | Data-driven reconstruction of Table 310.4(1) (Phase 3) |
| `scripts/merge_corrected_tables.py` | Merges corrected tables into clean output files (Phase 4) |

### Data files

| File | Description |
|------|-------------|
| `data/raw/NFPA 70 NEC 2023_paragraphs.json` | Raw OCR paragraph output (input to pipeline) |
| `data/intermediate/tables/table_llm_cache.json` | Full LLM cache (217 table reconstructions) |
| `data/intermediate/tables/table_corrections.json` | Triage classifications from review script |
| `data/intermediate/tables/table_<ID>.json` | Individual corrected table (JSON, `TableStructure` schema) |
| `data/intermediate/tables/table_<ID>.md` | Individual corrected table (human-readable markdown) |
| `data/intermediate/tables/table_310_4_1_reconstructed.json` | Reconstructed Table 310.4(1) (233 rows) |
| `data/intermediate/tables/table_310_4_1_corrections.md` | Detailed correction notes for Table 310.4(1) |
| `data/intermediate/tables/table_310_4_1_review.md` | Markdown rendering of Table 310.4(1) for review |
| `data/intermediate/tables/README.md` | Interactive correction workflow and progress checklist |
| `data/intermediate/NFPA 70 NEC 2023_clean.json` | Final cleaned paragraphs (tables merged) |
| `data/intermediate/NFPA 70 NEC 2023_clean.txt` | Final cleaned plain text (tables merged) |

### Related documentation

| File | Description |
|------|-------------|
| `docs/table_ambiguity.md` | Analysis of procedural formatting ambiguity (motivates LLM approach) |
| `src/nec_rag/cleaning/README.md` | Overview of the full cleaning pipeline |
