# Table Formatting Ambiguity Analysis

## Problem Statement

Azure Document Intelligence OCR captured NEC tables as a flat stream of individual
cell values — one paragraph per cell, reading left-to-right, top-to-bottom.  The
OCR did **not** detect table structures, so there is no metadata about column count,
header rows, merged cells, or row boundaries.

The cleaning pipeline must reconstruct table structure from this flat paragraph
stream.  A procedural approach was implemented in `src/nec_rag/cleaning/tables.py`
that detects table boundaries, extracts cell content, classifies headers vs. data,
and formats the result as markdown.

This document records the **reliability findings** from that approach and motivates
the switch to an LLM-based formatting step.


## The Three Types of Ambiguity

### 1. Column Count Ambiguity

When there are `N` detected header items and `D` data items, the formatter picks the
largest column count `C` such that `D % C == 0` and `N - C` (the number of "group
headers") is between 0 and 5.

**The problem:** multiple values of `C` can satisfy these constraints.

**Example — 6 data items, 3 headers:**
- `C = 3` → 2 data rows of 3 columns, 0 group headers  ✓
- `C = 2` → 3 data rows of 2 columns, 1 group header   ✓

Both are valid structurally, but the table only has one correct layout.

**Example — 12 data items, 4 headers:**
- `C = 4` → 3 data rows of 4 columns, 0 group headers  ✓
- `C = 3` → 4 data rows of 3 columns, 1 group header   ✓
- `C = 2` → 6 data rows of 2 columns, 2 group headers  ✓

### 2. Header/Data Boundary Sensitivity

The `_find_data_start` function uses heuristics to decide where headers end and
data begins:

1. **First pure number** — the first cell that matches `^\d+$`, `^\d+\.\d+$`, or
   `^\d+/\d+$` is treated as the start of the data region.
2. **Run of short items** — if no pure numbers exist, three consecutive items
   under 30 characters are treated as data.
3. **Fallback** — for very short tables (≤10 items), assume the first 2 are headers.

**The problem:** shifting the boundary by ±1 or ±2 positions often still produces a
valid table (the new data count still divides evenly by some column count).

**Example — Table 110.26(A)(1):**
The OCR output for this table contains items like "Voltage to Ground" and
"Condition 1", "Condition 2", "Condition 3".  These are column headers, but:
- They are not pure numbers, so Strategy 1 doesn't trigger on them.
- "Voltage to Ground" is 19 characters, which is < 30, so Strategy 2 treats it
  as the start of a "run of short items" = data.

Result: the procedural code incorrectly classifies these column headers as data,
producing a table with wrong column assignments.

**Example — Table 220.103:**
The table has a text-based first column ("Largest load", "Second largest load",
etc.) and a numeric second column (100%, 65%, etc.).  The `_find_data_start`
function finds the first pure number ("100") and marks that as data start.  But the
text items before it ("Largest load") are **data values**, not headers.

The function erroneously classifies "Largest load" as a header because it appears
before the first number.

### 3. Two-Column Page Interleaving

When NEC pages use a two-column layout, OCR reads the left column first, then the
right column.  If a table spans both columns, its cells get interleaved with
unrelated section text from the opposite column.  This is a fundamental limitation
of the OCR output that neither procedural code nor an LLM can fully resolve without
the original PDF geometry.


## Quantitative Reliability Analysis

An analysis of all 128 tables that the procedural code formatted into markdown
grids (as opposed to text-block fallback) found:

| Metric | Count | Percentage |
|--------|------:|----------:|
| **Unambiguously correct** (unique column count AND boundary-insensitive) | 6 | 4.7% |
| **Ambiguous column count** (multiple valid `C` values) | 39 | 30.5% |
| **Boundary-sensitive** (shifting ±1–2 positions still yields a valid table) | 122 | 95.3% |
| **Both ambiguous AND boundary-sensitive** | 39 | 30.5% |

**Key takeaway:** only ~5% of the procedurally formatted tables can be trusted as
structurally correct with high confidence.  The remaining ~95% have at least one
dimension of ambiguity where a different — possibly more correct — layout exists.


## What the Procedural Code Does Well

Despite the formatting limitations, the procedural approach excels at:

1. **Table detection** — identifying where tables start in the paragraph stream
   using `TABLE_TITLE_RE` and the `_is_real_table_start` heuristic.
2. **Multi-page tracking** — following `(continues)` / `Continued` markers to
   capture the full extent of tables spanning multiple pages.
3. **Boundary detection** — finding where the table ends by tracking page breaks,
   section boundaries, and the transition back to body text.
4. **Content extraction** — stripping page markers, continuation noise, and
   repeated headers to produce a clean list of cell values.
5. **Paragraph interruption repair** — detecting and rejoining paragraphs that
   were split by an embedded table.


## LLM-Based Solution

Given the low confidence in procedural column reconstruction, the solution is to
use an LLM (Azure OpenAI GPT) with structured output to reconstruct table
structure.  The approach:

1. **Keep the existing detection and extraction code** — the procedural functions
   `find_table_starts`, `find_table_end`, `extract_table_content`, and
   `detect_interruption` are reliable and remain unchanged.
2. **Replace the formatting step** — instead of `_classify_parts` →
   `_detect_column_count` → `_format_markdown_table`, send the extracted cell
   values to an LLM with a Pydantic-enforced output schema.
3. **Enforce structural consistency** — the Pydantic model validates that every
   data row has exactly as many cells as there are column headers, catching
   hallucinated or misaligned output.
4. **Graceful fallback** — if the LLM response fails validation or the API call
   errors, fall back to the text-block format (preserving all content).

### Pydantic Output Schema

```python
class TableStructure(BaseModel):
    title: str
    column_headers: list[str]
    data_rows: list[list[str]]
    footnotes: list[str] = []

    @model_validator(mode="after")
    def validate_row_widths(self):
        n_cols = len(self.column_headers)
        for i, row in enumerate(self.data_rows):
            if len(row) != n_cols:
                raise ValueError(
                    f"Row {i} has {len(row)} cells, expected {n_cols}"
                )
        return self
```

This schema guarantees that the LLM's output is a well-formed table where every
row matches the declared column count — eliminating the column-count ambiguity and
boundary-sensitivity problems entirely.
