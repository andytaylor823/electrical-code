# Cleaning Pipeline

Post-OCR cleaning pipeline for the NFPA 70 National Electrical Code (2023 Edition). The raw OCR output contains structural artifacts from PDF extraction — junk pages, broken words, and sentences split across page boundaries. This module applies a sequence of targeted fixes to produce clean, contiguous text suitable for downstream embedding and retrieval.

## Input

`data/intermediate/NFPA 70 NEC 2023_paragraphs.json` — raw paragraph-level JSON produced by the OCR step. Each entry is keyed by a string index and contains:

```json
{
  "0": { "content": "paragraph text...", "page": 26 },
  "1": { "content": "more text...", "page": 26 }
}
```

## Pipeline Steps

The pipeline is orchestrated by `clean.py` (steps 1–4) followed by a separate structuring step (step 5). Steps 1–4 run **in order**:

### 1. Remove Junk Pages (`remove_junk_pages.py`)

Drops all paragraphs outside the main NEC content range (pages 26–717, 1-indexed). This strips the cover, table of contents, appendices, index, and other non-code material. After filtering, paragraph keys are re-indexed to consecutive integers.

### 2. Detect and Format Tables (`tables.py`)

The OCR did not detect tables as structured objects — individual cells were captured as separate paragraphs in reading order. This step:

- **Detects table boundaries** by finding paragraphs that start with `Table X.Y` and scanning forward to locate the end of each table region (next section, next table, or transition out of short/numeric data).
- **Handles multi-page tables** that use `(continues)` / `Continued` markers by merging the continuation into the same table.
- **Repairs interrupted paragraphs** — when a table appears in the middle of a sentence (due to page layout), the split halves are re-joined and the table is placed as its own paragraph.
- **Formats table content** as markdown where column structure is detectable (headers + evenly-divisible data). Falls back to a clearly-labelled text block for complex tables.

This step runs *before* sentence merging so that scattered table cells don't confuse the runover heuristic.

### 3. Merge Sentence Runovers (`sentence_runover.py`)

Detects sentences that were split across page boundaries during OCR. When the last paragraph on one page and the first paragraph on the next page form a single continuous sentence, they are merged into one entry and assigned to the earlier page. Detection uses a set of heuristics:

- The first page's text does **not** end with sentence-terminating punctuation (`.`, `?`, `!`).
- The second page's text does **not** begin with a section number (e.g. `290.98`), a structural keyword (`Informational`, `Part`, `Table`, `Figure`, `(`), or all-caps text (headers).
- Multi-line content (e.g. formatted markdown tables) is never treated as a runover.

After merging, paragraph keys are re-indexed again.

### 4. Fix End-of-Line Hyphenation (`hyphens_endline.py`)

Removes hyphenation artifacts introduced by line breaks in the source PDF — for example, `electri- cal` becomes `electrical`. A regex matches any letter followed by `- ` (hyphen-space) and collapses the break by removing the hyphen and space.

### 5. Remove Page Furniture (`remove_page_furniture.py`)

Strips page headers, footers, page numbers, copyright lines, watermarks, and bare section-number repeats that the OCR captured at page boundaries. Runs *after* sentence merging because the runover heuristic relies on some of these markers to detect page boundaries.

### 6. Structure into Hierarchy (`structure.py`)

Parses the flat cleaned paragraphs into a nested JSON tree with four levels:

- **Chapter** (e.g. Chapter 2 Wiring and Protection)
- **Article** (e.g. Article 200 Use and Identification of Grounded Conductors)
- **Part** (e.g. Part I. General — implicit if the article has no explicit parts)
- **Subsection** (e.g. 200.6 Means of Identifying Grounded Conductors)

Within each subsection, content is split into **front_matter** (text before the first `(A)`/`(B)`/`(C)` marker) and **sub_items** (each lettered/numbered item with its continuation paragraphs).

Tables are parsed from markdown back into structured form (`id`, `title`, `column_headers`, `data_rows`, `footnotes`) and stored at the **Article** level.

Definitions (Article 100) are collected as a flat top-level list.

## Output

Steps 1–5 produce intermediate files in `data/intermediate/`:

| File | Description |
|---|---|
| `NFPA 70 NEC 2023_clean.json` | Cleaned paragraphs with page numbers, same schema as the input. |
| `NFPA 70 NEC 2023_clean.txt` | Plain-text concatenation of all cleaned paragraphs (non-ASCII characters stripped). |

Step 6 produces the final structured output in `data/prepared/`:

| File | Description |
|---|---|
| `NFPA 70 NEC 2023_structured.json` | Nested hierarchical JSON (Chapter > Article > Part > Subsection) with parsed tables and definitions. |

## Usage

Run the cleaning pipeline (steps 1–5) from the project root:

```bash
python -m nec_rag.data_preprocessing.text_cleaning.clean
```

Run the structuring step (step 6) separately:

```bash
python -m nec_rag.data_preprocessing.text_cleaning.structure
```

Each sub-step can also be run independently via its own `__main__` block for debugging or development.
