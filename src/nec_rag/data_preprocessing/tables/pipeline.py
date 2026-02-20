"""Main table-cleaning pipeline step and dict utilities.

Orchestrates the full table-detection-and-formatting pass over a paragraph
dict.  This is Phase 1 of the table cleaning process described in
docs/table_cleaning.md.

Pipeline position: runs AFTER remove_junk_pages but BEFORE sentence_runover,
because raw table content scattered across paragraphs confuses the sentence-
merge heuristic.
"""

import json
import logging
from pathlib import Path

from nec_rag.data_preprocessing.tables.classifiers import get_table_id
from nec_rag.data_preprocessing.tables.detection import (
    detect_interruption,
    extract_table_content,
    find_table_end,
    find_table_starts,
)
from nec_rag.data_preprocessing.tables.formatting import format_table

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()


# ─── Dict Utility ────────────────────────────────────────────────────────────


def resort_dict(d: dict[str, dict]) -> dict[str, dict]:
    """Re-index a dict with consecutive integer string keys."""
    sorted_items = sorted(d.items(), key=lambda item: int(item[0]))
    return {str(i): val for i, (_, val) in enumerate(sorted_items)}


# ─── Main Pipeline Step ──────────────────────────────────────────────────────


def run(paragraphs: dict[str, dict]) -> dict[str, dict]:
    """Detect tables, format them via LLM as markdown, and repair interrupted paragraphs.

    Returns a new paragraphs dict where every table region has been collapsed
    into a single paragraph containing markdown-formatted content.

    This is Phase 1 of the table cleaning process.  The output may still
    contain broken tables (empty rows, interleaved data, structural errors)
    which are addressed in Phases 2-4.  See docs/table_cleaning.md.
    """
    n = len(paragraphs)

    # ── 1. Locate every genuine table start ──────────────────────────────
    table_starts = find_table_starts(paragraphs)
    logger.info("Detected %d table starts", len(table_starts))

    # ── 2. Build table info (end idx, interruption, formatted content) ───
    table_info: list[dict] = []
    for idx, start in enumerate(table_starts):
        next_start = table_starts[idx + 1] if idx + 1 < len(table_starts) else None
        end = find_table_end(paragraphs, start, next_start)
        pre_idx, post_idx = detect_interruption(paragraphs, start, end)
        content_parts = extract_table_content(paragraphs, start, end)

        # Progress logging (LLM calls can be slow on first run)
        logger.info(
            "Formatting table %d/%d: %s (%d fragments)",
            idx + 1,
            len(table_starts),
            get_table_id(content_parts[0]) if content_parts else "unknown",
            len(content_parts),
        )

        table_info.append(
            {
                "table_id": get_table_id(content_parts[0]) if content_parts else "unknown",
                "start": start,
                "end": end,
                "pre_idx": pre_idx,
                "post_idx": post_idx,
                "formatted": format_table(content_parts[0] if content_parts else "", content_parts),
            }
        )

    # ── 3. Build skip-set and merge-map ──────────────────────────────────
    skip_set: set[int] = set()
    merge_map: dict[int, str] = {}  # pre_idx → merged content

    for info in table_info:
        # All paragraphs inside the table region are replaced by the formatted block
        for i in range(info["start"], info["end"] + 1):
            skip_set.add(i)

        # If the table interrupted a paragraph, also skip everything from
        # the table end to the continuation (page markers + the continuation itself)
        if info["post_idx"] is not None:
            for i in range(info["end"] + 1, info["post_idx"] + 1):
                skip_set.add(i)
            pre_content = paragraphs[str(info["pre_idx"])]["content"]
            post_content = paragraphs[str(info["post_idx"])]["content"]
            merge_map[info["pre_idx"]] = pre_content + " " + post_content

    # ── 4. Rebuild the paragraph dict ────────────────────────────────────
    table_start_map = {info["start"]: info for info in table_info}
    emitted_table_ids: set[str] = set()  # Track which table IDs already emitted (dedup)
    output: dict[str, dict] = {}
    out_idx = 0

    for i in range(n):
        if i in skip_set:
            # Emit the formatted table at the start of its region (once per table ID)
            if i in table_start_map:
                info = table_start_map[i]
                tid = info["table_id"]
                if tid not in emitted_table_ids:
                    emitted_table_ids.add(tid)
                    output[str(out_idx)] = {
                        "content": info["formatted"],
                        "page": paragraphs[str(info["start"])]["page"],
                    }
                    out_idx += 1
                else:
                    logger.debug("Dedup: skipping duplicate emission for %s at idx %d", tid, i)
            # All other paragraphs in the region are dropped
            continue

        if i in merge_map:
            # Emit the merged (repaired) paragraph
            output[str(out_idx)] = {
                "content": merge_map[i],
                "page": paragraphs[str(i)]["page"],
            }
            out_idx += 1
        else:
            # Regular paragraph — copy unchanged
            output[str(out_idx)] = paragraphs[str(i)].copy()
            out_idx += 1

    logger.info("Rebuilt paragraph dict: %d -> %d entries", n, out_idx)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Read raw paragraphs
    paragraphs_file = ROOT / "data" / "raw" / "NFPA 70 NEC 2023_paragraphs.json"
    with open(paragraphs_file, "r", encoding="utf-8") as fopen:
        raw_paragraphs = json.load(fopen)

    # Run cleaning
    result = run(raw_paragraphs)
    logger.info("Done: %d paragraphs in, %d out", len(raw_paragraphs), len(result))
