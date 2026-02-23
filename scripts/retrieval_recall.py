"""Measure retrieval recall on the master electrician exam.

For each exam question, runs vector retrieval at multiple n_results values
and checks whether the ground-truth NEC sections/tables appear in the
retrieved chunks.  Reports the rank (position) of the first matching chunk
per question, making it easy to see whether top-10, top-20, or top-50
retrieval is needed.

Usage:
    python scripts/retrieval_recall.py
    python scripts/retrieval_recall.py --model azure-large
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # pylint: disable=wrong-import-position

load_dotenv(ROOT / ".env")

from nec_rag.agent.resources import load_embedding_resources  # pylint: disable=wrong-import-position
from nec_rag.agent.utils import _retrieve  # pylint: disable=wrong-import-position
from nec_rag.data_preprocessing.embedding.config import MODELS  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground truth: for each exam question, the NEC section/table IDs that
# contain the answer.  A retrieved chunk "matches" if its section_id starts
# with any of the listed prefixes, OR if any of the listed table IDs appear
# in its referenced_tables metadata.
#
# section_prefixes: matched against the chunk's section_id (prefix match)
# table_ids:        matched against the chunk's referenced_tables field
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "q01": {"section_prefixes": ["220.82"], "table_ids": []},
    "q02": {"section_prefixes": ["220.103"], "table_ids": ["Table220.103"]},
    "q03": {"section_prefixes": ["511.3"], "table_ids": ["Table511.3"]},
    "q04": {"section_prefixes": ["501.15"], "table_ids": []},
    "q05": {"section_prefixes": ["404.8"], "table_ids": []},
    "q06": {"section_prefixes": ["314.28", "314.16"], "table_ids": ["Table314.16"]},
    "q07": {"section_prefixes": ["300.5"], "table_ids": ["Table300.5"]},
    "q08": {"section_prefixes": ["230.26"], "table_ids": []},
    "q09": {"section_prefixes": [], "table_ids": ["TableC.9"]},  # Annex C — likely missing from dataset
    "q10": {"section_prefixes": ["250.122"], "table_ids": ["Table250.122"]},
    "q11": {"section_prefixes": ["240.24", "404.8"], "table_ids": []},
    "q12": {"section_prefixes": ["300.5"], "table_ids": []},
    "q13": {"section_prefixes": ["314.23"], "table_ids": []},
    "q14": {"section_prefixes": ["705.31"], "table_ids": []},
    "q15": {"section_prefixes": ["551.73"], "table_ids": ["Table551.73"]},
    "q16": {"section_prefixes": ["501.15"], "table_ids": []},
    "q17": {"section_prefixes": ["590.6"], "table_ids": []},
    "q18": {"section_prefixes": ["250.53"], "table_ids": []},
    "q19": {"section_prefixes": [], "table_ids": []},  # Pure calculation — no specific NEC section
    "q20": {"section_prefixes": ["300.6"], "table_ids": []},
}

# Import exam cases (question text)
sys.path.insert(0, str(ROOT / "tests_integration"))
from test_master_electrician_exam import EXAM_CASES  # pylint: disable=wrong-import-position,wrong-import-order,import-error

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _chunk_matches(chunk_meta: dict, gt: dict) -> bool:
    """Return True if a retrieved chunk matches any ground-truth reference."""
    section_id = chunk_meta.get("section_id", "")

    # Check section_id prefix match
    for prefix in gt["section_prefixes"]:
        if section_id.startswith(prefix):
            return True

    # Check referenced tables
    refs_csv = chunk_meta.get("referenced_tables", "")
    if refs_csv and gt["table_ids"]:
        chunk_tables = set(refs_csv.split(","))
        for table_id in gt["table_ids"]:
            if table_id in chunk_tables:
                return True

    return False


def _find_first_match_rank(retrieved: list[dict], gt: dict) -> int | None:
    """Return the 1-based rank of the first matching chunk, or None if no match."""
    for i, item in enumerate(retrieved):
        if _chunk_matches(item["metadata"], gt):
            return i + 1
    return None


def _find_all_match_ranks(retrieved: list[dict], gt: dict) -> list[int]:
    """Return list of 1-based ranks for all matching chunks."""
    ranks = []
    for i, item in enumerate(retrieved):
        if _chunk_matches(item["metadata"], gt):
            ranks.append(i + 1)
    return ranks


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_recall_analysis(embed_fn, collection, n_values: list[int]):
    """Run retrieval for every exam question at each n_results value and report recall."""
    max_n = max(n_values)

    # Gather results: {qid: {n: rank_or_none}}
    results: dict[str, dict] = {}
    all_match_details: dict[str, dict] = {}

    for qid, question, _ in EXAM_CASES:
        gt = GROUND_TRUTH[qid]

        # Skip questions with no ground-truth sections to check
        if not gt["section_prefixes"] and not gt["table_ids"]:
            results[qid] = {n: "n/a" for n in n_values}
            all_match_details[qid] = {"gt": gt, "matches": [], "retrieved_sections": []}
            continue

        # Retrieve once at the maximum n, then slice for smaller n values
        retrieved = _retrieve(question, embed_fn, collection, n_results=max_n)

        # Record detailed info for the full retrieval
        all_ranks = _find_all_match_ranks(retrieved, gt)
        retrieved_sections = [(item["metadata"]["section_id"], item["distance"]) for item in retrieved]
        all_match_details[qid] = {"gt": gt, "matches": all_ranks, "retrieved_sections": retrieved_sections}

        # Evaluate at each n_results cutoff
        row = {}
        for n in n_values:
            subset = retrieved[:n]
            rank = _find_first_match_rank(subset, gt)
            row[n] = rank
        results[qid] = row

    return results, all_match_details


def print_summary_table(results: dict, n_values: list[int]):
    """Print the main recall table."""
    n_headers = "  ".join(f"{'n=' + str(n):>7s}" for n in n_values)
    print(f"\n{'QID':>4s}  {n_headers}   Ground Truth")
    print("─" * (8 + 9 * len(n_values) + 40))

    for qid, _, _ in EXAM_CASES:
        gt = GROUND_TRUTH[qid]
        gt_label = ", ".join(gt["section_prefixes"] + gt["table_ids"]) or "(calculation)"

        row_vals = []
        for n in n_values:
            val = results[qid][n]
            if val == "n/a":
                row_vals.append("   n/a ")
            elif val is None:
                row_vals.append("  MISS ")
            else:
                row_vals.append(f"  #{val:<4d} ")
        row_str = "  ".join(row_vals)
        print(f" {qid:>3s}  {row_str}   {gt_label}")

    # Recall summary
    print()
    for n in n_values:
        hits = sum(1 for qid in results if isinstance(results[qid][n], int))
        misses = sum(1 for qid in results if results[qid][n] is None)
        skipped = sum(1 for qid in results if results[qid][n] == "n/a")
        total_with_gt = hits + misses
        recall_pct = (hits / total_with_gt * 100) if total_with_gt > 0 else 0
        print(f"  n={n:<3d}  recall = {hits}/{total_with_gt} ({recall_pct:.0f}%)  |  {misses} misses, {skipped} skipped (no ground truth)")


def print_miss_details(results: dict, all_details: dict, n_values: list[int]):
    """For questions that miss at the largest n, print what was retrieved instead."""
    max_n = max(n_values)
    misses = [qid for qid, _, _ in EXAM_CASES if results[qid][max_n] is None]

    if not misses:
        print(f"\nNo misses at n={max_n} — all ground-truth sections were retrieved.")
        return

    print(f"\n{'=' * 80}")
    print(f"MISS DETAILS (questions where ground truth was not found in top {max_n})")
    print(f"{'=' * 80}")

    for qid in misses:
        detail = all_details[qid]
        gt = detail["gt"]
        question = next(q for q_id, q, _ in EXAM_CASES if q_id == qid)

        print(f"\n--- {qid} ---")
        print(f"  Question: {question[:120]}...")
        print(f"  Looking for: sections={gt['section_prefixes']}, tables={gt['table_ids']}")
        print("  Top 10 retrieved sections:")
        for i, (sec_id, dist) in enumerate(detail["retrieved_sections"][:10]):
            print(f"    [{i + 1:2d}] {sec_id:<30s}  dist={dist:.4f}")


def print_match_distribution(results: dict, n_values: list[int]):
    """Print a histogram of where matches land in the ranking."""
    max_n = max(n_values)
    ranks = [results[qid][max_n] for qid in results if isinstance(results[qid][max_n], int)]

    if not ranks:
        return

    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50)]
    print(f"\nMatch rank distribution (first match, n={max_n}):")
    for lo, hi in buckets:
        if hi > max_n:
            break
        count = sum(1 for r in ranks if lo <= r <= hi)
        histogram_bar = "█" * count
        print(f"  #{lo:>2d}-{hi:<2d}:  {count:>2d}  {histogram_bar}")


def main():
    """Run the retrieval recall analysis."""
    parser = argparse.ArgumentParser(description="Measure retrieval recall on master electrician exam")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="azure-large", help="Embedding model (default: azure-large)")
    parser.add_argument("--n-values", type=str, default="5,10,20,30,50", help="Comma-separated n_results values to test (default: 5,10,20,30,50)")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(",")]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Loading embedding resources (model=%s)...", args.model)
    embed_fn, collection = load_embedding_resources(args.model)
    logger.info("Collection loaded: %d chunks", collection.count())

    logger.info("Running retrieval for %d questions at n_values=%s", len(EXAM_CASES), n_values)
    t0 = time.perf_counter()
    results, all_details = run_recall_analysis(embed_fn, collection, n_values)
    elapsed = time.perf_counter() - t0
    logger.info("Analysis complete in %.1f seconds", elapsed)

    print_summary_table(results, n_values)
    print_match_distribution(results, n_values)
    print_miss_details(results, all_details, n_values)


if __name__ == "__main__":
    main()
