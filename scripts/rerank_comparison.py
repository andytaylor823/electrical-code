"""Compare retrieval ranking before and after cross-encoder re-ranking.

For each master electrician exam question, runs vector retrieval to get the
top-N chunks, then re-ranks them with a cross-encoder model.  Reports the
rank of the ground-truth section both before and after re-ranking so you can
see whether re-ranking promotes the correct answer higher in the list.

Usage:
    python scripts/rerank_comparison.py
    python scripts/rerank_comparison.py --model azure-large --top-n 50
    python scripts/rerank_comparison.py --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

# Ensure project root is importable
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # pylint: disable=wrong-import-position

load_dotenv(ROOT / ".env")

from nec_rag.agent.resources import load_embedding_resources  # pylint: disable=wrong-import-position
from nec_rag.agent.utils import _retrieve  # pylint: disable=wrong-import-position
from nec_rag.data_preprocessing.embedding.config import MODELS  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Ground truth (mirrored from retrieval_recall.py)
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
    "q09": {"section_prefixes": [], "table_ids": ["TableC.9"]},
    "q10": {"section_prefixes": ["250.122"], "table_ids": ["Table250.122"]},
    "q11": {"section_prefixes": ["240.24", "404.8"], "table_ids": []},
    "q12": {"section_prefixes": ["300.5"], "table_ids": []},
    "q13": {"section_prefixes": ["314.23"], "table_ids": []},
    "q14": {"section_prefixes": [], "table_ids": []},  # Section 705.31 removed in 2023 NEC — unanswerable
    "q15": {"section_prefixes": ["551.73"], "table_ids": ["Table551.73"]},
    "q16": {"section_prefixes": ["501.15"], "table_ids": []},
    "q17": {"section_prefixes": ["590.6"], "table_ids": []},
    "q18": {"section_prefixes": ["250.53"], "table_ids": []},
    "q19": {"section_prefixes": [], "table_ids": []},
    "q20": {"section_prefixes": ["300.6", "312.2"], "table_ids": []},
}

# Import exam cases (question text)
sys.path.insert(0, str(ROOT / "tests_integration" / "agent"))
from test_master_electrician_exam import EXAM_CASES  # pylint: disable=wrong-import-position,wrong-import-order,import-error

# ---------------------------------------------------------------------------
# Matching helpers (shared with retrieval_recall.py)
# ---------------------------------------------------------------------------


def _chunk_matches(chunk_meta: dict, gt: dict) -> bool:
    """Return True if a retrieved chunk matches any ground-truth reference."""
    section_id = chunk_meta.get("section_id", "")

    # Check section_id prefix match (subsection chunks)
    for prefix in gt["section_prefixes"]:
        if section_id.startswith(prefix):
            return True

    # Check table ID matches — handles both table chunks (where section_id IS
    # the table ID) and subsection chunks (via referenced_tables metadata)
    if gt["table_ids"]:
        if section_id in gt["table_ids"]:
            return True
        refs_csv = chunk_meta.get("referenced_tables", "")
        if refs_csv:
            chunk_tables = set(refs_csv.split(","))
            for table_id in gt["table_ids"]:
                if table_id in chunk_tables:
                    return True

    return False


def _find_first_match_rank(items: list[dict], gt: dict) -> int | None:
    """Return the 1-based rank of the first matching chunk, or None."""
    for i, item in enumerate(items):
        if _chunk_matches(item["metadata"], gt):
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking
# ---------------------------------------------------------------------------


def rerank_with_cross_encoder(question: str, retrieved: list[dict], cross_encoder: CrossEncoder) -> list[dict]:
    """Re-rank retrieved chunks using the cross-encoder and return sorted list.

    Each item gains a 'ce_score' field with the cross-encoder relevance score.
    """
    # Build (query, passage) pairs for the cross-encoder
    pairs = [(question, item["document"]) for item in retrieved]

    # Score all pairs in a single batch
    scores = cross_encoder.predict(pairs)

    # Attach scores to items
    for item, score in zip(retrieved, scores):
        item["ce_score"] = float(score)

    # Sort descending by cross-encoder score (most relevant first)
    reranked = sorted(retrieved, key=lambda x: x["ce_score"], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------


def run_rerank_comparison(embed_fn, collection, cross_encoder: CrossEncoder, top_n: int):
    """Retrieve and re-rank for every exam question, returning per-question results."""
    results = []

    for qid, question, _ in EXAM_CASES:
        gt = GROUND_TRUTH[qid]

        # Skip questions with no ground-truth sections to check
        if not gt["section_prefixes"] and not gt["table_ids"]:
            results.append({"qid": qid, "skipped": True, "original_rank": None, "reranked_rank": None, "gt": gt})
            continue

        # Retrieve with embedding similarity
        retrieved = _retrieve(question, embed_fn, collection, n_results=top_n)

        # Find rank in original embedding order
        original_rank = _find_first_match_rank(retrieved, gt)

        # Re-rank with cross-encoder
        reranked = rerank_with_cross_encoder(question, retrieved, cross_encoder)
        reranked_rank = _find_first_match_rank(reranked, gt)

        # Capture the top-5 original and reranked for debugging
        orig_top5 = [(item["metadata"]["section_id"], item["distance"]) for item in retrieved[:5]]
        reranked_top5 = [(item["metadata"]["section_id"], item["ce_score"]) for item in reranked[:5]]

        results.append(
            {
                "qid": qid,
                "skipped": False,
                "original_rank": original_rank,
                "reranked_rank": reranked_rank,
                "gt": gt,
                "orig_top5": orig_top5,
                "reranked_top5": reranked_top5,
            }
        )

    return results


def _fmt_rank(rank: int | None) -> str:
    """Format a rank value as a string for display."""
    if rank is None:
        return "MISS"
    return f"#{rank}"


def _classify_rank_change(orig: int | None, reranked: int | None) -> tuple[str, str]:
    """Return (change_str, category) for a before/after rank pair.

    category is one of 'improved', 'worsened', or 'unchanged'.
    Both-MISS counts as unchanged (still missed after re-ranking).
    """
    if orig is not None and reranked is not None:
        delta = orig - reranked
        if delta > 0:
            return f"+{delta:d}", "improved"
        if delta < 0:
            return f"{delta:d}", "worsened"
        return "=", "unchanged"
    if orig is None and reranked is not None:
        return "NEW", "improved"
    if orig is not None and reranked is None:
        return "LOST", "worsened"
    return "-", "unchanged"


def _print_summary_stats(results: list[dict]):
    """Print aggregate summary statistics from the comparison results."""
    improved = worsened = unchanged = 0
    total_orig = total_rerank = questions_with_both = promoted_to_top5 = 0

    for r in results:
        if r["skipped"]:
            continue
        orig, reranked = r["original_rank"], r["reranked_rank"]
        _, category = _classify_rank_change(orig, reranked)

        if category == "improved":
            improved += 1
        elif category == "worsened":
            worsened += 1
        elif category == "unchanged":
            unchanged += 1

        if orig is not None and reranked is not None:
            total_orig += orig
            total_rerank += reranked
            questions_with_both += 1
            if orig > 5 and reranked <= 5:  # pylint: disable=chained-comparison
                promoted_to_top5 += 1

    print(f"\n{'─' * 90}")
    print("SUMMARY")
    print(f"  Improved:  {improved:>3d}  (re-ranking promoted the correct section higher)")
    print(f"  Worsened:  {worsened:>3d}  (re-ranking pushed the correct section lower)")
    print(f"  Unchanged: {unchanged:>3d}")
    if questions_with_both > 0:
        avg_orig = total_orig / questions_with_both
        avg_rerank = total_rerank / questions_with_both
        print(f"\n  Avg rank  (embedding):   {avg_orig:.1f}")
        print(f"  Avg rank  (reranked):    {avg_rerank:.1f}")
        print(f"  Avg improvement:         {avg_orig - avg_rerank:+.1f} positions")
    if promoted_to_top5 > 0:
        print(f"\n  Promoted into top-5:     {promoted_to_top5}")


def print_comparison_table(results: list[dict], top_n: int):
    """Print the side-by-side ranking comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  Cross-Encoder Re-Ranking Comparison  (top_n={top_n})")
    print(f"{'=' * 90}")
    print(f"\n {'QID':>4s}  {'Embedding':>10s}  {'Reranked':>10s}  {'Change':>8s}   Ground Truth")
    print("─" * 90)

    for r in results:
        qid = r["qid"]
        gt_label = ", ".join(r["gt"]["section_prefixes"] + r["gt"]["table_ids"]) or "(calculation)"

        if r["skipped"]:
            print(f" {qid:>3s}       n/a         n/a       n/a    {gt_label}")
            continue

        orig, reranked = r["original_rank"], r["reranked_rank"]
        change_str, _ = _classify_rank_change(orig, reranked)
        print(f" {qid:>3s}  {_fmt_rank(orig):>10s}  {_fmt_rank(reranked):>10s}  {change_str:>8s}   {gt_label}")

    _print_summary_stats(results)


def print_detail_view(results: list[dict]):
    """Print per-question detail showing top-5 sections before/after re-ranking."""
    print(f"\n{'=' * 90}")
    print("  DETAIL: Top-5 sections before and after re-ranking")
    print(f"{'=' * 90}")

    for r in results:
        if r["skipped"]:
            continue

        orig = r["original_rank"]
        reranked = r["reranked_rank"]

        # Only show detail for questions where ranking changed noticeably
        if orig == reranked and orig is not None and orig <= 5:
            continue

        gt_label = ", ".join(r["gt"]["section_prefixes"] + r["gt"]["table_ids"])
        print(f"\n--- {r['qid']} (looking for: {gt_label}) ---")
        print(f"  Original: {_fmt_rank(orig)}  →  Reranked: {_fmt_rank(reranked)}")

        print("  Embedding order (top 5):")
        for i, (sec_id, dist) in enumerate(r.get("orig_top5", [])):
            marker = " ✓" if any(sec_id.startswith(p) for p in r["gt"]["section_prefixes"]) else ""
            print(f"    [{i + 1}] {sec_id:<30s}  dist={dist:.4f}{marker}")

        print("  Reranked order (top 5):")
        for i, (sec_id, score) in enumerate(r.get("reranked_top5", [])):
            marker = " ✓" if any(sec_id.startswith(p) for p in r["gt"]["section_prefixes"]) else ""
            print(f"    [{i + 1}] {sec_id:<30s}  score={score:.4f}{marker}")


def main():
    """Run the cross-encoder re-ranking comparison."""
    parser = argparse.ArgumentParser(description="Compare retrieval ranking before/after cross-encoder re-ranking")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="azure-large", help="Embedding model (default: azure-large)")
    parser.add_argument("--cross-encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model name from HuggingFace")
    parser.add_argument("--top-n", type=int, default=50, help="Number of chunks to retrieve before re-ranking (default: 50)")
    parser.add_argument("--detail", action="store_true", help="Show per-question top-5 detail")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load embedding resources
    logger.info("Loading embedding resources (model=%s)...", args.model)
    embed_fn, collection = load_embedding_resources(args.model)
    logger.info("Collection loaded: %d chunks", collection.count())

    # Load cross-encoder model
    logger.info("Loading cross-encoder model '%s'...", args.cross_encoder)
    cross_encoder = CrossEncoder(args.cross_encoder, activation_fn=torch.nn.Sigmoid())
    logger.info("Cross-encoder loaded.")

    # Run comparison
    logger.info("Running retrieval + re-ranking for %d questions (top_n=%d)...", len(EXAM_CASES), args.top_n)
    t0 = time.perf_counter()
    results = run_rerank_comparison(embed_fn, collection, cross_encoder, args.top_n)
    elapsed = time.perf_counter() - t0
    logger.info("Comparison complete in %.1f seconds", elapsed)

    # Display results
    print_comparison_table(results, args.top_n)

    if args.detail:
        print_detail_view(results)
    else:
        # Always show detail for significant changes
        significant = [r for r in results if not r["skipped"] and r["original_rank"] != r["reranked_rank"]]
        if significant:
            print_detail_view(results)


if __name__ == "__main__":
    main()
