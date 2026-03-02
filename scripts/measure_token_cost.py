"""Measure per-request token usage and USD cost for the NEC agent.

Sends a handful of representative queries through the agent, records
prompt / completion / total tokens and LLM call count for each, then
prints a summary table and a $20/day budget projection.

Usage:
    source .venv/bin/activate
    python scripts/measure_token_cost.py
"""

import logging
import time

from langchain_community.callbacks import get_openai_callback

from nec_rag.agent.agent import build_nec_agent
from nec_rag.agent.tools import reset_seen_sections, reset_vision_usage, get_vision_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Pricing aligned with traces.py defaults (Azure OpenAI GPT-5-mini)
PRICE_INPUT_PER_1K = 0.00025  # USD per 1,000 input tokens ($0.25/1M)
PRICE_OUTPUT_PER_1K = 0.002  # USD per 1,000 output tokens ($2.00/1M)
PRICE_EMBED_PER_1K = 0.00013  # USD per 1,000 embedding tokens (text-embedding-3-large)

SAMPLE_QUERIES = [
    # Simple lookup — should use 1 tool call (nec_lookup or rag_search)
    "What is the minimum burial depth for rigid metal conduit under a building?",
    # Moderate — likely rag_search + nec_lookup
    "What are the GFCI requirements for temporary construction sites?",
    # Calculation-heavy — multiple tool calls
    "Calculate the service entrance conductor size for a 200A residential service using copper THWN conductors.",
    # Broad exploratory — may trigger browse_nec_structure + rag_search
    "What NEC articles cover hazardous locations and how are they classified?",
    # Multi-part — should need several lookups
    "What is the maximum distance for conduit seals from an enclosure in a Class I Division 1 location, and what are the exceptions?",
]

DAILY_BUDGET_USD = 20.0


def main():
    """Run sample queries and measure token costs."""
    logger.info("Building NEC agent...")
    agent = build_nec_agent()
    logger.info("Agent ready. Sending %d sample queries...\n", len(SAMPLE_QUERIES))

    results = []

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        reset_vision_usage()
        reset_seen_sections()

        logger.info("--- Query %d/%d ---", i, len(SAMPLE_QUERIES))
        logger.info("Q: %s", query)

        start = time.time()
        with get_openai_callback() as cb:
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        elapsed = time.time() - start

        vision = get_vision_usage()
        prompt_tokens = cb.prompt_tokens + vision["prompt_tokens"]
        completion_tokens = cb.completion_tokens + vision["completion_tokens"]
        total_tokens = cb.total_tokens + vision["total_tokens"]
        llm_calls = cb.successful_requests

        # Extract reasoning tokens from the last message metadata
        final_msg = result["messages"][-1]
        reasoning_tokens = 0
        token_usage = getattr(final_msg, "response_metadata", {}).get("token_usage", {})
        completion_details = token_usage.get("completion_tokens_details", {})
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
        elif hasattr(completion_details, "reasoning_tokens"):
            reasoning_tokens = completion_details.reasoning_tokens or 0

        cost_llm = (prompt_tokens / 1000) * PRICE_INPUT_PER_1K + (completion_tokens / 1000) * PRICE_OUTPUT_PER_1K
        # Rough embedding cost: ~30 tokens per rag_search query, assume 1-2 rag_search calls
        cost_embed = (50 / 1000) * PRICE_EMBED_PER_1K * 2  # negligible but included

        results.append(
            {
                "query": query[:80],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "llm_calls": llm_calls,
                "cost_llm": cost_llm,
                "cost_embed": cost_embed,
                "cost_total": cost_llm + cost_embed,
                "latency_s": elapsed,
            }
        )

        logger.info(
            "Tokens: %d prompt + %d completion (%d reasoning) = %d total | %d LLM calls | $%.4f | %.1fs",
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            total_tokens,
            llm_calls,
            cost_llm,
            elapsed,
        )

    # Print summary table
    print("\n" + "=" * 120)
    print("TOKEN USAGE & COST SUMMARY")
    print("=" * 120)
    print(f"{'#':<3} {'Prompt':>10} {'Completion':>12} {'Reasoning':>10} {'Total':>10} {'Calls':>6} {'Cost':>8} {'Time':>7}  Query")
    print("-" * 120)

    total_prompt = 0
    total_completion = 0
    total_reasoning = 0
    total_all = 0
    total_cost = 0.0
    total_time = 0.0

    for i, r in enumerate(results, 1):
        print(
            f"{i:<3} {r['prompt_tokens']:>10,} {r['completion_tokens']:>12,} {r['reasoning_tokens']:>10,} "
            f"{r['total_tokens']:>10,} {r['llm_calls']:>6} {r['cost_llm']:>7.4f}  {r['latency_s']:>6.1f}s  {r['query']}"
        )
        total_prompt += r["prompt_tokens"]
        total_completion += r["completion_tokens"]
        total_reasoning += r["reasoning_tokens"]
        total_all += r["total_tokens"]
        total_cost += r["cost_llm"]
        total_time += r["latency_s"]

    n = len(results)
    avg_prompt = total_prompt / n
    avg_completion = total_completion / n
    avg_reasoning = total_reasoning / n
    avg_total = total_all / n
    avg_cost = total_cost / n

    print("-" * 120)
    print(f"AVG {avg_prompt:>10,.0f} {avg_completion:>12,.0f} {avg_reasoning:>10,.0f} " f"{avg_total:>10,.0f} {'':>6} {avg_cost:>7.4f}  {'':>7}  (average per request)")
    print(f"TOT {total_prompt:>10,} {total_completion:>12,} {total_reasoning:>10,} " f"{total_all:>10,} {'':>6} {total_cost:>7.4f}  {total_time:>6.1f}s  (total for {n} queries)")

    # Budget projection
    print("\n" + "=" * 120)
    print("BUDGET PROJECTION")
    print("=" * 120)
    print(f"Pricing:  ${PRICE_INPUT_PER_1K}/1K input  |  ${PRICE_OUTPUT_PER_1K}/1K output  |  ${PRICE_EMBED_PER_1K}/1K embedding")
    print(f"Average cost per request:          ${avg_cost:.4f}")
    print(f"Average tokens per request:        {avg_total:,.0f} ({avg_prompt:,.0f} input + {avg_completion:,.0f} output)")
    print(f"Average reasoning tokens:          {avg_reasoning:,.0f} (subset of completion tokens)")
    print()

    requests_per_dollar = 1.0 / avg_cost if avg_cost > 0 else float("inf")
    requests_per_day = DAILY_BUDGET_USD / avg_cost if avg_cost > 0 else float("inf")
    tokens_per_day = requests_per_day * avg_total

    print(f"$1.00 budget  =>  ~{requests_per_dollar:,.0f} requests")
    print(f"${DAILY_BUDGET_USD:.0f} budget/day  =>  ~{requests_per_day:,.0f} requests/day  (~{tokens_per_day:,.0f} total tokens/day)")
    print()

    # How many users could that serve?
    for queries_per_user in [5, 10, 20, 50]:
        users = requests_per_day / queries_per_user
        print(f"  At {queries_per_user:>2} queries/user/day:  ~{users:,.0f} users/day")

    print()
    embed_cost_per_request = (50 / 1000) * PRICE_EMBED_PER_1K
    print(f"Note: Embedding cost is ~${embed_cost_per_request:.6f}/request — negligible vs LLM cost of ~${avg_cost:.4f}/request")
    print()


if __name__ == "__main__":
    main()
