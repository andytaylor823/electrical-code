# Retrieval Recall Analysis

Analysis of whether the RAG retrieval pipeline surfaces the correct NEC sections for the 20-question master electrician practice exam. Conducted February 2026.

## Motivation

The agent scores 16/20 (80%) on the exam. Before tuning the LLM reasoning or adding more data, we needed to answer a fundamental question: **are the failures caused by the retrieval step (wrong sections retrieved) or the reasoning step (right sections retrieved, wrong answer produced)?**

Specifically:
- Is top-20 retrieval (`n_results=20`) sufficient, or does the ground-truth section fall outside the top 20?
- Would top-10 be enough (less noise for the LLM), or do we need top-50 (better recall)?
- For the questions the agent gets wrong, is the correct NEC section even in the retrieved context?

## Method

Script: `scripts/retrieval_recall.py`

For each of the 20 exam questions, we defined the ground-truth NEC section(s) and/or table(s) that contain the answer (extracted from the exam answer key). We then ran vector retrieval at `n_results` = 5, 10, 20, 30, and 50, and checked whether any retrieved chunk matched the ground-truth reference. A chunk "matches" if its `section_id` starts with the expected prefix (e.g., `220.82` matches `220.82(A)`) or if its `referenced_tables` metadata contains the expected table ID.

## Results

| QID | n=5 | n=10 | n=20 | n=30 | n=50 | Ground Truth |
|-----|-----|------|------|------|------|--------------|
| q01 | #3 | #3 | #3 | #3 | #3 | 220.82 |
| q02 | #2 | #2 | #2 | #2 | #2 | 220.103, Table 220.103 |
| q03 | #1 | #1 | #1 | #1 | #1 | 511.3, Table 511.3 |
| q04 | #5 | #5 | #5 | #5 | #5 | 501.15 |
| q05 | #1 | #1 | #1 | #1 | #1 | 404.8 |
| q06 | #1 | #1 | #1 | #1 | #1 | 314.28, 314.16, Table 314.16 |
| q07 | MISS | MISS | MISS | #24 | #24 | 300.5, Table 300.5 |
| q08 | #1 | #1 | #1 | #1 | #1 | 230.26 |
| q09 | MISS | MISS | MISS | MISS | MISS | Table C.9 |
| q10 | #2 | #2 | #2 | #2 | #2 | 250.122, Table 250.122 |
| q11 | #1 | #1 | #1 | #1 | #1 | 240.24, 404.8 |
| q12 | #4 | #4 | #4 | #4 | #4 | 300.5 |
| q13 | MISS | #10 | #10 | #10 | #10 | 314.23 |
| q14 | MISS | MISS | MISS | MISS | MISS | 705.31 |
| q15 | #1 | #1 | #1 | #1 | #1 | 551.73, Table 551.73 |
| q16 | MISS | MISS | #12 | #12 | #12 | 501.15 |
| q17 | MISS | MISS | #12 | #12 | #12 | 590.6 |
| q18 | #1 | #1 | #1 | #1 | #1 | 250.53 |
| q19 | n/a | n/a | n/a | n/a | n/a | (calculation) |
| q20 | MISS | MISS | MISS | MISS | #31 | 300.6 |

### Recall by n_results

| n_results | Recall | Misses |
|-----------|--------|--------|
| 5         | 12/19 (63%) | 7 |
| 10        | 13/19 (68%) | 6 |
| **20**    | **15/19 (79%)** | **4** |
| 30        | 16/19 (84%) | 3 |
| 50        | 17/19 (89%) | 2 |

### Rank distribution (n=50)

Most matches land in the top 5. The retrieval is strongly front-loaded — when the right section is found, it's almost always near the top.

```
  # 1-5 :  12  ████████████
  # 6-10:   1  █
  #11-15:   2  ██
  #16-20:   0
  #21-30:   1  █
  #31-50:   1  █
```

## Key Findings

### Top-20 is the right default

Going from 10 → 20 picks up q16 and q17 (both at rank 12). Going from 20 → 50 only adds q07 (rank 24) and q20 (rank 31) — marginal gains. Reducing to top-10 would lose q13, q16, and q17. The top-20 setting balances recall against context noise.

### Two permanent retrieval misses

**q09 — Annex C Table C.9 (conduit fill).** The Annex data is not in the dataset at all. This is a known gap tracked in TODO section 1. No amount of retrieval tuning will fix this — the data must be added to the pipeline.

**q14 — Section 705.31 (PV supply-side conductor connections).** Section 705 exists in the structured data, but the embedding similarity never surfaces it even at n=50. The question mentions "solar photovoltaic (PV) systems" which routes the embedding toward Article 690 (Solar Photovoltaic Systems) instead of Article 705 (Interconnected Electric Power Production Sources). The top-10 retrieved sections for q14 were all from Articles 690, 547, 691, and 625 — semantically adjacent but wrong.

### q07 and q20 are marginal — would benefit from a re-ranker

Both have the correct section in the dataset but at ranks 24 and 31 respectively. A re-ranker (cross-encoder) applied after an over-retrieval of 50 candidates could promote these into the top 20.

## Issue: Agent Query Formatting Degrades Retrieval

When the agent couldn't find an answer, we observed it reformulating queries in a **search-engine syntax** with quoted phrases:

```
"supply-side conductor connections" "overcurrent" "within" "service disconnecting means" "3 m" "10 ft" NEC 2023
```

This is harmful because `rag_search` uses **semantic embedding search**, not keyword search. The embedding model (`text-embedding-3-large`) encodes the entire query string into a single vector. It does not interpret quotation marks as "exact phrase match" operators. Instead, the quotes:

1. **Add noise tokens** — the model wastes attention encoding literal quote characters
2. **Fragment semantics** — instead of a coherent natural language question, the model sees a bag of disconnected quoted phrases, producing a less meaningful embedding vector
3. **Dilute intent** — appending keywords like `NEC 2023` adds nothing (all content is NEC 2023); padding the query with variations of the same terms doesn't improve cosine similarity

A clean natural language query like `"What is the maximum distance for PV supply-side conductor overcurrent protection from the service disconnect?"` would produce a much sharper embedding.

### Compounding cost of retry spirals

On q14, the agent called `rag_search` **6 times** before giving up. The ReAct loop is cumulative — each LLM call includes all prior messages and tool results. By the 7th call, the conversation carried 6 rounds of 20-chunk context. Total token usage for this single question: **211,263 prompt tokens** (over 50% of GPT-5-mini's 400k context window).

### Fixes applied

1. **System prompt: search guidelines.** Added instructions explaining that `rag_search` is semantic embedding search, with good/bad query examples. Tells the agent to use plain natural language without quotes or search syntax.

2. **System prompt: max 3 searches per question.** Caps `rag_search` calls at 3 per user question. If the answer isn't found after 3 attempts, the agent must respond with what it has and note the gap. This prevents token-burning retry spirals.

3. **New tools: `nec_lookup` and `browse_nec_structure`.** Added deterministic lookup tools so the agent doesn't have to rely solely on semantic search. `nec_lookup` fetches exact section/table content by ID. `browse_nec_structure` lets the agent navigate the NEC hierarchy (chapter → article → part → subsection) to discover what exists. These complement `rag_search` by providing precise access when the agent already knows (or can infer) the relevant section number.
