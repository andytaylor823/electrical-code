# RAG Design Decisions

Research and design notes for the NEC RAG embedding and retrieval pipeline.

## 1. Embedding Oversized Chunks

The `text-embedding-3-small` model has an 8,191-token limit. When a text chunk exceeds this, the current code (`src/nec_rag/data_preprocessing/embedding/embed.py`) recursively halves the text at sentence boundaries, embeds each piece independently, and averages the vectors.

### Problems with Current Halve-and-Average Approach

**Unweighted averaging is incorrect.** The current code gives equal weight to each sub-chunk regardless of length. A 1,200-token chunk and a 6,800-token chunk contribute equally to the final vector. OpenAI's cookbook recommends **token-weighted averaging**, where each sub-chunk's embedding is weighted by its token count:

```python
weights = [ntokens(s) for s in substrings]
total = sum(weights)
embedding = [
    sum(e[j] * w for e, w in zip(embeddings, weights)) / total
    for j in range(len(embeddings[0]))
]
```

**Independent embedding loses cross-attention.** Each sub-chunk is embedded in isolation — the transformer never sees the other half. References like "the values in the table above" lose meaning when the table is in the other chunk.

**For RAG, averaging defeats the purpose.** An averaged embedding is a "center of mass" that's somewhat near everything in the original text but precisely near nothing. If a long NEC section covers both grounding requirements and conductor sizing, the averaged vector won't match either query as well as two separate vectors would.

### Better Approaches

| Approach | Description | Tradeoff |
|---|---|---|
| **Token-weighted averaging** | Weight each sub-chunk's embedding by token count. OpenAI-recommended. | Minimal code change; strictly better than unweighted. Still loses cross-attention. |
| **Keep chunks separate** | Store each sub-chunk as its own retrievable entry with parent section metadata. | Best retrieval precision; requires metadata linkage. |
| **Long-context embedding model** | Use a model with a larger token limit (Voyage-3: 32K, Cohere embed-v4: 128K). | Avoids splitting entirely; requires switching provider. |
| **Late chunking** (Jina AI, 2024) | Pass full text through transformer, pool token embeddings per chunk *after* attention. Each chunk "knows about" the full document. | Requires model internals (not available via API); needs local model. |
| **Summarize-then-embed** | Use GPT to summarize the oversized chunk, then embed the summary. | Compresses semantics effectively; adds latency and cost. |

### Long-Context Embedding Models

| Model | Max Tokens | Provider |
|---|---|---|
| `text-embedding-3-small` (current) | 8,191 | OpenAI / Azure |
| `voyage-3` | 32,000 | Voyage AI |
| `voyage-3-large` | 32,000 | Voyage AI |
| Jina Embeddings v3 | 8,192 | Jina AI |
| Gemini `text-embedding-004` | 2,048 | Google |
| Cohere `embed-v4.0` | 128,000 | Cohere |

### Recommendation

1. **Immediate fix**: Switch to token-weighted averaging (5-line change).
2. **High-impact change**: Store oversized sub-chunks as separate retrievable entries with parent section metadata, rather than averaging into one vector.
3. **Long-term option**: Consider Voyage-3 (32K) or Cohere embed-v4 (128K) to avoid splitting entirely.

---

## 2. Multi-Granularity Embedding (Subsection vs. Part vs. Article)

The NEC hierarchy is Chapter > Article > Part > Subsection. The question: should we embed at multiple levels so the system can see "the forest and the trees"?

### Why Coarse-Grained Embeddings Underperform for Retrieval

Embedding models compress text into a fixed-dimension vector. More text = blurrier signal. An article-level embedding for Article 250 (Grounding and Bonding) covers dozens of distinct topics. It becomes a vague "grounding-ish" vector with lower cosine similarity to any specific query than the precise subsection that answers it.

In retrieval benchmarks, fine-grained chunks consistently outperform coarse chunks on precision and recall.

### "Retrieve Coarse, Then Zoom In" Is Largely Abandoned

Two-stage retrieval (find the right article, then search within it) was an early RAG pattern. Errors in stage 1 are unrecoverable — wrong article means the fine-grained search is looking in the wrong place. Direct fine-grained retrieval with modern embeddings typically outperforms this.

### Patterns That Do Work

**Parent Document Retrieval.** Retrieve by the finest-grain embedding, but return a larger context window to the LLM. Embed at subsection level; when a match is found, also pass the surrounding part or article context into the LLM prompt. The retriever uses the sharp subsection vector; the LLM gets the forest view.

**Auto-Merging Retrieval (LlamaIndex pattern).** Retrieve at leaf (subsection) level. If multiple retrieved chunks share the same parent (e.g., 3 of top-5 results are from Article 250 Part III), automatically merge up to the parent level. Coarse context is used only when the evidence supports it.

**Summary-level secondary index.** Generate an LLM summary of each article, embed those summaries as a secondary retrieval path. Catches vague/exploratory queries ("what does the NEC say about kitchens?") that no single subsection answers well. Lower priority — most NEC queries are specific.

### Recommendation

**Embed at subsection level only.** Store hierarchical metadata (article, part, chapter) alongside each vector. Use metadata at retrieval time to:

1. Expand context for the LLM (parent document retrieval).
2. Detect clustering and merge when multiple hits share a parent (auto-merging).
3. Provide proper citations with full hierarchical paths.

Add article-level summary embeddings later only if vague/exploratory queries become a pain point.

---

## 3. Table Handling in RAG

The NEC contains ~200+ tables (conductor ampacities, demand factors, conduit fill, etc.). Should tables be embedded directly, or handled as post-retrieval augmentation?

### Why Embedding Tables Directly Is Problematic

Embedding models are trained on natural language. A serialized table (`14 AWG | 15 | 20 | 25`) loses spatial relationships — the model doesn't understand that "15" belongs to the intersection of "14 AWG" and "60°C." The resulting vector is a vague "something about conductors and numbers" blob.

Meanwhile, the prose text that *references* a table ("Ampacities for insulated conductors shall be as permitted in Table 310.16") is natural language that embeds well and matches user queries.

### Recommended Approach: Metadata-Driven Augmentation

1. **Embed the prose text** that references tables. These chunks naturally contain language like "In accordance with Table 220.55..." that matches user queries.
2. **Tag each chunk with the table IDs it references.** This is metadata, not an embedding. A regex scan for `Table \d+\.\d+` patterns handles this.
3. **At retrieval time**, when a chunk references Table X, pull the full table from a structured store and inject it (as markdown) into the LLM's context alongside the retrieved text.

This plays to each component's strengths: the embedding model matches query intent to prose intent, and the LLM reads the structured table to extract specific values. LLMs are quite good at reading markdown tables.

### Optional: Table Description Embeddings

For each table, create a short natural language description and embed it as a separate retrievable chunk:

> "Table 310.16: Ampacities of insulated conductors rated up to and including 2000 volts, 60°C through 90°C, not more than three current-carrying conductors in raceway, cable, or earth. Covers conductor sizes from 14 AWG through 2000 kcmil for copper and aluminum conductors."

This catches queries that ask directly about a table or where the relevant subsection doesn't mention the table explicitly enough. NEC table titles are already decent descriptions and could be used directly or enriched with an LLM-generated summary.

### Concrete Implementation

Given the existing structure (`data/prepared/` hierarchy + cleaned tables in `data/intermediate/tables/`):

1. **Table store**: Maintain a lookup of `table_id -> markdown content` (largely exists already in the tables directory).
2. **Subsection embeddings**: Embed subsection text as-is (table references occur naturally in the prose).
3. **Table description embeddings** (optional): One chunk per table with a prose description, tagged with `table_id` metadata.
4. **Retrieval augmentation**: After retrieving top-k chunks, scan for table references via regex. For each referenced table, pull from the table store and append to the LLM context.

---

## References

- [OpenAI Cookbook: Embedding Long Inputs](https://cookbook.openai.com/examples/embedding_long_inputs) — token-weighted averaging recommendation
- [Late Chunking in Long-Context Embedding Models](https://arxiv.org/abs/2409.04701) (Jina AI, 2024) — contextual chunk embeddings via late pooling
- [LlamaIndex Auto-Merging Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/) — hierarchical retrieval with automatic parent merging
- [Voyage AI Embeddings](https://docs.voyageai.com/docs/embeddings) — 32K-token context embedding models
- [Cohere Embed v4](https://docs.cohere.com/docs/embed-api) — 128K-token context embedding model
