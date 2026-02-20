# NEC RAG — Roadmap & Next Steps

Current state: agent scores **16/20 (80%)** on a 20-question master electrician practice exam.
Four failures stem from missing Annex data, not rounding to standard sizes, and a couple of
nuanced code sections (GFCI for 250V receptacles, conduit airspace per 300.6).

> **Cost constraint:** This project is in demo stage. Everything must be **free** beyond the
> Azure OpenAI access already provisioned (GPT model + text-embeddings-3-small). No new paid
> services, hosting, or subscriptions until the project graduates past demo.

---

## 1. Include Annex Data in the Reference Text

The NEC Annexes (A–J) are not currently part of the cleaned/structured dataset. At least one
test failure (Annex C conduit fill tables) is directly attributable to this gap.

- [ ] Identify which pages of the OCR text contain Annex content (roughly pages 718+)
- [ ] Run the Annex pages through the existing cleaning pipeline (or a modified version)
- [ ] Structure the Annex data — Annexes have a different layout than the main code body
      (informational tables, examples, cross-references); decide on a schema
- [ ] Chunk and embed Annex content into ChromaDB alongside the main code
- [ ] Verify retrieval works for Annex-specific queries (e.g. "Annex C Table C.9 conduit fill")

---

## 2. Expand Practice Test Coverage

One 20-question exam is a narrow signal. More diverse tests will reveal systematic weaknesses.

- [ ] Research additional master electrician practice exams (free and paid sources)
- [ ] Look for journeyman-level exams as well — different difficulty, broader topic coverage
- [ ] Consider building a question bank organized by NEC chapter/article
- [ ] Add questions that specifically target known weak areas: Annex lookups, standard-size
      rounding, demand calculations, multi-step cross-reference chains
- [ ] Automate scoring across multiple test sets to track improvement over time

---

## 3. Context Window Optimization

The agent currently retrieves 20 subsections per query. Need to understand how much of the
context window is actually being used and whether there's room to pack in more useful content.

### 3a. Measure current context utilization
- [ ] Instrument `_build_context()` to log token counts (total context tokens vs. model limit)
- [ ] Sample a representative set of queries and record context size distribution
- [ ] Determine how close to the model's context limit the agent typically gets

### 3b. Evaluate retrieval quantity
- [ ] Experiment with increasing `n_results` (e.g. 25, 30) and measure impact on answer quality
- [ ] Check whether additional documents improve recall or just add noise
- [ ] Profile latency impact of larger retrievals

### 3c. Hydrate context beyond tables
Currently, referenced *tables* are appended to context. The same idea could apply to
cross-referenced *subsections* (e.g. when 250.122 says "see 250.120" the agent should have
both sections available).

- [ ] Parse section cross-references from retrieved chunks (regex for "Section X.Y", "see X.Y")
- [ ] Look up and append the full text of referenced subsections (avoid duplicates)
- [ ] Measure token budget impact — how many extra tokens does hydration typically add?
- [ ] Evaluate whether this improves answer quality on cross-reference-heavy questions

### 3d. Evaluate a re-ranker
A re-ranker (e.g. a cross-encoder like `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each
retrieved chunk against the query with full attention, which can surface more relevant results
than embedding cosine similarity alone. This pairs well with retrieving more chunks (3b) —
over-retrieve with embeddings, then re-rank to keep the best N.

- [ ] **Prerequisite:** expand the ground-truth question bank (section 2) so there's enough
      signal to measure whether re-ranking actually helps
- [ ] Pick a free, open-source cross-encoder model (runs locally, no new paid services)
- [ ] Implement a re-rank step between `_retrieve()` and `_build_context()`: retrieve
      a larger candidate set (e.g. 40–50), re-rank, keep top 20
- [ ] Compare retrieval precision before/after re-ranking on the ground-truth set
- [ ] Measure latency overhead — cross-encoders are heavier than a cosine lookup

---

## 4. Reorder Retrieved Context for Coherence

Once context is fully hydrated (sections + referenced tables + referenced subsections), the
ordering should be logical rather than arbitrary retrieval-rank order.

- [ ] Group content by chapter/article (all Chapter 2 sections together, then their tables, etc.)
- [ ] Within each chapter group, order sections numerically by section ID
- [ ] Place referenced tables immediately after the section group that references them
- [ ] Evaluate whether this reordering helps the LLM reason more coherently (A/B test)

---

## 5. Build a Frontend

Move beyond the CLI to a web interface. Must be **zero-cost** — no paid hosting or services.

- [ ] Use a free framework: Gradio or Streamlit (simplest), or FastAPI + a lightweight
      static frontend (more control). All run locally or on free-tier hosting.
- [ ] Expose the agent as an API endpoint (POST `/ask` with question text + optional image)
- [ ] Build a chat-style UI with message history
- [ ] Display NEC citations with section IDs, page numbers, and collapsible source text
- [ ] Support image upload for the `explain_image` tool
- [ ] Add a "sources" panel showing which chunks were retrieved and their relevance scores

---

## 6. Share with Adam (with Usage Safeguards)

Let Adam try the agent without using your laptop, while keeping costs at zero (or near-zero).

### 6a. Free deployment options
- [ ] **Option A — Tunnel from your machine:** Run the app locally and expose via a free
      tunnel (ngrok free tier, Cloudflare Tunnel, or `ssh -R`). Zero hosting cost; only
      runs when your machine is on.
- [ ] **Option B — Free-tier cloud:** Gradio apps can be shared via HuggingFace Spaces
      (free tier). Streamlit has Streamlit Community Cloud (free). Both require the app
      to call your existing Azure OpenAI endpoint for LLM/embedding — no new services.
- [ ] Dockerize the application (API + frontend + ChromaDB volume) so either option is
      reproducible

### 6b. Prevent runaway Azure OpenAI costs
- [ ] Add per-user rate limiting in the app itself (e.g. X queries per hour, Y per day)
- [ ] Set a hard monthly spending cap on the Azure OpenAI resource via Azure Portal
      budgets & alerts (this is free to configure)
- [ ] Log every LLM call with token counts and estimated cost
- [ ] Consider a simple "query budget": each user gets N queries per day, resets at midnight

### 6c. Simple auth
- [ ] Add a shared password or basic API key to gate access (no paid auth service)
- [ ] Log who asked what and when, so you can audit usage
