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

### 3a. Measure current context utilization ✅
- [x] Instrument `_build_context()` to log token counts (total context tokens vs. model limit)
- [x] Sample a representative set of queries and record context size distribution
- [x] Determine how close to the model's context limit the agent typically gets

See `docs/retrieval_recall_analysis.md` and `scripts/retrieval_recall.py`. Token tracking was
added to the agent via `langchain_community.callbacks.get_openai_callback` (for LangChain LLM
calls) and a manual accumulator for standalone vision API calls. **Conclusion:** a typical
single-search query uses ~6k tokens (~1.5% of GPT-5-mini's 400k context window). The agent
has massive headroom. However, retry spirals can blow up: one failed question consumed 211k
prompt tokens (53% of the context window) across 6 repeated `rag_search` calls due to the
cumulative ReAct message history. A max-3-searches-per-question guardrail was added to the
system prompt to prevent this.

### 3b. Evaluate retrieval quantity ✅
- [x] Experiment with increasing `n_results` (e.g. 25, 30) and measure impact on answer quality
- [x] Check whether additional documents improve recall or just add noise
- [x] Profile latency impact of larger retrievals

See `docs/retrieval_recall_analysis.md` and `scripts/retrieval_recall.py`. Ran retrieval at
n=5, 10, 20, 30, 50 against all 20 exam questions with ground-truth section matching.
**Conclusion:** top-20 is the right default (79% recall). 12 of 17 matched questions have the
answer in the top 5 — retrieval is strongly front-loaded. Going to top-50 only adds 2 more
questions (ranks 24 and 31). Reducing to top-10 loses 2 questions. The agent's failures are
retrieval misses (wrong sections retrieved), not reasoning failures from too much context.

### 3c. Hydrate context beyond tables ✅
Addressed differently than originally planned. Instead of automatically injecting
cross-referenced subsections into the RAG context, we added tools that let the agent fetch
additional context on demand:

- [x] `nec_lookup` — fetch exact subsection text or table content by section/table ID
- [x] `browse_nec_structure` — navigate the NEC hierarchy (chapter → article → part →
      subsection) to discover what exists before doing a targeted lookup

This is more flexible than static cross-reference hydration: the agent decides what additional
context it needs based on the question, rather than us guessing at retrieval time. It also
avoids bloating every query's context with potentially irrelevant cross-referenced sections.

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

## 5. Build a Frontend ✅ (scaffolded)

Move beyond the CLI to a web interface. Must be **zero-cost** — no paid hosting or services.

- [x] Use FastAPI + a lightweight vanilla HTML/CSS/JS static frontend
- [x] Expose the agent via `POST /api/chat` (text + optional images)
- [x] Build a ChatGPT-style chat UI with multi-turn message history
- [x] Support image upload (drag-and-drop / click-to-attach) for the `explain_image` tool
- [x] Simple password auth (configured via `NEC_APP_PASSWORD` in `.env`)
- [x] Markdown rendering in assistant messages (NEC citations with code blocks, tables, etc.)
- [ ] Streaming responses (SSE) for token-by-token output
- [ ] Display a "sources" panel showing which chunks were retrieved and their relevance scores
- [ ] Collapsible source text for NEC citations

### Running the web app

```bash
# Activate venv and start the server
source .venv/bin/activate
python -m nec_rag.web.app
# => Uvicorn running on http://localhost:8000

# Or equivalently:
python -m nec_rag.web
```

### Sharing with Adam via tunnel

```bash
# Install ngrok (one-time)
brew install ngrok

# In a separate terminal, expose localhost:8000
ngrok http 8000
# => Gives Adam a URL like https://abc123.ngrok-free.app

# Adam opens the URL, enters the shared NEC_APP_PASSWORD, and chats.
# Your machine must be running for this to work.
```

---

## 6. Share with Adam (with Usage Safeguards)

Let Adam try the agent without using your laptop, while keeping costs at zero (or near-zero).

### 6a. Free deployment options
- [x] **Option A — Tunnel from your machine:** Run the app locally and expose via ngrok
      or Cloudflare Tunnel (`ngrok http 8000`). Zero hosting cost; only runs when your
      machine is on. See section 5 for instructions.
- [ ] **Option B — Free-tier cloud:** Gradio apps can be shared via HuggingFace Spaces
      (free tier). Streamlit has Streamlit Community Cloud (free). Both require the app
      to call your existing Azure OpenAI endpoint for LLM/embedding — no new services.
- [ ] Dockerize the application (API + frontend + ChromaDB volume) so either option is
      reproducible

### 6b. Prevent runaway Azure OpenAI costs
- [ ] Add per-user rate limiting in the app itself (e.g. X queries per hour, Y per day)
- [ ] Set a hard monthly spending cap on the Azure OpenAI resource via Azure Portal
      budgets & alerts (this is free to configure)
- [x] Log every LLM call with token counts and estimated cost (token counts logged per
      request and shown in the chat UI)
- [ ] Consider a simple "query budget": each user gets N queries per day, resets at midnight

### 6c. Simple auth
- [x] Add a shared password to gate access (configured via `NEC_APP_PASSWORD` in `.env`)
- [ ] Log who asked what and when, so you can audit usage
