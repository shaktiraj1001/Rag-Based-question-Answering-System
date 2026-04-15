# Design Decisions & Observations

## 1. Why I Chose a Chunk Size of 500 Characters

**Rationale:**

The embedding model used — `all-MiniLM-L6-v2` from sentence-transformers — has a context window of **256 word-pieces (tokens)**. A 500-character chunk translates to roughly 80–120 tokens for standard English prose, leaving comfortable headroom before the model's limit.

**Trade-off analysis:**

| Chunk size | Pros | Cons |
|---|---|---|
| Small (< 200 chars) | Precise, low noise per chunk | Loses surrounding context; answer may span multiple chunks |
| Medium (500 chars) ✅ | Balances specificity with context | Occasional topic bleed at boundaries |
| Large (> 1 000 chars) | More context per chunk | Embedding diluted across multiple topics; slower retrieval |

The 500-char size performed best in informal tests: answers to factual questions (definitions, clauses, named values) were captured within a single chunk without diluting the embedding signal. The 50-character overlap ensures that sentences cut at a chunk boundary still appear — in full — in the adjacent chunk.

**Character-level vs token-level:**
Character-level splitting is deterministic and model-agnostic. Token counts change across embedding model versions; character counts do not. For a production system serving multiple embedding models, character-level chunking is more maintainable.

---

## 2. One Retrieval Failure Case Observed

**Failure: Cross-chunk answer (multi-hop retrieval)**

**Scenario:** A document described a product in one section and its price in a later, unrelated section. The question was:

> *"What is the price of Product X?"*

**What happened:** The chunk containing "Product X" (with its description) had a high cosine similarity to the query. However, the price was stated four paragraphs later in a different chunk (e.g., *"All standard tier products are priced at ₹499/month."*). That chunk mentioned no product names, so its similarity score to the query was low (≈ 0.18, below the threshold of 0.25). The system returned the description chunk but not the pricing chunk — and the LLM correctly said it could not find the price.

**Root cause:** Single-stage dense retrieval cannot handle implicit coreference across chunks. The price chunk lacked the lexical signal ("Product X") needed to score well against the query.

**How I'd fix it in a next iteration:**
- **Re-ranking:** Use a cross-encoder (e.g., `ms-marco-MiniLM`) to re-score retrieved chunks with full query context.
- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer first, then embed that for retrieval — aligns the query vector with the answer space rather than the question space.
- **Metadata-aware chunking:** Store section headings as chunk metadata and boost chunks from the same section as a top-scoring chunk.

---

## 3. One Metric Tracked: End-to-End Latency

**What was measured:**
The `/ask` endpoint records `latency_ms` — wall-clock time from the moment the query arrives to the moment the LLM response is received.

**Breakdown (approximate, local machine):**

| Step | Typical time |
|---|---|
| Query embedding (sentence-transformer) | ~15 ms |
| FAISS similarity search (IndexFlatIP) | < 1 ms |
| OpenAI GPT-3.5-turbo API call | 800–1 500 ms |
| **Total** | **~900–1 600 ms** |

**Insight:** Over 95% of latency lives in the LLM API call. Optimisations upstream (faster embeddings, ANN index) have negligible impact on perceived response time. The meaningful lever is LLM choice: switching to `gpt-3.5-turbo` over `gpt-4` saved ~2–3 seconds per call with acceptable quality for factual retrieval tasks.

**Secondary metric observed — similarity score distribution:**
Tracking per-query average similarity scores revealed a bimodal pattern:
- Queries on topics present in the document: average score ≈ 0.65
- Queries on topics absent from the document: average score ≈ 0.12

Setting the threshold at 0.25 created a clean decision boundary that filtered out hallucination-prone low-score retrievals while preserving all relevant chunks.
