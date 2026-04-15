# RAG-Based Question Answering System

A production-ready FastAPI service that lets you upload documents (PDF / TXT) and ask natural-language questions over them using **Retrieval-Augmented Generation (RAG)**.

---

## Architecture Overview

```
User
 │
 ▼
FastAPI  ──/upload──►  DocumentProcessor  (pdfplumber / plain text)
                              │ chunk + embed (sentence-transformers)
                              ▼
                        VectorStore (FAISS, persisted to disk)
                              ▲
FastAPI  ──/ask──────────────┘
                         similarity search → top-k chunks
                              │
                        RAGPipeline → OpenAI GPT-3.5-turbo
                              │
                         answer + sources + latency
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd rag-qa-system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Copy the sample env and add your OpenAI key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /upload`

Upload a PDF or TXT file.  Ingestion runs as a background job.

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@report.pdf"
# → {"job_id":"...", "filename":"report.pdf", "status":"queued"}
```

### `GET /status/{job_id}`

Poll until `status` is `done` or `failed`.

```bash
curl http://localhost:8000/status/<job_id>
```

### `POST /ask`

Ask a question (rate-limited: 10 req/min/IP).

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the refund policy?", "top_k": 4}'
```

**Response:**

```json
{
  "question": "What is the refund policy?",
  "answer": "Refunds are processed within 7 business days...",
  "sources": ["report.pdf"],
  "similarity_scores": [0.812, 0.741],
  "latency_ms": 923.5
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status":"ok","indexed_chunks":142}
```

---

## Project Structure

```
rag-qa-system/
├── main.py               # FastAPI app, endpoints, background tasks
├── config.py             # Centralised settings (pydantic-settings)
├── models.py             # Pydantic request/response schemas
├── document_processor.py # PDF + TXT parsing, sliding-window chunking
├── vector_store.py       # FAISS index + sentence-transformers embeddings
├── rag_pipeline.py       # Retrieval → LLM generation
├── requirements.txt
├── explanation.md        # Design decisions & failure analysis
└── .env.example
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | GPT-3.5-turbo access |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `SIMILARITY_THRESHOLD` | `0.25` | Min cosine score to include chunk |
| `DEFAULT_TOP_K` | `4` | Chunks retrieved per query |

---

## Tech Stack

| Concern | Choice |
|---|---|
| API framework | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (local, persisted) |
| LLM | OpenAI GPT-3.5-turbo |
| PDF parsing | pdfplumber |
| Validation | Pydantic v2 |
| Rate limiting | slowapi |
| Background jobs | FastAPI `BackgroundTasks` |
