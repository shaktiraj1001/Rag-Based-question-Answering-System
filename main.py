"""
RAG-Based Question Answering System
FastAPI entry point
"""

import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import AskRequest, AskResponse, UploadResponse, StatusResponse
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from config import settings

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG QA System",
    description="Upload documents and ask questions using Retrieval-Augmented Generation.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Singletons ─────────────────────────────────────────────────────────────────
processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
vector_store = VectorStore(index_path=settings.FAISS_INDEX_PATH)
pipeline = RAGPipeline(vector_store=vector_store)

# ── In-memory job tracker ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Background ingestion task ──────────────────────────────────────────────────
def ingest_document(job_id: str, file_path: Path, filename: str):
    """Chunk, embed, and index a document. Runs in the background."""
    try:
        jobs[job_id]["status"] = "processing"
        chunks = processor.process(file_path, filename)
        vector_store.add_chunks(chunks)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["chunks_indexed"] = len(chunks)
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse, summary="Upload a document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept PDF or TXT documents.
    Document ingestion (chunking + embedding + indexing) runs as a background job.
    Returns a job_id you can poll via /status/{job_id}.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)

    jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(ingest_document, job_id, dest, file.filename)

    return UploadResponse(job_id=job_id, filename=file.filename, status="queued")


@app.get("/status/{job_id}", response_model=StatusResponse, summary="Poll ingestion status")
def get_status(job_id: str):
    """Check whether a background ingestion job has completed."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    info = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=info["status"],
        filename=info.get("filename"),
        chunks_indexed=info.get("chunks_indexed"),
        error=info.get("error"),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question over indexed documents",
)
@limiter.limit("10/minute")          # ← rate limit: 10 requests / minute / IP
def ask_question(request: Request, body: AskRequest):
    """
    Retrieve relevant chunks and generate an answer using an LLM.
    Rate-limited to 10 requests per minute per IP.
    """
    if vector_store.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload documents first.",
        )

    t0 = time.perf_counter()
    answer, sources, scores = pipeline.answer(
        query=body.question,
        top_k=body.top_k,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return AskResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        similarity_scores=scores,
        latency_ms=latency_ms,
    )


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "indexed_chunks": vector_store.count()}
