"""
Pydantic request / response models — validation happens automatically.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="What is RAG?")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


# ── Response models ────────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str                         # queued | processing | done | failed
    filename: Optional[str] = None
    chunks_indexed: Optional[int] = None
    error: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    similarity_scores: list[float]
    latency_ms: float
