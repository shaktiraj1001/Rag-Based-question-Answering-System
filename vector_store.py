"""
VectorStore
───────────
Wraps FAISS for similarity search.
Embeddings generated with sentence-transformers (local, no API cost).
Index is persisted to disk so it survives restarts.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from document_processor import Chunk


class VectorStore:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".meta.json")

        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Flat inner-product index (cosine after L2 normalisation)
        self._index = faiss.IndexFlatIP(self.dim)
        self._metadata: list[dict] = []   # parallel list: {text, source, chunk_index}

        self._load_if_exists()

    # ── Write ──────────────────────────────────────────────────────────────────
    def add_chunks(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        self._index.add(embeddings)
        for c in chunks:
            self._metadata.append({
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
            })
        self._save()

    # ── Read ───────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 4) -> list[tuple[dict, float]]:
        """Return list of (metadata_dict, cosine_score) sorted by score desc."""
        if self.is_empty():
            return []
        q_emb = self._embed([query])
        scores, indices = self._index.search(q_emb, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < settings.SIMILARITY_THRESHOLD:
                continue
            results.append((self._metadata[idx], float(score)))
        return results

    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    def count(self) -> int:
        return self._index.ntotal

    # ── Persistence ────────────────────────────────────────────────────────────
    def _save(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self._metadata, ensure_ascii=False))

    def _load_if_exists(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._metadata = json.loads(self.meta_path.read_text())

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")
