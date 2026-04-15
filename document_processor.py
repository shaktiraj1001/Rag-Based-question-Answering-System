"""
DocumentProcessor
─────────────────
Handles two formats: PDF and TXT.
Splits text into overlapping chunks ready for embedding.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class Chunk:
    text: str
    source: str          # original filename
    chunk_index: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public ─────────────────────────────────────────────────────────────────
    def process(self, file_path: Path, filename: str) -> list[Chunk]:
        """Parse file → extract text → split into chunks."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        return self._split(text, filename)

    # ── Private ────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from all pages of a PDF."""
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text.strip())
        return "\n\n".join(pages)

    def _split(self, text: str, source: str) -> list[Chunk]:
        """
        Sliding-window character-level chunking.

        Why character-level?
        Token counts vary by model; character counts are deterministic and
        model-agnostic.  A 500-char chunk ≈ 100-120 tokens for English text,
        which fits comfortably within most embedding model context windows
        (256-512 tokens) while preserving enough context for retrieval.

        The 50-char overlap prevents answers that span a chunk boundary from
        being missed entirely.
        """
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, chunk_index=idx))
                idx += 1
            start = end - self.chunk_overlap   # slide back by overlap
        return chunks
