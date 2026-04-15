"""
RAGPipeline
───────────
Retrieves top-k chunks from the vector store and calls the LLM to
generate a grounded answer.  Falls back gracefully if the OpenAI key
is missing (returns retrieved context only).
"""

from openai import OpenAI

from config import settings
from vector_store import VectorStore


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context excerpts.
If the context does not contain enough information to answer, say:
"I could not find a clear answer in the uploaded documents."
Do not fabricate facts."""


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._client = None
        if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-YOUR"):
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def answer(
        self, query: str, top_k: int = 4
    ) -> tuple[str, list[str], list[float]]:
        """
        Returns:
            answer      – LLM-generated string
            sources     – list of source filenames
            scores      – list of cosine similarity scores
        """
        results = self.vector_store.search(query, top_k=top_k)

        if not results:
            return (
                "No relevant information found in the indexed documents.",
                [],
                [],
            )

        context_parts = []
        sources = []
        scores = []
        for meta, score in results:
            context_parts.append(f"[Source: {meta['source']}]\n{meta['text']}")
            sources.append(meta["source"])
            scores.append(round(score, 4))

        context = "\n\n---\n\n".join(context_parts)

        if self._client is None:
            # No valid API key — return retrieved context as the "answer"
            answer = (
                "⚠️  OpenAI key not configured. Retrieved context:\n\n" + context
            )
        else:
            answer = self._call_llm(query, context)

        return answer, sources, scores

    def _call_llm(self, query: str, context: str) -> str:
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely based only on the context above."
        )
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
