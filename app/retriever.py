"""Query side: embed the question, search FAISS, drop weak matches."""

from app.config import get_settings
from app.embeddings import embed_texts
from app.ingest import get_store
from app.vector_store import ChunkRecord


def retrieve(question: str, top_k: int | None = None) -> list[tuple[float, ChunkRecord]]:
    """Return (score, record) pairs at or above MIN_SIMILARITY, best first."""
    settings = get_settings()
    k = top_k or settings.top_k
    query_vector = embed_texts([question])[0]
    hits = get_store().search(query_vector, top_k=k)
    return [(score, record) for score, record in hits if score >= settings.min_similarity]