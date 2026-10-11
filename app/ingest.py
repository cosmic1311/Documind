"""Ingestion: PDF bytes -> pages -> chunks -> embeddings -> FAISS store.

Runs once per upload. Questions never re-process documents.
"""

import uuid
from functools import lru_cache

from app.chunker import chunk_pages
from app.config import get_settings
from app.embeddings import embed_texts, get_embedding_model
from app.pdf_loader import extract_pages
from app.schemas import DocumentInfo
from app.vector_store import ChunkRecord, VectorStore


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """One process-wide store. Data is lost on restart; persistence comes later."""
    dim = get_embedding_model().get_sentence_embedding_dimension()
    return VectorStore(dim=dim)


def ingest_pdf(pdf_bytes: bytes, filename: str) -> DocumentInfo:
    settings = get_settings()
    pages = extract_pages(pdf_bytes, filename)  # raises PDFProcessingError on bad input
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

    document_id = uuid.uuid4().hex
    records = [
        ChunkRecord(
            document_id=document_id,
            filename=filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )
        for chunk in chunks
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    get_store().add(vectors, records)

    # Note: this counts pages that had extractable text, not every page in the PDF.
    return DocumentInfo(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
    )