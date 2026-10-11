"""Sentence Transformer embeddings. The model is loaded once and reused."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the model on first use. Loading takes seconds, so the result is cached."""
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 array of L2-normalized vectors.

    Normalizing makes the inner product equal to cosine similarity, which is
    what the FAISS index in the next step relies on.
    """
    model = get_embedding_model()
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    vectors = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vectors.astype(np.float32)