"""FAISS vector index plus a parallel list of chunk metadata.

Invariant: row i of the FAISS index always matches records[i].
Vectors and their metadata are always added together, in the same call.
"""

from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


@dataclass
class VectorStore:
    dim: int
    index: faiss.IndexFlatIP = field(init=False)
    records: list[ChunkRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Inner product on L2-normalized vectors equals cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        if vectors.shape[0] != len(records):
            raise ValueError("Each vector needs exactly one metadata record.")
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {vectors.shape[1]}.")
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.records.extend(records)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, ChunkRecord]]:
        """Return up to top_k (score, record) pairs, best match first."""
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(query, k)
        return [
            (float(score), self.records[idx])
            for score, idx in zip(scores[0], ids[0])
            if idx != -1
        ]

    def __len__(self) -> int:
        return self.index.ntotal