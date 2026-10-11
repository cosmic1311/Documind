"""Application configuration loaded from environment variables (and an optional .env file).

Why a single config module: every tunable value (chunk size, top-k, upload limit,
API key) lives in one place, so the rest of the code never reads os.environ directly.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if it exists. Real environment variables still win.
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    embedding_model: str
    data_dir: Path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        max_upload_mb=_get_int("MAX_UPLOAD_MB", 10),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 4),
        min_similarity=_get_float("MIN_SIMILARITY", 0.25),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        data_dir=data_dir,
    )

    # Fail fast on inconsistent chunking settings instead of failing later during ingestion.
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    return settings