"""Pydantic models defining the API contract.

FastAPI uses these models to: validate incoming JSON, convert it to Python objects,
generate the OpenAPI docs at /docs, and serialize responses back to JSON.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    llm_configured: bool = Field(description="False means /ask will return evidence only.")


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourcePassage(BaseModel):
    document_id: str
    filename: str
    page_number: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True if the answer is supported by retrieved passages.")
    sources: list[SourcePassage]


class ErrorResponse(BaseModel):
    detail: str