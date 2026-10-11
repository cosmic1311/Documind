"""FastAPI application entry point.

Run from the documind/ folder with:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, status

from app.config import get_settings
from app.generator import NOT_FOUND_ANSWER, LLMError, generate_answer
from app.ingest import get_store, ingest_pdf
from app.pdf_loader import PDFProcessingError
from app.retriever import retrieve
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourcePassage,
    UploadResponse,
)

app = FastAPI(
    title="DocuMind API",
    description="Upload PDFs and ask grounded questions about them (RAG).",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check. Plain `def` (not `async def`) because it does no I/O."""
    settings = get_settings()
    return HealthResponse(status="ok", llm_configured=settings.llm_configured)


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def upload_document(file: UploadFile) -> UploadResponse:
    """Accept one PDF, then extract, chunk, embed, and index it.

    Plain `def` because reading the file and embedding are blocking work.
    FastAPI runs this in a threadpool, so the server stays responsive.
    """
    settings = get_settings()
    filename = file.filename or "upload.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted.",
        )

    # Read one byte past the limit: enough to detect "too large" without loading a huge file.
    pdf_bytes = file.file.read(settings.max_upload_bytes + 1)
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content is not a valid PDF.",
        )

    try:
        document = ingest_pdf(pdf_bytes, filename)
    except PDFProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document=document,
        message=f"Indexed {document.chunk_count} chunks from {filename}.",
    )


@app.post("/ask", response_model=AskResponse, tags=["questions"])
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve evidence for the question, then ask the LLM to answer from it only."""
    settings = get_settings()
    if not settings.llm_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM_API_KEY is not configured. Add it to .env.",
        )
    if len(get_store()) == 0:
        return AskResponse(
            answer="No documents have been uploaded yet.", grounded=False, sources=[]
        )

    passages = retrieve(request.question, top_k=request.top_k)
    if not passages:
        return AskResponse(answer=NOT_FOUND_ANSWER, grounded=False, sources=[])

    try:
        answer = generate_answer(request.question, passages)
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    grounded = answer != NOT_FOUND_ANSWER
    sources = (
        [
            SourcePassage(
                document_id=record.document_id,
                filename=record.filename,
                page_number=record.page_number,
                score=score,
                text=record.text,
            )
            for score, record in passages
        ]
        if grounded
        else []
    )
    return AskResponse(answer=answer, grounded=grounded, sources=sources)