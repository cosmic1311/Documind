"""PDF text extraction with page-level metadata (PyMuPDF)."""

from dataclasses import dataclass

import fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or has no extractable text."""


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, matching the page numbers a PDF viewer shows
    text: str


def extract_pages(pdf_bytes: bytes, filename: str) -> list[PageText]:
    """Return the text of every page that has extractable text.

    Pages with no text (for example, scanned images) are skipped, but the
    page numbers of the remaining pages are kept, so citations stay accurate.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"'{filename}' is not a readable PDF.") from exc

    try:
        if doc.needs_pass:
            raise PDFProcessingError(f"'{filename}' is password-protected.")
        if doc.page_count == 0:
            raise PDFProcessingError(f"'{filename}' has no pages.")

        pages: list[PageText] = []
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text").strip()
            if text:
                pages.append(PageText(page_number=index + 1, text=text))
    finally:
        doc.close()

    if not pages:
        raise PDFProcessingError(
            f"'{filename}' has no extractable text. It may be a scanned image; "
            "OCR is not supported in this version."
        )
    return pages