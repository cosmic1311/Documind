"""Split page text into overlapping chunks. Each chunk belongs to exactly one page."""

from dataclasses import dataclass

from app.pdf_loader import PageText


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int  # position of this chunk within its page, starting at 0
    text: str


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Slide a window of `chunk_size` characters over each page's text.

    Consecutive windows share `chunk_overlap` characters, so a sentence cut at
    one boundary still appears whole in the next chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for page in pages:
        # Collapse line breaks and repeated spaces from PDF extraction.
        text = " ".join(page.text.split())
        start = 0
        chunk_index = 0

        while start < len(text):
            piece =text[start : start + chunk_size]
            if piece.strip():
                chunks.append(
                    Chunk(page_number=page.page_number, chunk_index=chunk_index, text=piece)
                )
                chunk_index += 1
            if start + chunk_size >= len(text):
                break  # this window reached the end of the page
            start += step

    return chunks