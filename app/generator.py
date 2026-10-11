"""LLM answer generation. The only module that knows about the LLM provider.

Calls the Gemini REST API with the standard library, so no extra SDK is needed.
"""

import json
import time
import urllib.error
import urllib.request

from app.config import get_settings
from app.vector_store import ChunkRecord

NOT_FOUND_ANSWER = "I can't find the answer in the uploaded documents."

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Status codes that usually mean "try again shortly" rather than "your request is wrong".
RETRYABLE_STATUS = {429, 500, 503, 504}


class LLMError(Exception):
    """Raised when the LLM is not configured or the call fails."""


def build_prompt(question: str, passages: list[tuple[float, ChunkRecord]]) -> str:
    """Put numbered evidence in the prompt and instruct the model to stay within it."""
    evidence = "\n\n".join(
        f"[{number}] (file: {record.filename}, page {record.page_number})\n{record.text}"
        for number, (_, record) in enumerate(passages, start=1)
    )
    return (
        "You answer questions using ONLY the evidence passages below.\n"
        "Rules:\n"
        "- Base every claim on the passages. Do not use outside knowledge.\n"
        "- Cite the passage number for each claim, like [1] or [2][3].\n"
        f"- If the passages do not contain the answer, reply exactly: {NOT_FOUND_ANSWER}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _post_with_retries(request: urllib.request.Request, attempts: int = 3) -> dict:
    """Send the request, retrying briefly on overload, rate-limit, or network errors."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_STATUS and attempt < attempts:
                time.sleep(2**attempt)  # waits 2 seconds, then 4
                continue
            raise LLMError(f"LLM provider returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < attempts:
                time.sleep(2**attempt)
                continue
            raise LLMError("Could not reach the LLM provider (network error or timeout).") from exc
    raise LLMError("LLM provider did not respond after retries.")


def generate_answer(question: str, passages: list[tuple[float, ChunkRecord]]) -> str:
    settings = get_settings()
    if not settings.llm_configured:
        raise LLMError("LLM_API_KEY is not set. Add it to .env to enable answers.")

    body = {
        "contents": [{"parts": [{"text": build_prompt(question, passages)}]}],
        "generationConfig": {"temperature": 0.2},  # low temperature = more repeatable answers
    }
    request = urllib.request.Request(
        GEMINI_URL.format(model=settings.llm_model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": settings.llm_api_key},
        method="POST",
    )
    data = _post_with_retries(request)

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM response had an unexpected format.") from exc