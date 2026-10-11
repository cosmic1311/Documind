"""List the Gemini models your API key can use for text generation.

Run from the documind/ folder:  python list_models.py
"""

import json
import urllib.request

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is empty. Add it to .env first.")

    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    request = urllib.request.Request(url, headers={"x-goog-api-key": settings.llm_api_key})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    for model in data.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"].removeprefix("models/"))


if __name__ == "__main__":
    main()