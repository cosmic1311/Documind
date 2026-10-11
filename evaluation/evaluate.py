"""Recall@K evaluation for retrieval.

Run from documind/:  python evaluation/evaluate.py

Recall@K = share of questions where at least one of the top-K retrieved chunks
comes from a page listed as correct for that question.
"""

import json
from pathlib import Path

from app.embeddings import embed_texts
from app.ingest import get_store, ingest_pdf

EVAL_DIR = Path(__file__).resolve().parent
PDF_DIR = EVAL_DIR.parent / "data" / "eval_pdfs"
DATASET_PATH = EVAL_DIR / "test_questions.json"
KS = [1, 3, 5]


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    if not questions:
        raise SystemExit("test_questions.json has no questions yet.")

    for filename in dataset["documents"]:
        ingest_pdf((PDF_DIR / filename).read_bytes(), filename)
        print(f"Indexed {filename}")

    max_k = max(KS)
    hits_at_k = {k: 0 for k in KS}
    misses = []

    for item in questions:
        correct = {(item["filename"], page) for page in item["pages"]}
        query_vector = embed_texts([item["question"]])[0]
        results = get_store().search(query_vector, top_k=max_k)
        ranked = [(record.filename, record.page_number) for _, record in results]

        for k in KS:
            if any(pair in correct for pair in ranked[:k]):
                hits_at_k[k] += 1
        if not any(pair in correct for pair in ranked):
            misses.append((item["question"], item["filename"], item["pages"], ranked[:3]))

    total = len(questions)
    print(f"\nQuestions evaluated: {total}")
    for k in KS:
        print(f"Recall@{k}: {hits_at_k[k]}/{total} = {hits_at_k[k] / total:.0%}")

    if misses:
        print(f"\nMissed within top {max_k} ({len(misses)}):")
        for question, filename, pages, top3 in misses:
            print(f"- {question}\n  expected: {filename} pages {pages}\n  got top 3: {top3}")


if __name__ == "__main__":
    main()