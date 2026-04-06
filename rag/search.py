"""
BM25 search over the ingested corpus — designed to be called by Claude Code.

Usage:
  python rag/search.py "prompt injection detection methods"
  python rag/search.py "transformer fine-tuning" --top-k 8
  python rag/search.py "adversarial attacks" --source Reference5.pdf

Run  `python rag/ingest.py`  first to build the corpus.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rank_bm25 import BM25Okapi

CORPUS_FILE = Path(__file__).resolve().parent / "index" / "corpus.json"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_corpus(source_filter: str | None = None) -> list[dict]:
    if not CORPUS_FILE.exists():
        print(
            "Corpus not found. Run  `python rag/ingest.py`  first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(CORPUS_FILE, encoding="utf-8") as f:
        corpus = json.load(f)
    if source_filter:
        corpus = [c for c in corpus if source_filter.lower() in c["source"].lower()]
    return corpus


def search(query: str, top_k: int = 6, source_filter: str | None = None) -> None:
    corpus = load_corpus(source_filter)
    if not corpus:
        print("No chunks match the source filter.", file=sys.stderr)
        sys.exit(1)

    tokenized = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(tokenize(query))

    # Rank and pick top-k
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Query : {query}")
    print(f"Top-{top_k} results\n")

    for rank, (idx, score) in enumerate(ranked, start=1):
        chunk = corpus[idx]
        print(f"[{rank}] {chunk['source']}  page {chunk['page']}  (score {score:.2f})")
        print("-" * 60)
        print(chunk["text"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 search over the papers corpus")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--top-k", type=int, default=6, help="Number of results (default: 6)")
    parser.add_argument("--source", help="Filter to a specific paper, e.g. Reference5.pdf")
    args = parser.parse_args()

    search(
        query=" ".join(args.query),
        top_k=args.top_k,
        source_filter=args.source,
    )


if __name__ == "__main__":
    main()
