"""
Extract text from all PDFs and save a searchable corpus.

Usage:
  python rag/ingest.py

Output:
  rag/index/corpus.json   — list of chunks with source, page, and text
"""

import json
import sys
from pathlib import Path

import fitz  # pymupdf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR   = PROJECT_ROOT / "01 - List of Papers Studied-20260406T052002Z-3-001"
INDEX_DIR    = Path(__file__).resolve().parent / "index"
CORPUS_FILE  = INDEX_DIR / "corpus.json"

CHUNK_SIZE    = 1200   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks


def chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if piece:
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_pdf(pdf_path: Path) -> list[dict]:
    records = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            for chunk_idx, piece in enumerate(chunk(text)):
                records.append({
                    "source":    pdf_path.name,
                    "page":      page_num,
                    "chunk_idx": chunk_idx,
                    "text":      piece,
                })
        doc.close()
    except Exception as e:
        print(f"  [warn] {pdf_path.name}: {e}", file=sys.stderr)
    return records


def main() -> None:
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s)")

    corpus = []
    for pdf_path in tqdm(pdf_files, desc="Extracting"):
        corpus.extend(extract_pdf(pdf_path))

    if not corpus:
        print("No text extracted.", file=sys.stderr)
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)

    print(f"\nCorpus saved: {CORPUS_FILE}")
    print(f"Total chunks: {len(corpus):,}")
    print(f"Papers      : {len(pdf_files)}")


if __name__ == "__main__":
    main()
