"""CLI entry point for the Phase A MVP.

Usage:
    python scripts/ask.py <path-to-document> "<question>"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distriquery.pipeline import Pipeline  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print('Usage: python scripts/ask.py <path-to-document> "<question>"')
        sys.exit(1)

    document_path, question = sys.argv[1], sys.argv[2]

    pipeline = Pipeline()
    chunk_count = pipeline.ingest_document(document_path)
    print(f"Ingested {document_path} -> {chunk_count} chunks\n")

    payload = pipeline.answer(question)
    print(json.dumps(payload.to_dict(), indent=2))


if __name__ == "__main__":
    main()