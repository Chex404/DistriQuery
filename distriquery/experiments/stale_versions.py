"""
EXPERIMENT: Stale Document Versions (shrinking re-ingestion)

HYPOTHESIS
    Deterministic chunk IDs make re-ingestion overwrite positions that
    still exist. But if a document SHRINKS, positions beyond the new
    chunk count are never touched — their old content should remain as
    orphaned stale chunks.

SETUP
    Ingest a LONG version (multiple chunks), then a much SHORTER version
    at the exact same path.

VARIABLES
    Long version: ~4 chunks. Short version: 1 chunk. Same source path.

METRICS
    ingest_document()'s return value vs. len(vector_store) after the
    second ingest.

Usage:
    python -m distriquery.experiments.stale_versions
"""

import tempfile
from pathlib import Path

from distriquery.experiments.harness import experiment_settings
from distriquery.pipeline import Pipeline


def main():
    print(__doc__)

    settings = experiment_settings()
    settings.chunk_size = 80
    settings.chunk_overlap = 10
    pipeline = Pipeline(settings=settings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_path = Path(tmp_dir) / "shrinking_doc.txt"

        long_version = (
            "Section one is the introduction to the whole system.\n\n"
            "Section two covers the ingestion pipeline in detail.\n\n"
            "Section three covers the query pipeline in detail.\n\n"
            "Section four covers reranking and generation in detail."
        )
        doc_path.write_text(long_version)

        print("RESULTS")
        chunk_count_v1 = pipeline.ingest_document(str(doc_path))
        stored_after_v1 = len(pipeline.vector_store)
        print(
            f"  v1 (long):  ingest_document() returned {chunk_count_v1} chunks, "
            f"vector_store now has {stored_after_v1} chunks total"
        )

        short_version = "Just one short sentence now."
        doc_path.write_text(short_version)

        chunk_count_v2 = pipeline.ingest_document(str(doc_path))
        stored_after_v2 = len(pipeline.vector_store)
        print(
            f"  v2 (short): ingest_document() returned {chunk_count_v2} chunks, "
            f"vector_store now has {stored_after_v2} chunks total"
        )

    if stored_after_v2 > chunk_count_v2:
        orphaned = stored_after_v2 - chunk_count_v2
        print(
            f"\nINTERPRETATION\n"
            f"    CONFIRMED: {orphaned} chunk(s) from the OLD, longer version are still\n"
            f"    sitting in the vector store after re-ingesting a shorter version at\n"
            f"    the same path. This is a real limitation, worth fixing (e.g. delete\n"
            f"    every existing chunk for a source before re-ingesting it)."
        )
    else:
        print(
            "\nINTERPRETATION\n"
            "    FIXED: vector_store's chunk count correctly matches the new,\n"
            "    shorter document — no orphaned chunks remain. ingest_document()\n"
            "    now calls delete_by_source() before adding new chunks, giving\n"
            "    correct \"replace\" semantics instead of upsert-only. This limitation\n"
            "    WAS confirmed via this exact experiment; it has since been fixed."
        )

if __name__ == "__main__":
    main()