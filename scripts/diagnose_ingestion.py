"""Diagnostic: calls the EXACT same ingestion path the worker uses, but
synchronously, in one process, with no Kafka involved — to isolate whether
the write itself is broken, independent of any timing/process confusion.

Usage:
    python scripts/diagnose_ingestion.py "uploads\\1\\Neso Academy.docx"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

from distriquery.tenancy import get_pipeline_for_tenant, reset_registry  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print('Usage: python scripts/diagnose_ingestion.py "<path-to-file>"')
        sys.exit(1)

    file_path = sys.argv[1]
    tenant_id = 1

    reset_registry()

    pipeline = get_pipeline_for_tenant(tenant_id)
    print(f"vector_store type: {type(pipeline.vector_store).__name__}")
    print(f"vector_store collection name: {getattr(pipeline.vector_store, '_collection_name', 'N/A')}")

    print(f"\nCalling ingest_document({file_path!r}) ...")
    chunk_count = pipeline.ingest_document(file_path)
    print(f"ingest_document() returned: {chunk_count}")

    print(f"\nlen(pipeline.vector_store) reports: {len(pipeline.vector_store)}")

    print("\nChecking Qdrant server directly (separate client, bypassing our code entirely) ...")
    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection(f"tenant_{tenant_id}")
    print(f"Qdrant server reports points_count: {info.points_count}")


if __name__ == "__main__":
    main()