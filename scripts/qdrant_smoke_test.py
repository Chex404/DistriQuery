"""Qdrant smoke test — proves Python can write AND search vectors against
your real Qdrant container, completely isolated from the rest of the app.

Usage:
    python scripts/qdrant_smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from distriquery.vectorstore import QdrantVectorStore
from distriquery.chunker import Chunk

COLLECTION = "smoke_test_collection"
DIM = 4


def main():
    print(f"Connecting to Qdrant, using collection '{COLLECTION}' ...")
    store = QdrantVectorStore(collection_name=COLLECTION, dim=DIM, url="http://localhost:6333")

    chunk = Chunk(
        chunk_id="11111111-1111-1111-1111-111111111111",
        text="Kafka partitions events across brokers.",
        source="smoke_test",
        position=0,
    )
    vector = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    print("Writing one vector ...")
    store.add([chunk], vector)

    print("Searching for it back ...")
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)

    if not results:
        print("\nFAILED: search returned nothing.")
        sys.exit(1)

    found = results[0]
    print(f"Found -> chunk_id={found.chunk.chunk_id}, score={found.score}, text='{found.chunk.text}'")

    if found.chunk.chunk_id != chunk.chunk_id:
        print("\nFAILED: got back a different chunk than expected.")
        sys.exit(1)

    print(f"\nStore now has {len(store)} point(s) in this collection.")
    print("SUCCESS: Qdrant write + search both work from Python.")


if __name__ == "__main__":
    main()