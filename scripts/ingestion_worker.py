"""Kafka consumer worker — the actual distributed ingestion process.

Run one or more copies of this (each in its own terminal) to get genuine
distributed workers — Kafka's consumer group mechanism automatically
splits partitions between however many copies are running.

Usage:
    python scripts/ingestion_worker.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaConsumer  # noqa: E402

from distriquery.config import settings  # noqa: E402
from distriquery.ingestion_worker import process_event  # noqa: E402


def main():
    print(f"Config check -> vector_store_backend={settings.vector_store_backend!r}, qdrant_url={settings.qdrant_url!r}")
    print(f"Config check -> kafka_bootstrap_servers={settings.kafka_bootstrap_servers!r}\n")

    consumer = KafkaConsumer(
        settings.kafka_ingestion_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        group_id="ingestion-workers",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,
    )

    print(f"Listening on topic '{settings.kafka_ingestion_topic}' (Ctrl+C to stop) ...")
    for record in consumer:
        event = record.value
        print(f"\nReceived event: {event}")
        try:
            chunk_count = process_event(event)
            consumer.commit()
            print(f"OK -> {chunk_count} chunks ingested, offset committed.")
        except Exception as e:
            print(f"ERROR processing event {event}: {e}")
            print("Marked as 'failed' in the database. Offset NOT committed.")


if __name__ == "__main__":
    main()