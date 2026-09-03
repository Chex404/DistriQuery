"""Kafka consumer worker — the actual distributed ingestion process.

The offset is committed after EVERY event, whether it succeeded or ended
up in the dead-letter queue — a message sent to the DLQ has been
"handled" and must not keep blocking its partition.

Usage:
    python scripts/ingestion_worker.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaConsumer, KafkaProducer  # noqa: E402

from distriquery.config import settings  # noqa: E402
from distriquery.ingestion_worker import handle_event_with_retries  # noqa: E402


def main():
    consumer = KafkaConsumer(
        settings.kafka_ingestion_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        group_id="ingestion-workers",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,
    )
    dlq_producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Listening on topic '{settings.kafka_ingestion_topic}' (Ctrl+C to stop) ...")
    for record in consumer:
        event = record.value
        print(f"\nReceived event: {event}")

        succeeded = handle_event_with_retries(
            event,
            dlq_producer,
            dlq_topic=settings.kafka_dlq_topic,
            max_retries=settings.kafka_max_retries,
        )
        consumer.commit()

        if succeeded:
            print("OK -> processed successfully, offset committed.")
        else:
            print(
                f"FAILED after {settings.kafka_max_retries} attempts -> "
                f"sent to DLQ '{settings.kafka_dlq_topic}', offset committed."
            )


if __name__ == "__main__":
    main()