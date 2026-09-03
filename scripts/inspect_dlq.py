"""Inspect the dead-letter queue — prints every message currently sitting
in it.

Usage:
    python scripts/inspect_dlq.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaConsumer  # noqa: E402

from distriquery.config import settings  # noqa: E402


def main():
    consumer = KafkaConsumer(
        settings.kafka_dlq_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        group_id="dlq-inspector",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=5000,
    )

    print(f"Reading '{settings.kafka_dlq_topic}' ...\n")
    count = 0
    for record in consumer:
        count += 1
        event = record.value
        print(f"--- DLQ message {count} ---")
        print(f"  document_id: {event.get('document_id')}")
        print(f"  tenant_id:   {event.get('tenant_id')}")
        print(f"  path:        {event.get('path')}")
        print(f"  error:       {event.get('error')}")
        print()

    if count == 0:
        print("(empty — nothing has been dead-lettered)")
    else:
        print(f"Total: {count} message(s) in the DLQ.")


if __name__ == "__main__":
    main()