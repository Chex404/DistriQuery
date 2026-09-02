"""Kafka smoke test — proves Python can produce AND consume against your
real Kafka broker, completely isolated from the rest of the app.

Run this BEFORE touching any application code. If this doesn't work, the
problem is purely "kafka-python talking to this Kafka broker" — nothing
to do with FastAPI, the pipeline, or anything else we've built.

Usage:
    python scripts/kafka_smoke_test.py
"""

import json
import sys

from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC = "test-topic"


def main():
    print(f"Connecting producer to {BOOTSTRAP_SERVERS} ...")
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    message = {"hello": "from distriquery", "test": True}
    print(f"Sending: {message}")
    future = producer.send(TOPIC, message)
    result = future.get(timeout=10)
    print(f"Sent OK -> partition {result.partition}, offset {result.offset}")
    producer.flush()
    producer.close()

    print(f"\nConnecting consumer to read it back ...")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=8000,
    )

    found = False
    for record in consumer:
        print(f"Received -> partition {record.partition}, offset {record.offset}: {record.value}")
        found = True

    consumer.close()

    if not found:
        print("\nFAILED: consumer didn't read anything back.")
        sys.exit(1)

    print("\nSUCCESS: Kafka produce + consume both work from Python.")


if __name__ == "__main__":
    main()