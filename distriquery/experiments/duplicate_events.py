"""
EXPERIMENT: Duplicate Kafka Events

HYPOTHESIS
    Kafka's at-least-once delivery guarantee means the same ingestion
    event WILL be redelivered under real failure conditions. Because of
    deterministic chunk IDs and the ProcessedEvent dedup ledger,
    redelivering the same event should NOT change corpus size or
    degrade retrieval quality.

SETUP
    A single document is ingested via process_event() using a FIXED
    event_id, then redelivered several more times.

VARIABLES
    Cumulative deliveries of the identical event_id: 1, 2, 5

METRICS
    Corpus size and Precision@4/Recall@4/MRR after each round.

Usage:
    python -m distriquery.experiments.duplicate_events
"""

from distriquery.experiments.harness import (
    measure_quality,
    setup_tenant_and_document,
    sqlite_session,
    use_isolated_settings_globally,
)
from distriquery.ingestion_worker import process_event
from distriquery.tenancy import get_pipeline_for_tenant, reset_registry


def main():
    print(__doc__)

    # MUST come first — process_event() reads the global settings
    # singleton, which otherwise reflects your real .env (and could try
    # to write into your real, persistent Qdrant instance).

    use_isolated_settings_globally()
    reset_registry()
    db = sqlite_session()
    tenant, document = setup_tenant_and_document(db)

    event = {
        "event_id": "duplicate-events-experiment-fixed-id",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }

    print("RESULTS")
    delivery_batches = [1, 1, 3]
    total_deliveries = 0

    for batch_size in delivery_batches:
        for _ in range(batch_size):
            process_event(event, db=db)
            total_deliveries += 1

        pipeline = get_pipeline_for_tenant(tenant.id)
        measure_quality(pipeline, label=f"after {total_deliveries} deliveries")

    print(
        "\nINTERPRETATION\n"
        "    If corpus size and all three metrics stayed IDENTICAL across every\n"
        "    row above, idempotency held under real repeated delivery."
    )

    reset_registry()


if __name__ == "__main__":
    main()