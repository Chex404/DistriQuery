"""Shared setup helpers for Phase I experiments."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.config import Settings
from distriquery.db.models import Document, Tenant
from distriquery.db.session import Base
from distriquery.evaluation.golden_dataset import SAMPLE_TXT_GOLDEN_SET, SAMPLE_TXT_SOURCE
from distriquery.evaluation.runner import EvaluationSummary, evaluate_retrieval
from distriquery.evaluation.strategies import hybrid_strategy
from distriquery.pipeline import Pipeline


def experiment_settings() -> Settings:
    return Settings(
        chunk_size=500,
        chunk_overlap=50,
        top_k=4,
        embedding_backend="hashing",
        embedding_dim=128,
        llm_backend="fake",
        vector_store_backend="in-memory",
        reranker_backend="overlap",
        agent_enabled=False,
    )

def use_isolated_settings_globally() -> None:
    """Overrides the GLOBAL settings singleton's backend fields in place.

    process_event() and get_pipeline_for_tenant() are, by design, coupled
    to distriquery.config.settings — the SAME object the real API server
    and worker use, so they correctly share state in production. That
    means an experiment calling the real process_event() function reads
    the real global settings too — including your actual .env, and (if
    VECTOR_STORE_BACKEND=qdrant) your actual persistent Qdrant instance,
    with actual tenant collections that can collide by ID.

    Call this ONCE at the top of any experiment script that uses
    process_event()/get_pipeline_for_tenant() directly — it mutates
    global state for the rest of THIS PROCESS's lifetime, which is
    completely fine for a one-shot `python -m distriquery.experiments...`
    script that exits right after. Never call this inside a long-running
    process like the real API server.
    """
    from distriquery.config import settings

    settings.embedding_backend = "hashing"
    settings.embedding_dim = 128
    settings.llm_backend = "fake"
    settings.vector_store_backend = "in-memory"
    settings.reranker_backend = "overlap"
    settings.agent_enabled = False


def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def setup_tenant_and_document(db, source: str = SAMPLE_TXT_SOURCE):
    tenant = Tenant(name="experiment-tenant", api_key="experiment-key")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    document = Document(tenant_id=tenant.id, source=source, status="pending")
    db.add(document)
    db.commit()
    db.refresh(document)

    return tenant, document


def measure_quality(pipeline: Pipeline, label: str) -> EvaluationSummary:
    retrieval_fn = hybrid_strategy(pipeline)
    summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name=label)
    print(
        f"  [{label}] chunks_in_store={len(pipeline.vector_store)}  "
        f"Precision@4={summary.mean_precision_at_k:.3f}  "
        f"Recall@4={summary.mean_recall_at_k:.3f}  MRR={summary.mrr:.3f}"
    )
    return summary