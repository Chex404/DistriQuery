from distriquery.experiments.harness import (
    experiment_settings,
    measure_quality,
    setup_tenant_and_document,
    sqlite_session,
)
from distriquery.evaluation.golden_dataset import SAMPLE_TXT_SOURCE
from distriquery.evaluation.runner import EvaluationSummary
from distriquery.pipeline import Pipeline
from distriquery.tenancy import reset_registry


def test_experiment_settings_uses_isolated_backends():
    settings = experiment_settings()

    assert settings.embedding_backend == "hashing"
    assert settings.vector_store_backend == "in-memory"
    assert settings.reranker_backend == "overlap"
    assert settings.llm_backend == "fake"
    assert settings.agent_enabled is False


def test_sqlite_session_creates_usable_schema():
    db = sqlite_session()
    from distriquery.db.models import Tenant

    assert db.query(Tenant).count() == 0


def test_setup_tenant_and_document_creates_linked_records():
    db = sqlite_session()

    tenant, document = setup_tenant_and_document(db)

    assert document.tenant_id == tenant.id
    assert document.source == SAMPLE_TXT_SOURCE
    assert document.status == "pending"


def test_measure_quality_returns_evaluation_summary():
    reset_registry()
    pipeline = Pipeline(settings=experiment_settings())
    pipeline.ingest_document(SAMPLE_TXT_SOURCE)

    summary = measure_quality(pipeline, label="test-run")

    assert isinstance(summary, EvaluationSummary)
    assert summary.strategy_name == "test-run"
    reset_registry()

def test_use_isolated_settings_globally_overrides_real_settings():
    from distriquery.config import settings
    from distriquery.experiments.harness import use_isolated_settings_globally

    settings.vector_store_backend = "qdrant"
    settings.embedding_backend = "sentence-transformers"

    use_isolated_settings_globally()

    assert settings.vector_store_backend == "in-memory"
    assert settings.embedding_backend == "hashing"