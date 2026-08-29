"""Configuration.

Deliberately tiny for Phase A — a dataclass with defaults, overridable via
environment variables. No config framework, no YAML files: those become
worth it once Phase B adds a real backend with a database URL, Kafka
brokers, etc. Adding complexity here now would be exactly the kind of
premature engineering the project brief warns against.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Loads a .env file from the project root (if one exists) into the process's
# environment variables, BEFORE the Settings fields below read them. Without
# this call, a .env file would just sit there unused.
load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    chunk_size: int = _env_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 50)
    top_k: int = _env_int("TOP_K", 4)

    # "hashing" = dependency-free stand-in used in tests/this demo.
    # "sentence-transformers" = real embedding model (needs internet + torch).
    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "hashing")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 256)
    sentence_transformer_model: str = os.environ.get(
        "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )

    # "fake" = deterministic, extractive stand-in used in tests/this demo.
    # "anthropic" = real LLM call (needs an API key).
    llm_backend: str = os.environ.get("LLM_BACKEND", "fake")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


settings = Settings()