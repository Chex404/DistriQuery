"""Chunking.

Splits document text into retrievable units. Implemented as a recursive
structure-aware splitter (Theory Primer, Part 2.2): try splitting on the
"biggest" structural boundary first (paragraph breaks), and only fall back
to smaller boundaries (lines, sentences, words, raw characters) where a
piece is still too large. This preserves natural structure wherever
possible instead of cutting mid-sentence by default.

Deliberately hand-rolled instead of importing LangChain's splitter, per the
project's own rule: understand the mechanism before relying on a library
for it.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Fixed, arbitrary namespace for deterministic chunk IDs (uuid5) — any fixed
# UUID works here, it just needs to be constant across the whole app so the
# same (source, position) always maps to the same chunk_id. This is what
# makes re-ingesting the same file idempotent at the storage layer: Qdrant's
# upsert overwrites a point with an existing ID rather than duplicating it,
# instead of every retry creating a brand-new random ID (Theory Primer,
# Part 4.4 — idempotent workers, not just deduped events).
_CHUNK_ID_NAMESPACE = uuid.UUID("c9c8f8d4-2b1e-4a3f-9e1d-1a2b3c4d5e6f")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    position: int
    metadata: dict = field(default_factory=dict)


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, remaining_separators = separators[0], separators[1:]
    raw_parts = _split_on_separator(text, separator)

    parts = [
        part + separator if i < len(raw_parts) - 1 and separator else part
        for i, part in enumerate(raw_parts)
    ]

    results = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            results.append(part)
        else:
            results.extend(_recursive_split(part, remaining_separators, chunk_size))
    return results


def _merge_with_overlap(pieces: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if not pieces:
        return []

    merged: List[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue

        if current:
            merged.append(current)

        overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
        current = overlap_text + piece

        while len(current) > chunk_size:
            merged.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :] if chunk_overlap > 0 else current[chunk_size:]

    if current:
        merged.append(current)

    return merged


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = separators or DEFAULT_SEPARATORS
    pieces = _recursive_split(text, separators, chunk_size)
    merged_pieces = _merge_with_overlap(pieces, chunk_size, chunk_overlap)

    return [
        Chunk(
            chunk_id=str(uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{source}:{position}")),
            text=piece,
            source=source,
            position=position,
        )
        for position, piece in enumerate(merged_pieces)
    ]