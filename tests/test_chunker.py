import pytest

from distriquery.chunker import chunk_text


def test_short_text_returns_single_chunk():
    text = "just one short paragraph."
    chunks = chunk_text(text, source="doc1", chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].position == 0
    assert chunks[0].source == "doc1"


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", source="doc1")
    assert chunks == []


def test_long_text_is_split_into_multiple_chunks():
    paragraph = "This is a sentence about DistriQuery. " * 40
    chunks = chunk_text(paragraph, source="doc1", chunk_size=300, chunk_overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300


def test_positions_are_sequential():
    paragraph = "Sentence number filler text here. " * 30
    chunks = chunk_text(paragraph, source="doc1", chunk_size=200, chunk_overlap=20)

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_overlap_actually_overlaps():
    paragraph = "Alpha bravo charlie delta echo foxtrot golf hotel. " * 20
    chunks = chunk_text(paragraph, source="doc1", chunk_size=150, chunk_overlap=40)

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        tail = first.text[-40:]
        assert tail[:20] in second.text


def test_prefers_paragraph_boundaries_when_possible():
    text = "First paragraph is short.\n\nSecond paragraph is also short."
    chunks = chunk_text(text, source="doc1", chunk_size=40, chunk_overlap=5)

    assert any("First paragraph" in c.text for c in chunks)
    assert any("Second paragraph" in c.text for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", source="doc1", chunk_size=100, chunk_overlap=100)

def test_chunk_ids_are_deterministic_for_same_source_and_content():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks_1 = chunk_text(text, source="doc1", chunk_size=30, chunk_overlap=5)
    chunks_2 = chunk_text(text, source="doc1", chunk_size=30, chunk_overlap=5)

    ids_1 = [c.chunk_id for c in chunks_1]
    ids_2 = [c.chunk_id for c in chunks_2]
    assert ids_1 == ids_2

def test_chunk_ids_differ_across_different_sources():
    text = "Same content, different file."

    chunks_a = chunk_text(text, source="doc_a", chunk_size=100, chunk_overlap=10)
    chunks_b = chunk_text(text, source="doc_b", chunk_size=100, chunk_overlap=10)

    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id

def test_no_content_lost_ignoring_whitespace():
    text = "abcdefgh " * 100
    chunks = chunk_text(text, source="doc1", chunk_size=120, chunk_overlap=20)

    original_chars = set(text.replace(" ", ""))
    seen_chars = set("".join(c.text for c in chunks).replace(" ", ""))
    assert original_chars.issubset(seen_chars)