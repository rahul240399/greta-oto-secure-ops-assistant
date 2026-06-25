from secureops import chunk
from secureops.config import ChunkingCfg


def test_naive_chunk_overlap_and_coverage():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk.naive_chunk(text, size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    # consecutive chunks share the overlap region
    assert chunks[0][-20:] == chunks[1][:20]
    # every character is covered
    assert "".join(c[:80] for c in chunks).startswith(text[:80])


def test_naive_chunk_rejects_bad_size():
    import pytest

    with pytest.raises(ValueError):
        chunk.naive_chunk("x", size=0)


def test_sentence_pack_keeps_sentences_whole():
    text = "First sentence here. Second one follows. Third wraps it up."
    packed = chunk.sentence_pack(text, size=30, overlap=0)
    # no chunk should start or end mid-word for these short sentences
    for c in packed:
        assert c.strip() == c
    assert "First sentence here." in " ".join(packed)


def test_split_advisory_sections_finds_headers():
    text = ("Executive Summary: a bug exists. Risk Evaluation: high. "
            "Mitigations: patch now.")
    sections = chunk.split_advisory_sections(text)
    names = [n for n, _ in sections]
    assert "executive summary" in names
    assert "mitigations" in names


def test_split_advisory_sections_fallback():
    sections = chunk.split_advisory_sections("no known headers in this text")
    assert sections == [("", "no known headers in this text")]


def test_chunk_documents_attaches_metadata():
    docs = [
        {"source": "nist_sp800_82r3.pdf", "page": 5, "doc_type": "nist",
         "text": "A sentence. Another sentence. And a third for good measure."},
        {"source": "CISA: Example", "page": 1, "doc_type": "advisory",
         "text": "Mitigations: do the thing and then the other thing carefully."},
    ]
    cfg = ChunkingCfg(strategy="structure_aware", size=40, overlap=0)
    chunks, metas = chunk.chunk_documents(docs, cfg)
    assert len(chunks) == len(metas)
    assert all("source" in m and "doc_type" in m for m in metas)
    # advisory chunk should carry its section name
    assert any(m.get("section") == "mitigations" for m in metas)
