from __future__ import annotations

from gujarati_codemix_kit.rag import RagIndex
from gujarati_codemix_kit.rag_datasets import load_gujarat_facts_tiny


def _keyword_embed(texts: list[str]) -> list[list[float]]:
    # Deterministic tiny embedder for unit tests (no optional deps).
    #
    # This is NOT a semantic embedder; it only checks presence of a few Gujarati keywords
    # that appear in the packaged tiny dataset.
    # Pick keywords that appear in both the query text and the relevant doc text.
    keys = ["સાબરમતી", "રાજધાની", "નવરાત્રી", "શિયાળ", "ગિર", "ડાયમંડ"]
    out: list[list[float]] = []
    for t in texts:
        s = t or ""
        out.append([1.0 if k in s else 0.0 for k in keys])
    return out


def test_load_gujarat_facts_tiny_has_docs_and_queries() -> None:
    ds = load_gujarat_facts_tiny()
    assert ds.name == "gujarat_facts_tiny"
    assert ds.source == "packaged"
    assert len(ds.docs) >= 8
    assert len(ds.queries) >= 6


def test_rag_index_search_and_recall_at_1() -> None:
    ds = load_gujarat_facts_tiny()
    idx = RagIndex.build(docs=ds.docs, embed_texts=_keyword_embed, embedding_model="test-keywords")

    # Spot check: Sabarmati query should retrieve Sabarmati doc at rank-1.
    res = idx.search(query="અમદાવાદમાંથી કઈ નદી પસાર થાય છે?", embed_texts=_keyword_embed, topk=3)
    assert res
    assert res[0].doc_id == "doc_sabarmati"

    # With this keyword embedder, the tiny dataset should have perfect recall@1.
    assert idx.recall_at_k(queries=ds.queries, embed_texts=_keyword_embed, k=1) == 1.0


def test_rag_index_json_roundtrip(tmp_path) -> None:
    ds = load_gujarat_facts_tiny()
    idx = RagIndex.build(docs=ds.docs, embed_texts=_keyword_embed, embedding_model="test-keywords")

    p = tmp_path / "idx.json"
    idx.save_json(p)
    idx2 = RagIndex.load_json(p)
    assert idx2.embedding_model == "test-keywords"
    assert len(idx2.docs) == len(idx.docs)
    assert len(idx2.doc_embeddings) == len(idx.doc_embeddings)

    res = idx2.search(query="ડાયમંડ ઉદ્યોગ માટે કયું શહેર જાણીતું છે?", embed_texts=_keyword_embed, topk=3)
    assert res
    assert res[0].doc_id == "doc_surat"

