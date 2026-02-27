# RAG Quickstart (v0.5+)

The SDK includes small, optional RAG helpers intended for tiny curated corpora and demos.

## Packaged tiny dataset

```python
from open_vernacular_ai_kit import RagIndex, load_vernacular_facts_tiny

ds = load_vernacular_facts_tiny()

def keyword_embed(texts: list[str]) -> list[list[float]]:
    keys = ["gujarati", "hindi", "tamil", "kannada", "bengali", "marathi"]
    return [[1.0 if k in (t or "").lower() else 0.0 for k in keys] for t in texts]

idx = RagIndex.build(docs=ds.docs, embed_texts=keyword_embed, embedding_model="keywords")
hits = idx.search(
    query="which language is commonly used in gujarat customer support workflows (gujarati)?",
    embed_texts=keyword_embed,
    topk=3,
)
print([h.doc_id for h in hits])
```

## HF embeddings (optional)

```bash
pip install -e ".[rag-embeddings]"
```

```python
from open_vernacular_ai_kit import RagIndex, load_vernacular_facts_tiny, make_hf_embedder

ds = load_vernacular_facts_tiny()
embed = make_hf_embedder(model_id_or_path="/path/to/local/hf/model", allow_remote_models=False)
idx = RagIndex.build(docs=ds.docs, embed_texts=embed, embedding_model="local-model")
```

