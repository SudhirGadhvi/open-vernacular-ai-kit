# Install

## Editable install (recommended for repo usage)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[dev]"
```

## Feature extras

- `indic`: Indic-script normalization + transliteration helpers
- `ml`: optional lightweight ML LID helper
- `fasttext`: optional fastText LID fallback
- `lexicon`: YAML support for user lexicons
- `dialect-ml`: Transformers backends for dialect utilities (optional)
- `api`: FastAPI + uvicorn service wrapper
- `eval`: eval harness dependencies
- `demo`: Streamlit demo
- `rag`: requests-based dataset download helpers
- `rag-embeddings`: torch/transformers embeddings helpers
- `docs`: mkdocs site dependencies

Example (full local dev):

```bash
.venv/bin/pip install -e ".[api,indic,ml,fasttext,lexicon,dialect-ml,eval,demo,rag,rag-embeddings,docs,dev]"
```
