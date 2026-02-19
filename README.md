 # Gujarati CodeMix Kit

[![CI](https://github.com/SudhirGadhvi/gujarati-codemix-kit/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/SudhirGadhvi/gujarati-codemix-kit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://sudhirgadhvi.github.io/gujarati-codemix-kit/)
[![PyPI](https://img.shields.io/pypi/v/gujarati-codemix-kit.svg)](https://pypi.org/project/gujarati-codemix-kit/)
[![Python](https://img.shields.io/pypi/pyversions/gujarati-codemix-kit.svg)](https://pypi.org/project/gujarati-codemix-kit/)
 
 `gujarati-codemix-kit` is a small SDK + CLI for cleaning up Gujarati-English code-mixed text,
 especially messy WhatsApp-style inputs where Gujarati might appear in:
 
 - Gujarati script (ગુજરાતી)
 - Romanized Gujarati (Gujlish)
 - Mixed script in the same sentence
 
 The goal is to normalize text *before* sending it to downstream models (Sarvam-M / Mayura /
 Sarvam-Translate), and to postprocess certain outputs (e.g., Saaras `codemix`).

This repo is alpha-quality but SDK-first: the public API centers on `CodeMixConfig` + `CodeMixPipeline`.
 
 ## Install
 
 For full functionality (recommended):
 
 ```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[indic,ml,eval,demo,dev,dialect-ml,rag]"
 ```
 
 Minimal (CLI + basic heuristics only):
 
 ```bash
 pip install -e .
 ```
 
 ## CLI
 
 Normalize text:
 
 ```bash
 gck normalize "મારું business plan ready છે!!!"
 ```
 
 Render clean code-mix (Gujarati tokens in Gujarati script, English preserved):
 
 ```bash
 gck codemix "maru business plan ready chhe!!!"
 ```
 
Canonical output format:

- Gujarati stays in Gujarati script
- English stays in Latin
- Gujlish (romanized Gujarati) tokens are transliterated to Gujarati script when possible

Quick success metric (% Gujlish tokens transliterated):

```bash
gck codemix --stats "maru business plan ready chhe!!!" 1>/dev/null
```

Run eval (downloads public Gujlish eval CSVs into `~/.cache/gujarati-codemix-kit`):

```bash
gck eval --dataset gujlish --report eval/out/report.json
```

Dialect evals (uses a tiny packaged JSONL by default, or provide your own):

```bash
gck eval --dataset dialect_id
gck eval --dataset dialect_normalization
```

 ## Demo (Streamlit)
 
 ```bash
 streamlit run demo/streamlit_app.py
 ```
 
 If you export `SARVAM_API_KEY`, the demo can optionally call Sarvam APIs.

## RAG Utilities (v0.5)

v0.5 adds small, optional RAG helpers intended for tiny curated corpora and demos:

- `RagIndex`: build a small embeddings index and do top-k retrieval
- `load_gujarat_facts_tiny()`: packaged mini dataset (docs + queries) for quick recall evals and demos
- `download_gujarat_facts_dataset(...)`: opt-in download helper (URLs required; offline-first by default)

To enable HF embedding models:

```bash
.venv/bin/pip install -e ".[rag-embeddings]"
```

Example (keyword embedder, no ML deps):

```python
from gujarati_codemix_kit import RagIndex, load_gujarat_facts_tiny

ds = load_gujarat_facts_tiny()

def keyword_embed(texts: list[str]) -> list[list[float]]:
    keys = ["અમદાવાદ", "રાજધાની", "નવરાત્રી", "શિયાળ", "ગિર", "ડાયમંડ"]
    return [[1.0 if k in (t or "") else 0.0 for k in keys] for t in texts]

idx = RagIndex.build(docs=ds.docs, embed_texts=keyword_embed, embedding_model="keywords")
hits = idx.search(query="ગુજરાતની રાજધાની કઈ છે?", embed_texts=keyword_embed, topk=3)
print([h.doc_id for h in hits])
```

## Dialects (Full SDK)

Dialect support is offline-first and pluggable:

- Dialect ID backends: `heuristic` (default), `transformers` (fine-tuned model), `none`
- Dialect normalization backends: `heuristic` (rules), `seq2seq` (optional), `auto` (rules + optional seq2seq)

Safety default: remote HuggingFace model downloads are disabled unless you explicitly enable them:

- SDK: `allow_remote_models=False` (default)
- Demo: "Allow remote model downloads" checkbox (off by default)

Example (heuristic dialect normalization gated by confidence):

```python
from gujarati_codemix_kit import analyze_codemix

a = analyze_codemix(
    "kamaad thaalu rakhje",
    dialect_backend="heuristic",
    dialect_normalize=True,
    dialect_min_confidence=0.7,
)
print(a.codemix)
```

## Training (Optional)

This repo includes simple training scripts (you provide data):

```bash
python3 scripts/train_dialect_id.py --train path/to/dialect_id_train.jsonl --output-dir out/dialect_id
python3 scripts/train_dialect_normalizer.py --train path/to/dialect_norm_train.jsonl --output-dir out/dialect_norm
```
 
 ## Disclaimer
 
This is alpha software. Core code-mix rendering is designed to be stable, but dialect detection and
normalization are limited by available labeled data and model choices.
 
