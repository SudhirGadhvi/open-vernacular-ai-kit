 # Gujarati CodeMix Kit

[![CI](https://github.com/SudhirGadhvi/gujarati-codemix-kit/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/SudhirGadhvi/gujarati-codemix-kit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://sudhirgadhvi.github.io/gujarati-codemix-kit/)
[![PyPI](https://img.shields.io/pypi/v/gujarati-codemix-kit.svg)](https://pypi.org/project/gujarati-codemix-kit/)
[![Python](https://img.shields.io/pypi/pyversions/gujarati-codemix-kit.svg)](https://pypi.org/project/gujarati-codemix-kit/)

`gujarati-codemix-kit` is an SDK + CLI (plus a web demo) for normalizing Gujarati-English code-mixed
text. It is designed for WhatsApp-style inputs where Gujarati may appear as:

- Gujarati script (ગુજરાતી)
- Romanized Gujarati (Gujlish)
- Mixed script in the same sentence

Core goal: produce a stable, canonical form before sending text to downstream systems (LLMs, search,
routing, analytics).

Canonical output format:

- Gujarati stays Gujarati script
- English stays Latin
- Gujlish tokens are transliterated to Gujarati script when possible

The project is SDK-first: the stable surface centers on `CodeMixConfig` + `CodeMixPipeline`.

## Install

From PyPI:

```bash
pip install gujarati-codemix-kit
```

From source (editable, recommended for dev):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[dev]"
```

Common extras:

```bash
.venv/bin/pip install -e ".[indic,demo,eval,rag]"
```

## Webpage Demo (Streamlit)

Run locally:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[demo,indic]"
streamlit run demo/streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

Optional (to enable Sarvam comparison in the demo):

```bash
.venv/bin/pip install -e ".[sarvam]"
export SARVAM_API_KEY="..."
```

Hosting notes (Streamlit Community Cloud / Spaces) live in `demo/README.md`.

## CLI

Normalize text:

```bash
gck normalize "મારું business plan ready છે!!!"
```

Render canonical code-mix:

```bash
gck codemix "maru business plan ready chhe!!!"
```

Stats (success metric = % Gujlish tokens transliterated):

```bash
gck codemix --stats "maru business plan ready chhe!!!" 1>/dev/null
```

Eval harness (optional deps):

```bash
gck eval --dataset gujlish --report eval/out/report.json
gck eval --dataset retrieval
gck eval --dataset dialect_id
gck eval --dataset dialect_normalization
```

## SDK

```python
from gujarati_codemix_kit import render_codemix

print(render_codemix("maru business plan ready chhe!!!", translit_mode="sentence"))
```

Pipeline API (recommended for repeated use):

```python
from gujarati_codemix_kit import CodeMixConfig, CodeMixPipeline

pipe = CodeMixPipeline(config=CodeMixConfig(translit_mode="sentence"))
res = pipe.run("hu aaje office jaish!!")
print(res.codemix)
```

## RAG Utilities

Optional helpers for tiny curated corpora and demos:

- `RagIndex`: build a small embeddings index and do top-k retrieval
- `load_gujarat_facts_tiny()`: packaged mini dataset (docs + queries)
- `download_gujarat_facts_dataset(...)`: opt-in download helper (URLs required)

HF embeddings (optional):

```bash
.venv/bin/pip install -e ".[rag-embeddings]"
```

## Dialects (Optional)

Dialect utilities are offline-first and pluggable:

- Dialect ID backends: `heuristic` (default), `transformers` (fine-tuned model), `none`
- Dialect normalization backends: `heuristic` (rules), `seq2seq` (optional), `auto` (rules + optional seq2seq)

Safety default: remote HuggingFace model downloads are disabled unless you explicitly enable them:

- SDK: `allow_remote_models=False` (default)
- Demo: “Allow remote model downloads” checkbox (off by default)

## Docs

Project docs are published at:

- https://sudhirgadhvi.github.io/gujarati-codemix-kit/

Build/preview locally:

```bash
.venv/bin/pip install -e ".[docs]"
mkdocs serve
```

## Training (Optional)

This repo includes simple training scripts (you provide data):

```bash
python3 scripts/train_dialect_id.py --train path/to/dialect_id_train.jsonl --output-dir out/dialect_id
python3 scripts/train_dialect_normalizer.py --train path/to/dialect_norm_train.jsonl --output-dir out/dialect_norm
```

## Disclaimer

The API is intended to be stable, but output quality depends on the available backends, heuristics,
and the kind of mixed text you process. Dialect components in particular depend on labeled data and
model choices.
