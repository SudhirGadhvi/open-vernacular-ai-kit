# AGENTS.md

## Project

- Name: `open-vernacular-ai-kit`
- Goal: provide a production-ready vernacular normalization layer for mixed Indian language + English text.
- Primary package path: `src/open_vernacular_ai_kit`
- Primary public interfaces: `CodeMixConfig`, `CodeMixPipeline`, `render_codemix`, `analyze_codemix`, CLI `gck`.

## Engineering Priorities

1. Preserve backward compatibility for SDK and CLI behavior unless explicitly requested otherwise.
2. Keep offline-first safety defaults (`allow_remote_models=False` by default).
3. Keep base installation lightweight; gate heavy dependencies behind extras.
4. Preserve deterministic behavior in normalization/transliteration paths.
5. Add tests for each behavior change before completion.

## Code Constraints

- Edit only `src/open_vernacular_ai_kit` for library changes.
- Keep `CodeMixConfig` JSON roundtrip compatibility (`to_dict`, `from_dict`, `schema_version` policy).
- Keep optional imports lazy and raise clear `OptionalDependencyError` or `OfflinePolicyError` messages.
- Prefer small, composable functions and explicit typed dataclasses for outputs.

## Module Map

- `config.py`: config schema, coercion, validation.
- `pipeline.py`: staged execution and result packaging.
- `token_lid.py`: tokenization and language tagging.
- `transliterate.py`: transliteration backends and exception handling.
- `dialects.py`, `dialect_backends.py`, `dialect_normalizers.py`: dialect detection and normalization.
- `codeswitch.py`: code-mixing metrics.
- `app_flows.py`: WhatsApp parsing and batch CSV/JSONL helpers.
- `rag.py`, `rag_datasets.py`: optional RAG utilities and datasets.
- `eval_harness.py`: reproducible eval suites.
- `demo/streamlit_app.py`: demo UI.

## Required Validation

Run these after meaningful code changes:

```bash
pytest -q
ruff check .
```

Run task-specific checks when relevant:

```bash
gck codemix "maru business plan ready chhe!!!"
gck eval --dataset gujlish --report eval/out/report.json
```

## Docs and Release Hygiene

- Update docs in `docs/` when CLI flags, config fields, workflows, or eval behavior changes.
- Keep `README.md` examples aligned with real outputs.
- Ensure CI workflows continue to pass across the existing Python matrix.
- Keep versioning and release notes aligned with `pyproject.toml` and release workflow.

## Product and GTM Guidance

- Anchor claims to measurable outputs (eval metrics, latency, test coverage, real integrations).
- Prioritize adoption assets:
  - API-ready examples
  - Docker/API deployment path
  - before/after quality evidence
  - short domain case studies (support, ecommerce, chat)
- Prefer one strong Gujarati-first story with clear outcomes before broad language expansion messaging.
