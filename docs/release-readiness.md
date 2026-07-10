# vNext Release Readiness

This page tracks the current release candidate story after `v1.3.0`.

Latest published release on GitHub: `v1.3.0`.
Current release-prep target: `v1.4.0`.

## Candidate Theme

The next release candidate should tell a practical adoption story:

- stronger Gujarati-first proof through a support/ecommerce case study
- improved Hindi beta support-chat normalization
- a copy-paste FastAPI and Docker quickstart
- clearer docs for validation, deployment, and beta-language boundaries

## User-Facing Changes Since v1.3.0

| Area | Change | Evidence |
| --- | --- | --- |
| Gujarati adoption | Added a support/ecommerce case study with raw-vs-normalized examples and committed uplift metrics | `docs/case-studies/gujarati-support-ecommerce.md` |
| Hindi beta | Hardened support/ecommerce chat normalization for tokens such as `nhi`, `kr`, `rha`, `bheja`, `koi`, `dusra`, and `mangwana` | `docs/hindi-beta.md`, `tests/test_language_quality.py` |
| API deployment | Added local FastAPI, Docker, and Python-client smoke path | `docs/api-quickstart.md` |
| Docs navigation | Linked adoption, API, and Hindi beta pages from docs navigation and README | `mkdocs.yml`, `README.md`, `docs/index.md` |

## Release Validation Commands

Run these before preparing a release PR:

```bash
python3 -m pytest -q
ruff check .
.venv/bin/python -m mkdocs build --strict
```

Run task-specific smoke checks:

```bash
gck codemix "maru business plan ready chhe!!!"
gck codemix --language hi --translit-mode sentence "coupon code bilkul work nhi kr rha"
gck eval --dataset language_sentences --language hi
```

If API docs or deployment changed, run:

```bash
python3 -m pytest tests/test_api_service.py -q
```

If benchmark snapshots changed, regenerate and review:

```bash
python3 scripts/snapshot_north_star_metrics.py --output docs/data/north_star_metrics_snapshot.json --iterations 200
python3 scripts/snapshot_downstream_uplift_metrics.py \
  --output docs/data/downstream_uplift_snapshot.json \
  --include-answer-quality \
  --include-prompt-stability
```

The downstream snapshot command may require optional eval dependencies, Sarvam API access, and cached
or live model generations.

## Release PR Checklist

- [x] Confirm whether the target is `v1.4.0`, `v1.3.1`, or a prerelease tag.
- [x] Update `pyproject.toml` and `src/open_vernacular_ai_kit/__init__.py` to the intended version.
- [ ] Keep `CodeMixConfig` JSON roundtrip compatibility unchanged.
- [ ] Confirm `allow_remote_models=False` remains the default.
- [ ] Run the release validation commands above.
- [ ] Open the release PR from `develop` to `main`.
- [ ] After merge to `main`, create and push the annotated `v*` tag.

## Suggested Release Notes Draft

```text
Open Vernacular AI Kit vNext focuses on practical adoption:

- Adds a Gujarati support/ecommerce case study grounded in committed uplift metrics.
- Improves Hindi beta support-chat normalization for noisy ecommerce and support phrasing.
- Adds a copy-paste FastAPI/Docker quickstart for local and deployment smoke checks.
- Keeps offline-first defaults intact and preserves the existing SDK/CLI interfaces.
```
