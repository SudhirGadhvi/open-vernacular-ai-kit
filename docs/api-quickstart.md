# API Quickstart

This quickstart gives you a local smoke path for the optional FastAPI service. It keeps the default
runtime offline-first: remote model usage remains opt-in through `CodeMixConfig`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[api]"
```

Use heavier extras only when your deployment needs them:

```bash
.venv/bin/pip install -e ".[api,indic,ml,lexicon]"
```

## Run The Service

```bash
.venv/bin/uvicorn open_vernacular_ai_kit.api_service:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/healthz
```

Expected shape:

```json
{"ok":true,"schema_version":1}
```

## Smoke Gujarati Code-Mix

```bash
curl -s http://localhost:8000/codemix \
  -H 'content-type: application/json' \
  -d '{"text":"maru order status shu chhe?","config":{"language":"gu","translit_mode":"sentence"}}'
```

Expected response fields:

- `schema_version`
- `language`
- `codemix`
- `transliteration_backend`
- token/transliteration counts

The `codemix` field should preserve English support terms while rendering confident Gujarati tokens
in Gujarati script.

## Smoke Hindi Beta

```bash
curl -s http://localhost:8000/codemix \
  -H 'content-type: application/json' \
  -d '{"text":"coupon code bilkul work nhi kr rha","config":{"language":"hi","translit_mode":"sentence"}}'
```

Expected `codemix` field:

```text
coupon code बिल्कुल work नहीं कर रहा
```

## Python Client

```python
import requests


def render_codemix(text: str, *, language: str = "gu") -> str:
    response = requests.post(
        "http://localhost:8000/codemix",
        json={
            "text": text,
            "config": {
                "language": language,
                "translit_mode": "sentence",
            },
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["codemix"]


print(render_codemix("maru payment nathi thayu"))
print(render_codemix("coupon code bilkul work nhi kr rha", language="hi"))
```

## Docker Smoke

Build and run the local image:

```bash
docker build -t ovak-api:local .
docker run --rm -p 8000:8000 ovak-api:local
```

From another terminal:

```bash
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/codemix \
  -H 'content-type: application/json' \
  -d '{"text":"maru order ready chhe","config":{"language":"gu","translit_mode":"sentence"}}'
```

## Production Checklist

- Pin the image tag or package version you deploy.
- Keep `allow_remote_models` unset or `false` unless you intentionally allow model downloads.
- Use `/healthz` for liveness and readiness checks.
- Log `schema_version`, `language`, `transliteration_backend`, and token counts for rollout analysis.
- Add your own domain golden cases before changing language profile behavior in production.
