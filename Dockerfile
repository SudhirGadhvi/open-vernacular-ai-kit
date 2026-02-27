FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN python -m pip install -U pip \
    && python -m pip install -e ".[api,indic,ml,lexicon]"

EXPOSE 8000

CMD ["uvicorn", "open_vernacular_ai_kit.api_service:app", "--host", "0.0.0.0", "--port", "8000"]
