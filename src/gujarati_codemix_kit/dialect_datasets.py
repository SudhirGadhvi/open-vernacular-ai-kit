from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dialects import GujaratiDialect


@dataclass(frozen=True)
class DialectIdExample:
    """
    One labeled sample for dialect identification.

    Schema (JSONL):
      {"text": "...", "dialect": "kathiawadi", "source": "...", "meta": {...}}
    """

    text: str
    dialect: GujaratiDialect
    source: str = "unknown"
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class DialectNormalizationExample:
    """
    One labeled sample for dialect normalization.

    Schema (JSONL):
      {"input": "...", "dialect": "kathiawadi", "expected": "...", "source": "...", "meta": {...}}
    """

    input: str
    dialect: GujaratiDialect
    expected: str
    source: str = "unknown"
    meta: dict[str, Any] | None = None


def _parse_dialect(x: Any) -> GujaratiDialect:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return GujaratiDialect(s)  # type: ignore[arg-type]
    except Exception:
        return GujaratiDialect.UNKNOWN


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            yield json.loads(s)


def load_dialect_id_jsonl(path: str | Path) -> list[DialectIdExample]:
    out: list[DialectIdExample] = []
    for rec in iter_jsonl(path):
        out.append(
            DialectIdExample(
                text=str(rec.get("text", "") or ""),
                dialect=_parse_dialect(rec.get("dialect")),
                source=str(rec.get("source", "unknown") or "unknown"),
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
            )
        )
    return out


def load_dialect_normalization_jsonl(path: str | Path) -> list[DialectNormalizationExample]:
    out: list[DialectNormalizationExample] = []
    for rec in iter_jsonl(path):
        out.append(
            DialectNormalizationExample(
                input=str(rec.get("input", "") or ""),
                dialect=_parse_dialect(rec.get("dialect")),
                expected=str(rec.get("expected", "") or ""),
                source=str(rec.get("source", "unknown") or "unknown"),
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
            )
        )
    return out


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dump_dialect_id_jsonl(path: str | Path, examples: Iterable[DialectIdExample]) -> None:
    def iter_records() -> Iterable[dict[str, Any]]:
        for ex in examples:
            yield {
                "text": ex.text,
                "dialect": ex.dialect.value,
                "source": ex.source,
                "meta": ex.meta or {},
            }

    write_jsonl(path, iter_records())


def dump_dialect_normalization_jsonl(
    path: str | Path, examples: Iterable[DialectNormalizationExample]
) -> None:
    def iter_records() -> Iterable[dict[str, Any]]:
        for ex in examples:
            yield {
                "input": ex.input,
                "dialect": ex.dialect.value,
                "expected": ex.expected,
                "source": ex.source,
                "meta": ex.meta or {},
            }

    write_jsonl(path, iter_records())


def packaged_data_path(filename: str) -> Path:
    """
    Return a file path under the package `_data` directory.

    This is primarily used for tiny demo/eval datasets (not large corpora).
    """

    return Path(__file__).with_name("_data") / filename

