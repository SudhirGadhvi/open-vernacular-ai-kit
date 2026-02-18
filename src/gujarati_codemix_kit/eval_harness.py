from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import regex as re

from .codemix_render import render_codemix
from .normalize import normalize_text
from .token_lid import TokenLang, tag_tokens, tokenize

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    text_column: str


_GUJLISH_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="in22",
        url="https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20IN22.csv",
        text_column="guj",
    ),
    DatasetSpec(
        name="xnli",
        url="https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20XNLI.csv",
        text_column="guj",
    ),
]


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "gujarati-codemix-kit"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("requests is required for eval harness; install with: pip install -e '.[eval]'") from e

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _iter_texts_from_csv(path: Path, *, text_column: str) -> Iterable[str]:
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for eval harness; install with: pip install -e '.[eval]'") from e

    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise RuntimeError(f"Missing expected column '{text_column}' in {path.name}")
    for v in df[text_column].astype(str).tolist():
        yield v


def _has_gujarati(text: str) -> bool:
    return bool(_GUJARATI_RE.search(text))


def _analyze_one(text: str, *, topk: int = 1) -> dict[str, Any]:
    raw = text
    norm = normalize_text(raw)
    out = render_codemix(norm, topk=topk)

    toks = tokenize(norm)
    tagged = tag_tokens(toks)
    gu_roman = [t.text for t in tagged if t.lang == TokenLang.GU_ROMAN]

    # How many roman tokens changed after codemix render?
    changed = 0
    for t in gu_roman:
        if t and t in out:
            # If token remains in output as-is, it's likely not transliterated.
            continue
        changed += 1

    return {
        "raw": raw,
        "normalized": norm,
        "codemix": out,
        "has_gujarati_raw": _has_gujarati(raw),
        "has_gujarati_codemix": _has_gujarati(out),
        "n_tokens": len(toks),
        "n_gu_roman_tokens": len(gu_roman),
        "n_gu_roman_tokens_changed_est": changed,
    }


def run_eval(dataset: str = "gujlish", *, topk: int = 1, max_rows: Optional[int] = 2000) -> dict[str, Any]:
    """
    Run a lightweight eval that answers: does codemix rendering produce Gujarati script
    from Gujlish inputs, and how often?

    This is not a translation-quality benchmark. It's an MVP harness that helps show
    measurable normalization effects.
    """
    if dataset != "gujlish":
        raise ValueError("Only dataset='gujlish' is supported right now")

    cache_dir = _default_cache_dir()

    per_split: dict[str, Any] = {}
    for spec in _GUJLISH_SPECS:
        csv_path = cache_dir / f"{dataset}-{spec.name}.csv"
        _download(spec.url, csv_path)

        rows: list[dict[str, Any]] = []
        for i, text in enumerate(_iter_texts_from_csv(csv_path, text_column=spec.text_column)):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(_analyze_one(text, topk=topk))

        n = len(rows) or 1
        has_gu_raw = sum(1 for r in rows if r["has_gujarati_raw"])
        has_gu_out = sum(1 for r in rows if r["has_gujarati_codemix"])
        n_gu_roman = sum(int(r["n_gu_roman_tokens"]) for r in rows)
        n_changed = sum(int(r["n_gu_roman_tokens_changed_est"]) for r in rows)

        per_split[spec.name] = {
            "n_rows": len(rows),
            "pct_has_gujarati_raw": has_gu_raw / n,
            "pct_has_gujarati_codemix": has_gu_out / n,
            "n_gu_roman_tokens": n_gu_roman,
            "pct_gu_roman_tokens_changed_est": (n_changed / n_gu_roman) if n_gu_roman else 0.0,
            "examples": rows[:8],
        }

    return {
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "topk": int(topk),
        "max_rows": None if max_rows is None else int(max_rows),
        "splits": per_split,
    }

