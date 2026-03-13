from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .codemix_render import render_codemix
from .errors import InvalidConfigError
from .normalize import normalize_text
from .sarvam_review import SarvamTeacherReviewedRecord


@dataclass(frozen=True)
class LanguageSentenceCaseRecord:
    language: str
    raw: str
    expected: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "language": self.language,
            "raw": self.raw,
            "expected": self.expected,
            "source": self.source,
        }


def _contains_gujarati(text: str) -> bool:
    return any("\u0A80" <= ch <= "\u0AFF" for ch in text)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def infer_sentence_case_language(reviewed: SarvamTeacherReviewedRecord) -> str:
    if reviewed.candidate.language_hint in {"gu", "hi"}:
        return reviewed.candidate.language_hint

    expected = reviewed.reviewed_expected
    has_gu = _contains_gujarati(expected)
    has_hi = _contains_devanagari(expected)
    if has_gu and not has_hi:
        return "gu"
    if has_hi and not has_gu:
        return "hi"
    raise InvalidConfigError(
        f"Could not infer target language for sentence case: {reviewed.candidate.input!r}"
    )


def load_language_sentence_case_records(path: str | Path) -> list[LanguageSentenceCaseRecord]:
    out: list[LanguageSentenceCaseRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue
            out.append(
                LanguageSentenceCaseRecord(
                    language=str(rec.get("language", "gu") or "gu").strip().lower(),
                    raw=str(rec.get("raw", "") or ""),
                    expected=str(rec.get("expected", "") or ""),
                    source=str(rec.get("source", "unknown") or "unknown"),
                )
            )
    return out


def dump_language_sentence_case_records(
    path: str | Path, rows: Iterable[LanguageSentenceCaseRecord]
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def promote_sentence_cases_from_review(
    reviewed_rows: Iterable[SarvamTeacherReviewedRecord],
    *,
    existing_rows: Iterable[LanguageSentenceCaseRecord],
    source_suffix: str = "sarvam_review",
    require_pass: bool = True,
) -> tuple[list[LanguageSentenceCaseRecord], dict[str, Any]]:
    existing = list(existing_rows)
    index = {(row.language, row.raw): row for row in existing}
    additions: list[LanguageSentenceCaseRecord] = []
    duplicates_same = 0
    duplicates_conflict: list[dict[str, str]] = []
    skipped_non_sentence = 0
    validation_failures: list[dict[str, str]] = []

    for reviewed in reviewed_rows:
        if reviewed.review_action != "accept_sentence_case":
            skipped_non_sentence += 1
            continue

        lang = infer_sentence_case_language(reviewed)
        raw = reviewed.candidate.input
        expected = reviewed.reviewed_expected
        source = f"{reviewed.candidate.source}:{source_suffix}"
        candidate = LanguageSentenceCaseRecord(
            language=lang,
            raw=raw,
            expected=expected,
            source=source,
        )

        if require_pass:
            got = render_codemix(candidate.raw, language=candidate.language, translit_mode="sentence")
            if normalize_text(got) != normalize_text(candidate.expected):
                validation_failures.append(
                    {
                        "language": candidate.language,
                        "raw": candidate.raw,
                        "expected": candidate.expected,
                        "got": got,
                    }
                )
                continue

        key = (candidate.language, candidate.raw)
        existing_row = index.get(key)
        if existing_row is not None:
            if existing_row.expected == candidate.expected:
                duplicates_same += 1
                continue
            duplicates_conflict.append(
                {
                    "language": candidate.language,
                    "raw": candidate.raw,
                    "expected_existing": existing_row.expected,
                    "expected_reviewed": candidate.expected,
                }
            )
            continue

        additions.append(candidate)
        index[key] = candidate

    merged = existing + additions
    report = {
        "n_existing": len(existing),
        "n_added": len(additions),
        "n_duplicates_same": duplicates_same,
        "n_duplicates_conflict": len(duplicates_conflict),
        "n_skipped_non_sentence": skipped_non_sentence,
        "n_validation_failures": len(validation_failures),
        "conflicts": duplicates_conflict,
        "validation_failures": validation_failures,
        "added_examples": [row.to_dict() for row in additions[:10]],
    }
    return merged, report
