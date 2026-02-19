from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .token_lid import Token, TokenLang


@dataclass(frozen=True)
class CodeSwitchMetrics:
    """
    Lightweight code-switching metrics for Gujarati-English mixed text.

    This intentionally stays heuristic and dependency-free:
    - We collapse Gujarati (native script + Gujlish roman) into a single "gu" bucket.
    - We ignore TokenLang.OTHER for metrics/switch computations (digits/punct/other scripts).

    The most useful product metric is `cmi` (Code-Mixing Index), plus `n_switch_points`.
    """

    n_tokens_total: int
    n_tokens_considered: int

    n_gu_tokens: int
    n_en_tokens: int

    # Code-Mixing Index (0..100). Higher => more mixed.
    cmi: float

    # Number of language switches between consecutive lexical tokens.
    n_switch_points: int

    # Number of contiguous spans ("runs") of the same language across lexical tokens.
    n_spans: int


def _bucket_lang(lang: TokenLang) -> Optional[str]:
    if lang in (TokenLang.GU_NATIVE, TokenLang.GU_ROMAN):
        return "gu"
    if lang == TokenLang.EN:
        return "en"
    return None


def compute_code_switch_metrics(tagged_tokens: Iterable[Token]) -> CodeSwitchMetrics:
    toks = list(tagged_tokens)

    buckets: list[str] = []
    for t in toks:
        b = _bucket_lang(t.lang)
        if b is None:
            continue
        buckets.append(b)

    n_gu = sum(1 for b in buckets if b == "gu")
    n_en = sum(1 for b in buckets if b == "en")
    n = len(buckets)

    # Gamback & Das-style CMI (simplified):
    # CMI = 100 * (1 - max(count(lang))/N), with N excluding language-independent tokens.
    if n == 0:
        cmi = 0.0
    else:
        cmi = 100.0 * (1.0 - (max(n_gu, n_en) / n))

    n_switch = 0
    n_spans = 0
    prev: Optional[str] = None
    for b in buckets:
        if prev is None:
            n_spans = 1
        elif b != prev:
            n_switch += 1
            n_spans += 1
        prev = b

    return CodeSwitchMetrics(
        n_tokens_total=len(toks),
        n_tokens_considered=n,
        n_gu_tokens=n_gu,
        n_en_tokens=n_en,
        cmi=float(cmi),
        n_switch_points=int(n_switch),
        n_spans=int(n_spans),
    )

