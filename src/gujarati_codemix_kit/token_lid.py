from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import regex as re


class TokenLang(str, Enum):
    EN = "en"
    GU_NATIVE = "gu_native"
    GU_ROMAN = "gu_roman"
    OTHER = "other"


@dataclass(frozen=True)
class Token:
    text: str
    lang: TokenLang


_GUJARATI_CHAR_RE = re.compile(r"[\p{Gujarati}]")
_LATIN_CHAR_RE = re.compile(r"[\p{Latin}]")

# Tokenization that preserves punctuation as separate tokens.
_TOKEN_RE = re.compile(
    r"([\p{L}\p{M}]+|\p{N}+|[^\p{L}\p{M}\p{N}\s])",
    flags=re.VERSION1,
)

_COMMON_GU_ROMAN = {
    # Very common Gujlish tokens that don't contain distinctive clusters.
    "hu",
    "tu",
    "tame",
    "ame",
    "maru",
    "mare",
    "taro",
    "tari",
    "tamaro",
    "tamari",
    "chhe",
    "che",
    "nathi",
    "hato",
    "hati",
    "hase",
    "shu",
    "su",
    "kem",
    "kya",
    "kyare",
    "aaje",
    "kaale",
}


def tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _is_gujarati_script(token: str) -> bool:
    return bool(_GUJARATI_CHAR_RE.search(token))


def _is_latin(token: str) -> bool:
    return bool(_LATIN_CHAR_RE.search(token))


def _looks_like_english(token: str) -> bool:
    # Cheap heuristic: common English function words.
    t = token.lower()
    return t in {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "i",
        "you",
        "we",
        "they",
        "it",
    }


def _looks_like_gujarati_roman(token: str) -> bool:
    """
    Fast heuristic for Gujarati romanization (Gujlish).

    Not perfect; the ML classifier (if present) should override this.
    """
    t = token.lower()
    if len(t) <= 2:
        return t in _COMMON_GU_ROMAN
    if t in _COMMON_GU_ROMAN:
        return True
    clusters = (
        "aa",
        "ee",
        "oo",
        "ai",
        "au",
        "kh",
        "gh",
        "ch",
        "chh",
        "jh",
        "th",
        "dh",
        "ph",
        "bh",
        "sh",
        "gn",
    )
    return any(c in t for c in clusters)


def _model_path() -> Path:
    return Path(__file__).with_name("_data") / "latin_lid.joblib"


@lru_cache(maxsize=1)
def _load_latin_classifier() -> Optional[object]:
    """
    Load a trained sklearn Pipeline, if present and joblib is installed.

    This is optional; the toolkit remains usable without it.
    """
    p = _model_path()
    if not p.exists():
        return None

    try:
        import joblib
    except Exception:
        return None

    try:
        return joblib.load(p)
    except Exception:
        return None


def _latin_predict_is_gu_roman(token: str) -> Optional[bool]:
    clf = _load_latin_classifier()
    if clf is None:
        return None
    try:
        pred = clf.predict([token])
        return bool(pred[0])
    except Exception:
        return None


def detect_token_lang(token: str) -> TokenLang:
    if not token:
        return TokenLang.OTHER

    if _is_gujarati_script(token):
        return TokenLang.GU_NATIVE

    if _is_latin(token):
        if token.lower() in _COMMON_GU_ROMAN:
            return TokenLang.GU_ROMAN

        ml = _latin_predict_is_gu_roman(token)
        if ml is not None:
            return TokenLang.GU_ROMAN if ml else TokenLang.EN

        if _looks_like_english(token):
            return TokenLang.EN
        if _looks_like_gujarati_roman(token):
            return TokenLang.GU_ROMAN
        return TokenLang.EN

    return TokenLang.OTHER


def tag_tokens(tokens: Iterable[str]) -> list[Token]:
    return [Token(text=t, lang=detect_token_lang(t)) for t in tokens]
 
