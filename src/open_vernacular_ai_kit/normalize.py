from __future__ import annotations

import unicodedata

import regex as re

from .errors import InvalidConfigError

_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")  # ZW*, WORD JOINER, BOM
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.:;!?])")
_REPEATED_PUNCT_RE = re.compile(r"([!?])\1{2,}")
_REPEATED_DOT_RE = re.compile(r"\.{4,}")
_PIPE_DANDA_RE = re.compile(r"(?<=\S)\s*\|\s*(?=\S)")

# Basic punctuation normalization. Keep it conservative; the goal is stable text for models.
_PUNCT_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",  # NBSP
    "\u2026": "...",  # ellipsis
}

_GUJARATI_DIGITS = str.maketrans(
    {
        "૦": "0",
        "૧": "1",
        "૨": "2",
        "૩": "3",
        "૪": "4",
        "૫": "5",
        "૬": "6",
        "૭": "7",
        "૮": "8",
        "૯": "9",
    }
)


def _try_indic_normalize_gu(text: str) -> str:
    """
    Optional Gujarati normalization via Indic NLP Library.

    This stays best-effort: if indicnlp isn't installed, we fall back to simpler rules.
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    except Exception:
        return text

    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("gu", remove_nuktas=False)
    return normalizer.normalize(text)


def normalize_text(text: str, *, numerals: str = "keep") -> str:
    """
    Normalize vernacular/English code-mixed text for downstream models.

    - Unicode NFKC
    - Strip zero-width chars
    - Normalize quotes/NBSP
    - Normalize some punctuation (ellipsis, repeated !/?)
    - Optionally normalize Gujarati digits to ASCII
    - Collapse whitespace
    - Optional Indic NLP Gujarati normalization (if available)
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _ZW_RE.sub("", text)
    for k, v in _PUNCT_MAP.items():
        text = text.replace(k, v)

    # Normalize a common "poor man's danda" used in Indic typing.
    text = _PIPE_DANDA_RE.sub("।", text)

    # Gujarati-specific canonicalization (optional)
    text = _try_indic_normalize_gu(text)

    if numerals not in {"keep", "ascii"}:
        raise InvalidConfigError("numerals must be one of: keep, ascii")
    if numerals == "ascii":
        text = text.translate(_GUJARATI_DIGITS)

    # Punctuation tidying.
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)
    text = _REPEATED_DOT_RE.sub("...", text)

    # Collapse whitespace late, after other transforms.
    text = _WS_RE.sub(" ", text).strip()
    return text
 
