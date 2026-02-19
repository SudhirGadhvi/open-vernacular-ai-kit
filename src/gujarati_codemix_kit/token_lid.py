from __future__ import annotations

import os
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
    confidence: float = 0.0
    reason: str = ""


_GUJARATI_CHAR_RE = re.compile(r"[\p{Gujarati}]")
_LATIN_CHAR_RE = re.compile(r"[\p{Latin}]")
_LATIN_ONLY_RE = re.compile(r"[^\p{Latin}]+", flags=re.VERSION1)

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


def _latin_predict_proba_is_gu_roman(token: str) -> Optional[float]:
    """
    Return P(GU_ROMAN) for the Latin-token classifier if predict_proba is available.
    """
    clf = _load_latin_classifier()
    if clf is None:
        return None
    fn = getattr(clf, "predict_proba", None)
    if fn is None:
        return None
    try:
        proba = fn([token])
        # Shape: (1, n_classes)
        row = list(proba[0])
        classes = list(getattr(clf, "classes_", []))
        # Common cases: classes_ == [False, True] or [0, 1]
        if True in classes:
            idx = classes.index(True)
        elif 1 in classes:
            idx = classes.index(1)
        else:
            # Fallback: assume positive class is last.
            idx = len(row) - 1
        p = float(row[idx])
        if p != p:  # NaN guard
            return None
        return max(0.0, min(1.0, p))
    except Exception:
        return None


def _normalize_latin_key(token: str) -> str:
    # Match lexicon normalization: lower + latin-only.
    return _LATIN_ONLY_RE.sub("", (token or "").strip().lower())


def _resolve_fasttext_model_path(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env = os.getenv("GCK_FASTTEXT_MODEL_PATH")
    if env:
        candidates.append(Path(env).expanduser())
    # Common local filenames/locations users tend to keep it in.
    candidates.append(Path("lid.176.ftz"))
    candidates.append(Path.home() / ".cache" / "gujarati_codemix_kit" / "lid.176.ftz")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


@lru_cache(maxsize=2)
def _load_fasttext_model(path_str: str) -> Optional[object]:
    try:
        import fasttext  # type: ignore[reportMissingImports]
    except Exception:
        return None
    try:
        return fasttext.load_model(path_str)
    except Exception:
        return None


def _fasttext_predict_language(token: str, *, model_path: Optional[str]) -> Optional[tuple[str, float]]:
    p = _resolve_fasttext_model_path(model_path)
    if p is None:
        return None
    model = _load_fasttext_model(str(p))
    if model is None:
        return None
    try:
        labels, probs = model.predict(token, k=1)
        if not labels or not probs:
            return None
        lab = str(labels[0]).strip()
        prob = float(probs[0])
        if lab.startswith("__label__"):
            lab = lab[len("__label__") :]
        return lab, max(0.0, min(1.0, prob))
    except Exception:
        return None


def analyze_token(
    token: str,
    *,
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
) -> Token:
    """
    Token-level LID with lightweight confidence + reason codes.

    Reasons are intentionally stable strings since they can be surfaced in logs/analysis.
    """
    if not token:
        return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="empty")

    if _is_gujarati_script(token):
        return Token(text=token, lang=TokenLang.GU_NATIVE, confidence=1.0, reason="gujarati_script")

    if not _is_latin(token):
        # Includes digits and punctuation and other scripts.
        if token.isdigit():
            return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="digits")
        return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="non_latin")

    t_lower = token.lower()
    norm = _normalize_latin_key(token)
    if lexicon_keys and norm in lexicon_keys:
        return Token(text=token, lang=TokenLang.GU_ROMAN, confidence=0.98, reason="user_lexicon")

    if t_lower in _COMMON_GU_ROMAN:
        return Token(text=token, lang=TokenLang.GU_ROMAN, confidence=0.95, reason="common_gujlish")

    # If present, the sklearn model is the strongest signal for Gujlish vs English.
    p_gu = _latin_predict_proba_is_gu_roman(token)
    if p_gu is not None:
        if p_gu >= 0.5:
            return Token(text=token, lang=TokenLang.GU_ROMAN, confidence=p_gu, reason="ml_latin_lid")
        return Token(text=token, lang=TokenLang.EN, confidence=1.0 - p_gu, reason="ml_latin_lid")

    ml = _latin_predict_is_gu_roman(token)
    if ml is not None:
        return Token(
            text=token,
            lang=TokenLang.GU_ROMAN if ml else TokenLang.EN,
            confidence=0.7,
            reason="ml_latin_lid_no_proba",
        )

    # Optional fastText: use as a *fallback* signal (primarily to confidently identify English).
    ft = _fasttext_predict_language(token, model_path=fasttext_model_path)
    if ft is not None:
        lab, prob = ft
        if lab == "en" and prob >= 0.85 and len(norm) >= 3:
            return Token(text=token, lang=TokenLang.EN, confidence=prob, reason="fasttext_en")

    if _looks_like_english(token):
        return Token(text=token, lang=TokenLang.EN, confidence=0.8, reason="english_function_word")
    if _looks_like_gujarati_roman(token):
        return Token(text=token, lang=TokenLang.GU_ROMAN, confidence=0.6, reason="gujlish_clusters")
    return Token(text=token, lang=TokenLang.EN, confidence=0.5, reason="default_en")


def detect_token_lang(token: str) -> TokenLang:
    return analyze_token(token).lang


def tag_tokens(
    tokens: Iterable[str],
    *,
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
) -> list[Token]:
    return [
        analyze_token(t, lexicon_keys=lexicon_keys, fasttext_model_path=fasttext_model_path)
        for t in tokens
    ]
 
