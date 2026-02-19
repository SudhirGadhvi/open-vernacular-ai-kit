from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence

from .token_lid import Token, TokenLang, tokenize


class GujaratiDialect(str, Enum):
    UNKNOWN = "unknown"
    STANDARD = "standard"
    KATHIAWADI = "kathiawadi"
    SURATI = "surati"


@dataclass(frozen=True)
class DialectDetection:
    """
    MVP dialect detection result.

    This is intentionally heuristic: we count marker hits. It is meant to be a lightweight,
    explainable signal you can surface in UX (not a linguistics-grade classifier).
    """

    dialect: GujaratiDialect
    scores: dict[str, int]
    markers_found: dict[str, list[str]]


@dataclass(frozen=True)
class DialectNormalizationResult:
    dialect: GujaratiDialect
    changed: bool
    tokens_in: list[str]
    tokens_out: list[str]


def _lower_if_latin_token(tok: str) -> str:
    # TokenLang isn't available at this layer for raw tokens, so keep it simple:
    # treat ASCII-ish text as case-insensitive for marker matching.
    try:
        return tok.lower()
    except Exception:
        return tok


# NOTE: These marker words and mappings are intentionally small and easy to extend.
# They are based on commonly cited "tadapadi" Kathiawadi words and a couple of Surati examples.
#
# Source used for the Kathiawadi list (human-curated blog post with Gujarati equivalents):
# https://anjani-kd.medium.com/will-these-rural-kathiyawadi-words-disappear-in-the-next-generation-5c91791afd0f
_KATHIAWADI_MARKERS_ROMAN = {
    "kamaad",
    "thaam",
    "lugdaa",
    "vihamo",
    "nihaho",
    "hatanu",
    "karashiyo",
    "thekane",
    "atane",
    "katane",
    "amtho",
    "aabhlu",
    "aaste",
    "oru",
    "oro",
    "ori",
    "ghoide",
    "chenk",
    "jhanli",
    "jhajhu",
    "vadhe",
}

_KATHIAWADI_MARKERS_NATIVE = {
    "કમાડ",
    "ઠામ",
    "લુગડા",
    "વિહામો",
    "નિહાહો",
    "હટાણું",
    "કરાશિયો",
    "ઠેકાણે",
    "અટાણે",
    "કટાણે",
    "અમથો",
    "આભલું",
    "આસ્તે",
    "ઑરો",
    "ઘોડ્ય",
    "છેક",
    "ઝાલી",
    "ઝાઝા",
    "વઢે",
}

_SURATI_MARKERS_ROMAN = {
    # A commonly cited Surati usage: "ghavadaave chhe" ("it itches")
    "ghavadaave",
    "ghavdaave",
}

_SURATI_MARKERS_NATIVE = {
    "ઘવડાવે",
}


# Phrase + token normalization rules (MVP). These are applied on token sequences.
# Output defaults to standard Gujarati script where possible (fits this kit's canonical output).
_KATHIAWADI_PHRASE_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (("vihamo", "khavo"), ["આરામ", "કરવો"]),
    (("nihaho", "khavo"), ["આરામ", "કરવો"]),
    (("malkai", "chhe"), ["શરમાઇ", "છે"]),
    (("વિહામો", "ખાવો"), ["આરામ", "કરવો"]),
    (("નિહાહો", "ખાવો"), ["આરામ", "કરવો"]),
    (("મલકાઈ", "છે"), ["શરમાઇ", "છે"]),
]

_KATHIAWADI_TOKEN_RULES: dict[str, str] = {
    "kamaad": "દરવાજો",
    "thaam": "વાસણ",
    "taap": "ગરમી",
    "lugdaa": "કપડા",
    "unu": "ગરમ",
    "hatanu": "ખરીદી",
    "karashiyo": "લોટો",
    "asal": "સરસ",
    "nishaal": "વિદ્યા",
    "thekane": "જગ્યાએ",
    "atane": "અત્યારે",
    "katane": "ખોટા",
    "amtho": "ખોટો",
    "aabhlu": "આકાશ",
    "aaste": "ધીરે",
    "oru": "નજીક",
    "oro": "નજીક",
    "ori": "નજીક",
    "ghoide": "જેમ",
    "chenk": "દૂર",
    "jhanli": "પકડીને",
    "bakalu": "શાક",
    "jhajhu": "વધારે",
    "vadhe": "ખીજાય",
}

_KATHIAWADI_TOKEN_RULES_NATIVE: dict[str, str] = {
    "કમાડ": "દરવાજો",
    "ઠામ": "વાસણ",
    "તાપ": "ગરમી",
    "લુગડા": "કપડા",
    "ઉણુ": "ગરમ",
    "હટાણું": "ખરીદી",
    "કરાશિયો": "લોટો",
    "અસલ": "સરસ",
    "ઠેકાણે": "જગ્યાએ",
    "અટાણે": "અત્યારે",
    "કટાણે": "ખોટા",
    "અમથો": "ખોટો",
    "આભલું": "આકાશ",
    "આસ્તે": "ધીરે",
    "ઑરો": "નજીક",
    "ઘોડ્ય": "જેમ",
    "છેક": "દૂર",
    "ઝાલી": "પકડીને",
    "બકાલું": "શાક",
    "ઝાઝા": "વધારે",
    "વઢે": "ખીજાય",
}

_SURATI_PHRASE_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (("ghavadaave", "chhe"), ["ખંજવાળ", "આવે", "છે"]),
    (("ghavdaave", "chhe"), ["ખંજવાળ", "આવે", "છે"]),
    (("ઘવડાવે", "છે"), ["ખંજવાળ", "આવે", "છે"]),
]

_SURATI_TOKEN_RULES: dict[str, str] = {
    "ghavadaave": "ખંજવાળ",
    "ghavdaave": "ખંજવાળ",
}

_SURATI_TOKEN_RULES_NATIVE: dict[str, str] = {
    "ઘવડાવે": "ખંજવાળ",
}


def detect_dialect_from_tokens(tokens: Sequence[str]) -> DialectDetection:
    """
    Detect dialect from raw tokens (output of `tokenize()`).
    """

    kath_hits: list[str] = []
    sur_hits: list[str] = []

    for tok in tokens:
        t = _lower_if_latin_token(tok)
        if t in _KATHIAWADI_MARKERS_ROMAN or tok in _KATHIAWADI_MARKERS_NATIVE:
            kath_hits.append(tok)
        if t in _SURATI_MARKERS_ROMAN or tok in _SURATI_MARKERS_NATIVE:
            sur_hits.append(tok)

    scores = {"kathiawadi": len(kath_hits), "surati": len(sur_hits)}
    markers_found = {"kathiawadi": kath_hits, "surati": sur_hits}

    if scores["kathiawadi"] == 0 and scores["surati"] == 0:
        return DialectDetection(
            dialect=GujaratiDialect.UNKNOWN,
            scores=scores,
            markers_found=markers_found,
        )
    if scores["kathiawadi"] > scores["surati"]:
        d = GujaratiDialect.KATHIAWADI
    elif scores["surati"] > scores["kathiawadi"]:
        d = GujaratiDialect.SURATI
    else:
        # Tie: keep it conservative.
        d = GujaratiDialect.UNKNOWN

    return DialectDetection(dialect=d, scores=scores, markers_found=markers_found)


def detect_dialect(text: str) -> DialectDetection:
    return detect_dialect_from_tokens(tokenize(text or ""))


def detect_dialect_from_tagged_tokens(tagged_tokens: Iterable[Token]) -> DialectDetection:
    """
    Detect dialect from the pipeline's tagged tokens.

    We only use Gujarati-ish tokens for marker matching:
    - Gujarati script tokens: match as-is.
    - Gujlish roman tokens: match lowercased text.
    """

    kath_hits: list[str] = []
    sur_hits: list[str] = []

    for t in tagged_tokens:
        if t.lang == TokenLang.GU_NATIVE:
            if t.text in _KATHIAWADI_MARKERS_NATIVE:
                kath_hits.append(t.text)
            if t.text in _SURATI_MARKERS_NATIVE:
                sur_hits.append(t.text)
        elif t.lang == TokenLang.GU_ROMAN:
            low = (t.text or "").lower()
            if low in _KATHIAWADI_MARKERS_ROMAN:
                kath_hits.append(t.text)
            if low in _SURATI_MARKERS_ROMAN:
                sur_hits.append(t.text)

    scores = {"kathiawadi": len(kath_hits), "surati": len(sur_hits)}
    markers_found = {"kathiawadi": kath_hits, "surati": sur_hits}

    if scores["kathiawadi"] == 0 and scores["surati"] == 0:
        return DialectDetection(
            dialect=GujaratiDialect.UNKNOWN,
            scores=scores,
            markers_found=markers_found,
        )
    if scores["kathiawadi"] > scores["surati"]:
        d = GujaratiDialect.KATHIAWADI
    elif scores["surati"] > scores["kathiawadi"]:
        d = GujaratiDialect.SURATI
    else:
        d = GujaratiDialect.UNKNOWN

    return DialectDetection(dialect=d, scores=scores, markers_found=markers_found)


def _apply_phrase_rules(tokens: Sequence[str], rules: list[tuple[tuple[str, ...], list[str]]]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        applied = False
        for pat, rep in rules:
            n = len(pat)
            if i + n > len(tokens):
                continue
            window = tokens[i : i + n]
            # Compare case-insensitively for roman patterns; native patterns are exact.
            if tuple(_lower_if_latin_token(t) for t in window) == pat:
                out.extend(rep)
                i += n
                applied = True
                break
        if not applied:
            out.append(tokens[i])
            i += 1
    return out


def normalize_dialect_tokens(
    tokens: Sequence[str],
    *,
    dialect: Optional[GujaratiDialect] = None,
) -> DialectNormalizationResult:
    """
    Apply MVP dialect normalization rules to a token stream.

    - If `dialect` is None, we auto-detect and normalize only when we're confident enough
      to choose a non-UNKNOWN dialect.
    - Output tokens may switch scripts (roman -> Gujarati) for normalized terms.
    """

    tokens_in = list(tokens)
    det = detect_dialect_from_tokens(tokens_in) if dialect is None else None
    d_eff = det.dialect if det is not None else (dialect or GujaratiDialect.UNKNOWN)

    if d_eff == GujaratiDialect.KATHIAWADI:
        tmp = _apply_phrase_rules([_lower_if_latin_token(t) for t in tokens_in], _KATHIAWADI_PHRASE_RULES)
        out: list[str] = []
        for tok in tmp:
            low = _lower_if_latin_token(tok)
            out.append(_KATHIAWADI_TOKEN_RULES_NATIVE.get(tok, _KATHIAWADI_TOKEN_RULES.get(low, tok)))
        return DialectNormalizationResult(
            dialect=d_eff,
            changed=out != tokens_in,
            tokens_in=tokens_in,
            tokens_out=out,
        )

    if d_eff == GujaratiDialect.SURATI:
        tmp = _apply_phrase_rules([_lower_if_latin_token(t) for t in tokens_in], _SURATI_PHRASE_RULES)
        out = []
        for tok in tmp:
            low = _lower_if_latin_token(tok)
            out.append(_SURATI_TOKEN_RULES_NATIVE.get(tok, _SURATI_TOKEN_RULES.get(low, tok)))
        return DialectNormalizationResult(
            dialect=d_eff,
            changed=out != tokens_in,
            tokens_in=tokens_in,
            tokens_out=out,
        )

    # UNKNOWN / STANDARD: no-op (MVP).
    return DialectNormalizationResult(
        dialect=d_eff,
        changed=False,
        tokens_in=tokens_in,
        tokens_out=tokens_in,
    )

