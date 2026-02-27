from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence

import regex as re

from .token_lid import Token, TokenLang, tokenize


class GujaratiDialect(str, Enum):
    UNKNOWN = "unknown"
    STANDARD = "standard"
    KATHIAWADI = "kathiawadi"
    SURATI = "surati"
    CHAROTAR = "charotar"
    NORTH_GUJARAT = "north_gujarat"


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
    backend: str = "heuristic"
    confidence: float = 0.0


@dataclass(frozen=True)
class DialectNormalizationResult:
    dialect: GujaratiDialect
    changed: bool
    tokens_in: list[str]
    tokens_out: list[str]
    backend: str = "heuristic"


def _lower_if_latin_token(tok: str) -> str:
    # TokenLang isn't available at this layer for raw tokens, so keep it simple:
    # treat ASCII-ish text as case-insensitive for marker matching.
    try:
        return tok.lower()
    except Exception:
        return tok


_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)


def _is_native_pattern(pat: tuple[str, ...]) -> bool:
    return any(_GUJARATI_RE.search(p or "") for p in pat)


def _roman_lang_ok(t: Token) -> bool:
    return t.lang == TokenLang.GU_ROMAN


def _native_lang_ok(t: Token) -> bool:
    return t.lang == TokenLang.GU_NATIVE


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
        # If we see Gujarati script but no specific dialect markers, treat as standard Gujarati.
        has_gu_script = any(_GUJARATI_RE.search(tok or "") for tok in tokens)
        return DialectDetection(
            dialect=GujaratiDialect.STANDARD if has_gu_script else GujaratiDialect.UNKNOWN,
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
        # If the text contains Gujarati-ish tokens but no markers, call it standard.
        has_gu = any(t.lang in {TokenLang.GU_NATIVE, TokenLang.GU_ROMAN} for t in tagged_tokens)
        return DialectDetection(
            dialect=GujaratiDialect.STANDARD if has_gu else GujaratiDialect.UNKNOWN,
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


def normalize_dialect_tagged_tokens(
    tagged_tokens: Sequence[Token],
    *,
    dialect: Optional[GujaratiDialect] = None,
) -> DialectNormalizationResult:
    """
    Dialect normalization constrained by token LID.

    Key property: we do NOT rewrite tokens tagged as English (or OTHER), even if they
    coincidentally match a roman marker string.
    """

    tokens_in = [t.text for t in tagged_tokens]
    if not tokens_in:
        d_eff = dialect or GujaratiDialect.UNKNOWN
        return DialectNormalizationResult(
            dialect=d_eff,
            changed=False,
            tokens_in=[],
            tokens_out=[],
        )

    det = detect_dialect_from_tagged_tokens(tagged_tokens) if dialect is None else None
    d_eff = det.dialect if det is not None else (dialect or GujaratiDialect.UNKNOWN)

    if d_eff not in {GujaratiDialect.KATHIAWADI, GujaratiDialect.SURATI}:
        return DialectNormalizationResult(
            dialect=d_eff,
            changed=False,
            tokens_in=tokens_in,
            tokens_out=tokens_in,
        )

    if d_eff == GujaratiDialect.KATHIAWADI:
        phrase_rules = _KATHIAWADI_PHRASE_RULES
        token_rules = _KATHIAWADI_TOKEN_RULES
        token_rules_native = _KATHIAWADI_TOKEN_RULES_NATIVE
    else:
        phrase_rules = _SURATI_PHRASE_RULES
        token_rules = _SURATI_TOKEN_RULES
        token_rules_native = _SURATI_TOKEN_RULES_NATIVE

    # Phrase rewrite: we only match when the window is fully Gujarati-ish and the pattern type
    # agrees with token LID.
    out: list[str] = []
    i = 0
    while i < len(tagged_tokens):
        applied = False
        for pat, rep in phrase_rules:
            n = len(pat)
            if i + n > len(tagged_tokens):
                continue
            win = tagged_tokens[i : i + n]

            pat_is_native = _is_native_pattern(pat)
            if pat_is_native:
                if not all(_native_lang_ok(t) for t in win):
                    continue
                # Native patterns require exact match.
                if tuple(t.text for t in win) != pat:
                    continue
            else:
                if not all(_roman_lang_ok(t) for t in win):
                    continue
                if tuple((t.text or "").lower() for t in win) != pat:
                    continue

            out.extend(rep)
            i += n
            applied = True
            break

        if applied:
            continue

        # Token rewrite: only on Gujarati-ish tokens.
        t = tagged_tokens[i]
        if t.lang == TokenLang.GU_NATIVE:
            out.append(token_rules_native.get(t.text, t.text))
        elif t.lang == TokenLang.GU_ROMAN:
            low = (t.text or "").lower()
            out.append(token_rules.get(low, t.text))
        else:
            out.append(t.text)
        i += 1

    return DialectNormalizationResult(
        dialect=d_eff,
        changed=out != tokens_in,
        tokens_in=tokens_in,
        tokens_out=out,
    )


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

