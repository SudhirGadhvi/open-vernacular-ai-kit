from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import regex as re

from .normalize import normalize_text
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import translit_gu_roman_to_native_configured, transliteration_backend

_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "₹"}
_NO_SPACE_AFTER = {"(", "[", "{", "₹"}
_APOSTROPHE = {"'", "’"}
_JOINERS = {"-", "_", "/", "@"}
_EMOJI_RE = re.compile(r"\p{Extended_Pictographic}", flags=re.VERSION1)
_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)


def _is_emoji(tok: str) -> bool:
    return bool(_EMOJI_RE.search(tok))


def _render_tokens(tokens: Iterable[str]) -> str:
    out: list[str] = []
    prev: str | None = None

    for t in tokens:
        if not out:
            out.append(t)
            prev = t
            continue

        if t in _NO_SPACE_BEFORE:
            out.append(t)
        elif prev in _NO_SPACE_AFTER:
            out.append(t)
        elif t in _APOSTROPHE or prev in _APOSTROPHE:
            out.append(t)
        elif t in _JOINERS or prev in _JOINERS:
            out.append(t)
        elif _is_emoji(t) or _is_emoji(prev or ""):
            out.append(" " + t)
        else:
            out.append(" " + t)

        prev = t

    return "".join(out)


@dataclass(frozen=True)
class CodeMixAnalysis:
    """
    Product-facing analysis of CodeMix rendering.

    Canonical output format for this kit is the `codemix` string:
    Gujarati stays Gujarati script; English stays Latin; Gujlish tokens are transliterated when possible.
    """

    raw: str
    normalized: str
    codemix: str

    n_tokens: int
    n_en_tokens: int
    n_gu_native_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int
    pct_gu_roman_transliterated: float

    transliteration_backend: str


def analyze_codemix(
    text: str,
    *,
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
) -> CodeMixAnalysis:
    """
    Analyze + render CodeMix in one pass.

    The primary "success metric" is `pct_gu_roman_transliterated`, i.e. the estimated fraction of
    Gujarati-roman (Gujlish) tokens that were converted into Gujarati script.
    """
    raw = text or ""
    numerals_eff = numerals
    if not preserve_numbers:
        numerals_eff = "ascii"

    norm = normalize_text(raw, numerals=numerals_eff)
    if not preserve_case:
        # Keep this simple: lowercasing does not affect Gujarati script, but it makes Latin tokens
        # (including Gujlish) more consistent for LID/transliteration.
        norm = norm.lower()
    if not norm:
        return CodeMixAnalysis(
            raw=raw,
            normalized="",
            codemix="",
            n_tokens=0,
            n_en_tokens=0,
            n_gu_native_tokens=0,
            n_gu_roman_tokens=0,
            n_gu_roman_transliterated=0,
            pct_gu_roman_transliterated=0.0,
            transliteration_backend=transliteration_backend(),
        )

    toks = tokenize(norm)
    tagged = tag_tokens(toks)

    rendered: list[str] = []
    n_transliterated = 0
    n_en = 0
    n_gu_native = 0
    n_gu_roman = 0
    for tok in tagged:
        if tok.lang == TokenLang.EN:
            n_en += 1
        elif tok.lang == TokenLang.GU_NATIVE:
            n_gu_native += 1
        elif tok.lang == TokenLang.GU_ROMAN:
            n_gu_roman += 1

    rendered, n_transliterated = _render_tagged_tokens(
        tagged,
        topk=topk,
        preserve_case=preserve_case,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,
    )

    out = normalize_text(_render_tokens(rendered), numerals=numerals_eff)
    pct = (n_transliterated / n_gu_roman) if n_gu_roman else 0.0
    return CodeMixAnalysis(
        raw=raw,
        normalized=norm,
        codemix=out,
        n_tokens=len(toks),
        n_en_tokens=n_en,
        n_gu_native_tokens=n_gu_native,
        n_gu_roman_tokens=n_gu_roman,
        n_gu_roman_transliterated=n_transliterated,
        pct_gu_roman_transliterated=pct,
        transliteration_backend=transliteration_backend(),
    )


def render_codemix(
    text: str,
    *,
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
) -> str:
    """
    Convert mixed Gujarati/English text into a stable code-mix representation:

    - Gujarati stays in Gujarati script
    - English stays in Latin
    - Romanized Gujarati tokens are transliterated to Gujarati script if possible
    """
    numerals_eff = numerals
    if not preserve_numbers:
        numerals_eff = "ascii"

    text = normalize_text(text, numerals=numerals_eff)
    if not preserve_case:
        text = text.lower()
    if not text:
        return ""

    toks = tokenize(text)
    tagged = tag_tokens(toks)
    rendered, _ = _render_tagged_tokens(
        tagged,
        topk=topk,
        preserve_case=preserve_case,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,
    )
    return normalize_text(_render_tokens(rendered), numerals=numerals_eff)


def _render_token(tok: Token, *, topk: int) -> str:
    return _render_token_with_info(tok, topk=topk)[0]


def _render_token_with_info(tok: Token, *, topk: int) -> tuple[str, bool]:
    """
    Render one token; returns (rendered_text, did_transliterate).

    `did_transliterate` is only meaningful for GU_ROMAN tokens.
    """
    if tok.lang in {TokenLang.GU_NATIVE, TokenLang.EN}:
        return tok.text, False
    if tok.lang == TokenLang.GU_ROMAN:
        cands = translit_gu_roman_to_native_configured(
            tok.text,
            topk=topk,
            preserve_case=True,
            aggressive_normalize=False,
        )
        if not cands:
            return tok.text, False
        best = cands[0]
        return best, best != tok.text
    return tok.text, False


def _render_tagged_tokens(
    tagged: list[Token],
    *,
    topk: int,
    preserve_case: bool,
    aggressive_normalize: bool,
    translit_mode: str,
) -> tuple[list[str], int]:
    """
    Render a token stream; returns (rendered_tokens, n_gu_roman_tokens_transliterated).

    `translit_mode`:
      - "token": transliterate GU_ROMAN tokens one by one
      - "sentence": join contiguous GU_ROMAN runs into a phrase and attempt phrase-level translit
    """
    if translit_mode not in {"token", "sentence"}:
        raise ValueError("translit_mode must be one of: token, sentence")

    rendered: list[str] = []
    n_transliterated = 0

    if translit_mode == "token":
        for tok in tagged:
            if tok.lang != TokenLang.GU_ROMAN:
                rendered.append(tok.text if preserve_case else tok.text.lower())
                continue

            cands = translit_gu_roman_to_native_configured(
                tok.text,
                topk=topk,
                preserve_case=preserve_case,
                aggressive_normalize=aggressive_normalize,
            )
            if not cands:
                rendered.append(tok.text if preserve_case else tok.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != tok.text and _GUJARATI_RE.search(best):
                n_transliterated += 1

        return rendered, n_transliterated

    # sentence mode
    i = 0
    while i < len(tagged):
        tok = tagged[i]
        if tok.lang != TokenLang.GU_ROMAN:
            rendered.append(tok.text if preserve_case else tok.text.lower())
            i += 1
            continue

        j = i
        run: list[Token] = []
        while j < len(tagged) and tagged[j].lang == TokenLang.GU_ROMAN:
            run.append(tagged[j])
            j += 1

        phrase = " ".join(t.text for t in run)
        cands = translit_gu_roman_to_native_configured(
            phrase,
            topk=topk,
            preserve_case=preserve_case,
            aggressive_normalize=aggressive_normalize,
        )
        if cands:
            best = cands[0]
            if _GUJARATI_RE.search(best):
                # Tokenize the Gujarati output so spacing/punct render stays consistent.
                rendered.extend(tokenize(best))
                n_transliterated += len(run)
                i = j
                continue

        # Fallback: token-by-token.
        for t in run:
            cands = translit_gu_roman_to_native_configured(
                t.text,
                topk=topk,
                preserve_case=preserve_case,
                aggressive_normalize=aggressive_normalize,
            )
            if not cands:
                rendered.append(t.text if preserve_case else t.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != t.text and _GUJARATI_RE.search(best):
                n_transliterated += 1

        i = j

    return rendered, n_transliterated
 
