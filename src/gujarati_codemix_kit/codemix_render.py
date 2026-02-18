from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import regex as re

from .normalize import normalize_text
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import translit_gu_roman_to_native, transliteration_backend

_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "₹"}
_NO_SPACE_AFTER = {"(", "[", "{", "₹"}
_APOSTROPHE = {"'", "’"}
_JOINERS = {"-", "_", "/", "@"}
_EMOJI_RE = re.compile(r"\p{Extended_Pictographic}", flags=re.VERSION1)


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


def analyze_codemix(text: str, *, topk: int = 1, numerals: str = "keep") -> CodeMixAnalysis:
    """
    Analyze + render CodeMix in one pass.

    The primary "success metric" is `pct_gu_roman_transliterated`, i.e. the estimated fraction of
    Gujarati-roman (Gujlish) tokens that were converted into Gujarati script.
    """
    raw = text or ""
    norm = normalize_text(raw, numerals=numerals)
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

        rendered_tok, did = _render_token_with_info(tok, topk=topk)
        if did:
            n_transliterated += 1
        rendered.append(rendered_tok)

    out = normalize_text(_render_tokens(rendered), numerals=numerals)
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


def render_codemix(text: str, *, topk: int = 1, numerals: str = "keep") -> str:
    """
    Convert mixed Gujarati/English text into a stable code-mix representation:

    - Gujarati stays in Gujarati script
    - English stays in Latin
    - Romanized Gujarati tokens are transliterated to Gujarati script if possible
    """
    text = normalize_text(text, numerals=numerals)
    if not text:
        return ""

    toks = tokenize(text)
    tagged = tag_tokens(toks)

    rendered: list[str] = []
    for tok in tagged:
        rendered.append(_render_token(tok, topk=topk))

    return normalize_text(_render_tokens(rendered), numerals=numerals)


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
        cands = translit_gu_roman_to_native(tok.text, topk=topk)
        if not cands:
            return tok.text, False
        best = cands[0]
        return best, best != tok.text
    return tok.text, False
 
