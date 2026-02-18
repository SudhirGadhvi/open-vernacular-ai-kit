from __future__ import annotations

from typing import Iterable

import regex as re

from .normalize import normalize_text
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import translit_gu_roman_to_native

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


def render_codemix(text: str, *, topk: int = 1) -> str:
    """
    Convert mixed Gujarati/English text into a stable code-mix representation:

    - Gujarati stays in Gujarati script
    - English stays in Latin
    - Romanized Gujarati tokens are transliterated to Gujarati script if possible
    """
    text = normalize_text(text)
    if not text:
        return ""

    toks = tokenize(text)
    tagged = tag_tokens(toks)

    rendered: list[str] = []
    for tok in tagged:
        rendered.append(_render_token(tok, topk=topk))

    return normalize_text(_render_tokens(rendered))


def _render_token(tok: Token, *, topk: int) -> str:
    if tok.lang == TokenLang.GU_NATIVE:
        return tok.text
    if tok.lang == TokenLang.EN:
        return tok.text
    if tok.lang == TokenLang.GU_ROMAN:
        cands = translit_gu_roman_to_native(tok.text, topk=topk)
        return cands[0] if cands else tok.text
    return tok.text
 
