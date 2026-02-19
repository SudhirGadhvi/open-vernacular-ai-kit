from __future__ import annotations

from typing import Iterable

import regex as re

# Spacing rules for rendering token streams into a readable sentence.
# Include Gujarati danda (।) since we normalize pipe-danda to it.
_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "।"}
# Currency should attach to the number after it (₹500), but not to the word before it.
_NO_SPACE_AFTER = {"(", "[", "{", "₹"}
_APOSTROPHE = {"'", "’"}
_JOINERS = {"-", "_", "/", "@"}
_EMOJI_RE = re.compile(r"\p{Extended_Pictographic}", flags=re.VERSION1)


def _is_emoji(tok: str) -> bool:
    return bool(_EMOJI_RE.search(tok))


def render_tokens(tokens: Iterable[str]) -> str:
    """
    Render a token stream with stable, language-agnostic spacing/punctuation rules.

    This is intentionally simple and deterministic; it is used by the pipeline and is
    covered by unit tests as part of the v0.2 quality baseline.
    """

    toks = list(tokens)
    out: list[str] = []
    prev: str | None = None
    prev2: str | None = None

    # Track whether we're currently inside a tight span like an email address.
    # We reset this when we explicitly insert a whitespace boundary.
    span_has_at = False

    for t in toks:
        if not out:
            out.append(t)
            prev = t
            prev2 = None
            span_has_at = ("@" in t)
            continue

        # Default is to add a space before the next token.
        add_space = True

        if t in _NO_SPACE_BEFORE:
            add_space = False
        elif prev in _NO_SPACE_AFTER:
            add_space = False
        elif t in _APOSTROPHE or prev in _APOSTROPHE:
            add_space = False
        elif t in _JOINERS or prev in _JOINERS:
            add_space = False
        elif prev == "." and span_has_at:
            # Email-ish / handle-ish span: keep domain tokens tight (test@abc.com).
            add_space = False
        elif prev == "." and (prev2 or "").isdigit() and t.isdigit():
            # Decimal numbers: keep tight (3.14).
            add_space = False
        elif _is_emoji(t) or _is_emoji(prev or ""):
            add_space = True

        if add_space:
            out.append(" " + t)
            span_has_at = ("@" in t)
        else:
            out.append(t)
            span_has_at = span_has_at or ("@" in t)

        prev2 = prev
        prev = t

    return "".join(out)

