from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional


@lru_cache(maxsize=1)
def _get_xlit_engine():
    # Best-effort: if the AI4Bharat Indic-Xlit python package is installed (it may pull heavy deps),
    # use it. We do not depend on it by default because its transitive deps can be brittle.
    try:
        from ai4bharat.transliteration import XlitEngine
    except Exception:
        return None

    # Gujarati language code for the engine is "gu"
    return XlitEngine("gu", beam_width=10, rescore=True)


@lru_cache(maxsize=1)
def _get_sanscript():
    try:
        from indic_transliteration import sanscript
    except Exception:
        return None
    return sanscript


def translit_gu_roman_to_native(token: str, *, topk: int = 1) -> Optional[list[str]]:
    """
    Transliterate a romanized Gujarati token into Gujarati script candidates.

    Returns:
      - list[str] of candidates if available
      - None if transliteration engine not installed or fails
    """
    if not token:
        return None

    engine = _get_xlit_engine()
    if engine is None:
        sanscript = _get_sanscript()
        if sanscript is None:
            return None

        # Fallback: treat input as ITRANS-ish ASCII and convert to Gujarati script.
        # This won't perfectly match informal Gujlish spellings, but it's cheap and offline.
        try:
            out = sanscript.transliterate(token.lower(), sanscript.ITRANS, sanscript.GUJARATI)
            out = out.strip()
            return [out] if out else None
        except Exception:
            return None

    try:
        out = engine.translit_word(token, topk=max(1, int(topk)))
        cands = out.get("gu") or out.get("guj") or []
        cands = [c for c in cands if isinstance(c, str) and c]
        if not cands:
            return None
        return cands[:topk]
    except Exception:
        return None


def translit_tokens_gu_roman(tokens: Iterable[str], *, topk: int = 1) -> list[Optional[str]]:
    """
    Transliterate tokens (romanized Gujarati) one by one.

    Returns list matching input length; each entry is the best candidate or None.
    """
    out: list[Optional[str]] = []
    for t in tokens:
        cands = translit_gu_roman_to_native(t, topk=topk)
        out.append(cands[0] if cands else None)
    return out
 
