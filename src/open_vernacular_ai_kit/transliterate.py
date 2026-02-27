from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal, Optional

import regex as re

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)

# Small, high-precision exception dictionary for common Gujlish function words.
# This is intentionally conservative; it should increase useful conversions even when optional
# transliteration backends are not installed.
#
# Keys are compared after `_normalize_gujlish_key()`.
_DEFAULT_GUJLISH_EXCEPTIONS: dict[str, str] = {
    "hu": "હું",
    "tu": "તું",
    "tame": "તમે",
    "ame": "અમે",
    "maru": "મારું",
    "mare": "મારે",
    "taro": "તારો",
    "tari": "તારી",
    "tamaro": "તમારો",
    "tamari": "તમારી",
    "chhe": "છે",
    "che": "છે",
    "nathi": "નથી",
    "hato": "હતો",
    "hati": "હતી",
    "hase": "હશે",
    "shu": "શું",
    "su": "શું",
    "kem": "કેમ",
    "kya": "ક્યાં",
    "kyare": "ક્યારે",
    "aaje": "આજે",
    "kaale": "કાલે",
    "naam": "નામ",
}


def _normalize_gujlish_key(s: str) -> str:
    s = (s or "").strip().lower()
    # Keep ASCII-ish latin only; this makes dict lookups stable.
    s = re.sub(r"[^\p{Latin}]+", "", s, flags=re.VERSION1)
    return s


def _collapse_repeats(s: str, *, max_run: int = 2) -> str:
    """Collapse repeated letters: 'maaaru' -> 'maar u' style, but keep aa/ee/oo intact."""
    if not s:
        return s
    out: list[str] = []
    prev = ""
    run = 0
    for ch in s:
        if ch == prev:
            run += 1
        else:
            prev = ch
            run = 1
        if run <= max_run:
            out.append(ch)
    return "".join(out)


def _gujlish_variants(token: str, *, preserve_case: bool, aggressive_normalize: bool) -> list[str]:
    """
    Generate a small set of spelling variants to increase transliteration hit-rate.

    Keep this bounded (for speed) and deterministic.
    """
    t0 = (token or "").strip()
    if not t0:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = (v or "").strip()
        if not v:
            return
        if v in seen:
            return
        seen.add(v)
        variants.append(v)

    if preserve_case:
        add(t0)
    add(t0.lower())

    if aggressive_normalize:
        base = _collapse_repeats(t0.lower(), max_run=2)
        add(base)
        # Some common Gujlish ambiguity pairs.
        swaps: list[tuple[str, str]] = [
            ("chh", "ch"),
            ("oo", "u"),
            ("oo", "o"),
            ("ee", "i"),
            ("ee", "e"),
            ("aa", "a"),
            ("v", "w"),
            ("w", "v"),
        ]
        # Apply swaps in up to 2 rounds to allow simple chaining (e.g. "chhee" -> "chee" -> "che").
        frontier = [base]
        for _ in range(2):
            next_frontier: list[str] = []
            for cur in frontier:
                for a, b in swaps:
                    if a not in cur:
                        continue
                    v = cur.replace(a, b)
                    if v != cur:
                        add(v)
                        next_frontier.append(v)
            frontier = next_frontier
        # Drop trailing 'h' (e.g. "chhe" vs "che") but keep "sh".
        if base.endswith("h") and not base.endswith("sh") and len(base) >= 3:
            add(base[:-1])

    # Hard cap to avoid pathological blow-ups.
    return variants[:12]


@lru_cache(maxsize=1)
def _get_xlit_engine():
    # Best-effort: if the AI4Bharat Indic-Xlit python package is installed (it may pull heavy deps),
    # use it. We do not depend on it by default because its transitive deps can be brittle.
    try:
        from ai4bharat.transliteration import XlitEngine  # pyright: ignore[reportMissingImports]
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


def transliteration_backend() -> str:
    """
    Return which transliteration backend is selected/available (best-effort).

    This is used for product/demo reporting and should stay cheap and side-effect free.
    """

    return transliteration_backend_configured(preferred="auto")


TranslitBackend = Literal["auto", "ai4bharat", "sanscript", "none"]


def transliteration_backend_configured(*, preferred: TranslitBackend = "auto") -> str:
    """
    Resolve the actual backend used, considering user preference + availability.

    Returns one of: ai4bharat, sanscript, none
    """
    if preferred == "none":
        return "none"
    if preferred == "ai4bharat":
        return "ai4bharat" if _get_xlit_engine() is not None else "none"
    if preferred == "sanscript":
        return "sanscript" if _get_sanscript() is not None else "none"

    # auto
    if _get_xlit_engine() is not None:
        return "ai4bharat"
    if _get_sanscript() is not None:
        return "sanscript"
    return "none"


def transliteration_available() -> bool:
    """True if any transliteration backend is importable."""
    return transliteration_backend() != "none"


def translit_gu_roman_to_native(token: str, *, topk: int = 1) -> Optional[list[str]]:
    """
    Transliterate a romanized Gujarati token into Gujarati script candidates.

    Returns:
      - list[str] of candidates if available
      - None if transliteration engine not installed or fails
    """
    if not token:
        return None

    return translit_gu_roman_to_native_configured(token, topk=topk)


def translit_gu_roman_to_native_configured(
    text: str,
    *,
    topk: int = 1,
    preserve_case: bool = True,
    aggressive_normalize: bool = False,
    exceptions: Optional[dict[str, str]] = None,
    backend: TranslitBackend = "auto",
) -> Optional[list[str]]:
    """
    Transliterate romanized Gujarati into Gujarati script candidates (token or phrase).

    Enhancements over `translit_gu_roman_to_native()`:
    - Gujlish exception dictionary (works even without optional backends installed)
    - Spelling variants (`aggressive_normalize=True`)
    - Best-effort support for multi-word phrases
    """
    if not text:
        return None

    topk = max(1, int(topk))

    exc: dict[str, str] = dict(_DEFAULT_GUJLISH_EXCEPTIONS)
    if exceptions:
        # User dict should win.
        for k, v in exceptions.items():
            if k and v:
                exc[_normalize_gujlish_key(k)] = v

    s = text.strip()
    if not s:
        return None

    # Phrase mode: try exceptions-only join (offline), then backends.
    if re.search(r"\s", s, flags=re.VERSION1):
        parts = [p for p in re.split(r"\s+", s, flags=re.VERSION1) if p]
        if parts:
            mapped: list[str] = []
            for p in parts:
                k = _normalize_gujlish_key(p)
                if k in exc:
                    mapped.append(exc[k])
                else:
                    mapped = []
                    break
            if mapped:
                return [" ".join(mapped)]

        # Backends may or may not support phrases; we try best-effort.
        cands = _translit_backend(s, topk=topk, preserve_case=preserve_case, backend=backend)
        if cands:
            return cands[:topk]
        return None

    # Single-token: exceptions first (cheap + precise).
    for v in _gujlish_variants(s, preserve_case=True, aggressive_normalize=aggressive_normalize):
        k = _normalize_gujlish_key(v)
        if k in exc:
            return [exc[k]]

    # Try a few spelling variants and merge candidates.
    merged: list[str] = []
    seen: set[str] = set()
    for variant in _gujlish_variants(
        s, preserve_case=preserve_case, aggressive_normalize=aggressive_normalize
    ):
        cands = _translit_backend(variant, topk=topk, preserve_case=preserve_case, backend=backend)
        if not cands:
            continue
        for c in cands:
            if c and c not in seen:
                seen.add(c)
                merged.append(c)
        if len(merged) >= topk:
            break

    return merged[:topk] if merged else None


def _extract_candidates(obj: object) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        o = obj.strip()
        return [o] if o else []
    if isinstance(obj, dict):
        for key in ("gu", "guj", "gu-Gujr", "Gujarati"):
            v = obj.get(key)
            if isinstance(v, str):
                v = v.strip()
                return [v] if v else []
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str) and x]
        return []
    if isinstance(obj, list):
        out: list[str] = []
        for it in obj:
            out.extend(_extract_candidates(it))
        return out
    return []


_GUJARATI_VIRAMA = "\u0acd"  # Gujarati sign virama (halant)


def _postprocess_gujarati_candidate(s: str) -> str:
    """
    Normalize common transliterator artifacts.

    Some backends (notably ITRANS-ish fallbacks) may emit a *terminal* virama, e.g. "કામ્".
    Gujarati orthography typically does not write a terminal virama, so we strip it.
    """

    out = (s or "").strip()
    while out.endswith(_GUJARATI_VIRAMA):
        out = out[:-1]
    return out


def _translit_backend(
    text: str, *, topk: int, preserve_case: bool, backend: TranslitBackend
) -> Optional[list[str]]:
    selected = transliteration_backend_configured(preferred=backend)
    if selected == "none":
        return None

    engine = _get_xlit_engine() if selected == "ai4bharat" else None
    if engine is None:
        sanscript = _get_sanscript() if selected == "sanscript" else None
        if sanscript is None:
            return None

        # Fallback: treat input as ITRANS-ish ASCII and convert to Gujarati script.
        # This won't perfectly match informal Gujlish spellings, but it's cheap and offline.
        try:
            out = sanscript.transliterate(text.lower(), sanscript.ITRANS, sanscript.GUJARATI)
            out = _postprocess_gujarati_candidate(out)
            return [out] if out else None
        except Exception:
            return None

    # Indic-Xlit engine: prefer sentence-aware method when whitespace exists.
    try:
        if re.search(r"\s", text, flags=re.VERSION1):
            for meth in ("translit_sentence", "translit_line"):
                fn = getattr(engine, meth, None)
                if fn is None:
                    continue
                try:
                    out = fn(text, topk=topk)
                except TypeError:
                    out = fn(text)
                cands = _extract_candidates(out)
                if cands:
                    cleaned = [_postprocess_gujarati_candidate(c) for c in cands]
                    cleaned = [c for c in cleaned if c]
                    return cleaned[:topk] if cleaned else None

            # Fallback: transliterate each word and join (still improves with variants/exceptions).
            words = [w for w in re.split(r"\s+", text.strip(), flags=re.VERSION1) if w]
            if not words:
                return None
            rendered: list[str] = []
            for w in words:
                out = engine.translit_word(w, topk=1)
                cands = _extract_candidates(out)
                best = _postprocess_gujarati_candidate(cands[0]) if cands else ""
                rendered.append(best if best else w)
            joined = " ".join(rendered).strip()
            return [joined] if joined and _GUJARATI_RE.search(joined) else None

        out = engine.translit_word(text if preserve_case else text.lower(), topk=topk)
        cands = _extract_candidates(out)
        cleaned = [_postprocess_gujarati_candidate(c) for c in cands]
        cleaned = [c for c in cleaned if c]
        return cleaned[:topk] if cleaned else None
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
 
