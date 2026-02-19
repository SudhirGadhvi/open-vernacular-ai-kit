from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

import regex as re

from .codeswitch import CodeSwitchMetrics, compute_code_switch_metrics
from .config import CodeMixConfig
from .dialects import DialectDetection, detect_dialect_from_tagged_tokens
from .lexicon import LexiconLoadResult, load_user_lexicon
from .normalize import normalize_text
from .rendering import render_tokens
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import (
    translit_gu_roman_to_native_configured,
    transliteration_backend_configured,
)

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)
_JOINERS = {"-", "_", "/", "@"}

EventHook = Callable[[dict[str, Any]], None]


def _emit(hook: Optional[EventHook], event: dict[str, Any]) -> None:
    if hook is None:
        return
    try:
        hook(event)
    except Exception:
        # Logging must never break text processing.
        return


@dataclass(frozen=True)
class CodeMixPipelineResult:
    raw: str
    normalized: str
    tokens: list[str]
    tagged_tokens: list[Token]
    codeswitch: CodeSwitchMetrics
    dialect: DialectDetection
    rendered_tokens: list[str]
    codemix: str

    n_tokens: int
    n_en_tokens: int
    n_gu_native_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int

    transliteration_backend: str


def normalize_stage(text: str, *, config: CodeMixConfig, on_event: Optional[EventHook] = None) -> str:
    cfg = config.normalized()
    numerals_eff = cfg.numerals_effective()

    norm = normalize_text(text or "", numerals=numerals_eff)
    if not cfg.preserve_case:
        norm = norm.lower()

    _emit(
        on_event,
        {
            "stage": "normalize",
            "numerals": numerals_eff,
            "preserve_case": cfg.preserve_case,
            "raw_len": len(text or ""),
            "normalized_len": len(norm),
        },
    )
    return norm


def tokenize_stage(text: str, *, on_event: Optional[EventHook] = None) -> list[str]:
    toks = tokenize(text or "")
    _emit(
        on_event,
        {
            "stage": "tokenize",
            "n_tokens": len(toks),
            "tokens_preview": toks[:12],
        },
    )
    return toks


def lid_stage(
    tokens: list[str],
    *,
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
    user_lexicon_source: str = "none",
    on_event: Optional[EventHook] = None,
) -> list[Token]:
    tagged = tag_tokens(tokens, lexicon_keys=lexicon_keys, fasttext_model_path=fasttext_model_path)
    counts = {k.value: 0 for k in TokenLang}
    for t in tagged:
        counts[t.lang.value] = counts.get(t.lang.value, 0) + 1
    _emit(
        on_event,
        {
            "stage": "lid",
            "counts": counts,
            "preview": [
                {"text": t.text, "lang": t.lang.value, "confidence": t.confidence, "reason": t.reason}
                for t in tagged[:12]
            ],
            "user_lexicon": user_lexicon_source,
        },
    )
    return tagged


@lru_cache(maxsize=8)
def _load_user_lexicon_cached(path: str) -> LexiconLoadResult:
    # Cache by path string; this is a convenience for repeated pipeline runs.
    return load_user_lexicon(path)


def transliterate_stage(
    tagged: list[Token],
    *,
    config: CodeMixConfig,
    lexicon: Optional[dict[str, str]] = None,
    on_event: Optional[EventHook] = None,
) -> tuple[list[str], int]:
    cfg = config.normalized()

    if cfg.translit_mode not in ("token", "sentence"):
        raise ValueError("translit_mode must be one of: token, sentence")

    rendered: list[str] = []
    n_transliterated = 0

    if cfg.translit_mode == "token":
        for tok in tagged:
            if tok.lang != TokenLang.GU_ROMAN:
                rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
                continue

            cands = translit_gu_roman_to_native_configured(
                tok.text,
                topk=cfg.topk,
                preserve_case=cfg.preserve_case,
                aggressive_normalize=cfg.aggressive_normalize,
                exceptions=lexicon,
                backend=cfg.translit_backend,
            )
            if not cands:
                rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != tok.text and _GUJARATI_RE.search(best):
                n_transliterated += 1

        _emit(
            on_event,
            {
                "stage": "transliterate",
                "mode": "token",
                "n_gu_roman_transliterated": n_transliterated,
            },
        )
        return rendered, n_transliterated

    # sentence mode
    i = 0
    while i < len(tagged):
        tok = tagged[i]
        if tok.lang != TokenLang.GU_ROMAN:
            rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
            i += 1
            continue

        # Include common joiners (e.g. "hu-maru-naam") as part of a roman span so that
        # phrase-level transliteration can improve backend hit-rate.
        j = i
        span: list[Token] = []
        while j < len(tagged):
            cur = tagged[j]
            if cur.lang == TokenLang.GU_ROMAN:
                span.append(cur)
                j += 1
                continue
            if (
                cur.text in _JOINERS
                and j + 1 < len(tagged)
                and tagged[j + 1].lang == TokenLang.GU_ROMAN
            ):
                span.append(cur)  # keep joiner
                span.append(tagged[j + 1])
                j += 2
                continue
            break

        roman_words = [t for t in span if t.lang == TokenLang.GU_ROMAN]
        joiners = [t.text for t in span if t.text in _JOINERS and t.lang != TokenLang.GU_ROMAN]
        phrase = " ".join(t.text for t in roman_words)
        cands = translit_gu_roman_to_native_configured(
            phrase,
            topk=cfg.topk,
            preserve_case=cfg.preserve_case,
            aggressive_normalize=cfg.aggressive_normalize,
            exceptions=lexicon,
            backend=cfg.translit_backend,
        )
        if cands:
            best = cands[0]
            if _GUJARATI_RE.search(best):
                out_toks = tokenize(best)
                if joiners:
                    # Preserve joiners only when we can align word-to-word.
                    if len(out_toks) == len(roman_words):
                        for idx, w in enumerate(out_toks):
                            rendered.append(w)
                            if idx < len(joiners):
                                rendered.append(joiners[idx])
                        n_transliterated += len(roman_words)
                        i = j
                        continue
                else:
                    # Tokenize the Gujarati output so spacing/punct render stays consistent.
                    rendered.extend(out_toks)
                    n_transliterated += len(roman_words)
                    i = j
                    continue

        # Fallback: token-by-token.
        for t in span:
            if t.lang != TokenLang.GU_ROMAN:
                rendered.append(t.text)
                continue
            cands = translit_gu_roman_to_native_configured(
                t.text,
                topk=cfg.topk,
                preserve_case=cfg.preserve_case,
                aggressive_normalize=cfg.aggressive_normalize,
                exceptions=lexicon,
                backend=cfg.translit_backend,
            )
            if not cands:
                rendered.append(t.text if cfg.preserve_case else t.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != t.text and _GUJARATI_RE.search(best):
                n_transliterated += 1

        i = j

    _emit(
        on_event,
        {
            "stage": "transliterate",
            "mode": "sentence",
            "n_gu_roman_transliterated": n_transliterated,
        },
    )
    return rendered, n_transliterated


def render_stage(
    rendered_tokens: list[str], *, config: CodeMixConfig, on_event: Optional[EventHook] = None
) -> str:
    cfg = config.normalized()
    numerals_eff = cfg.numerals_effective()

    out = normalize_text(render_tokens(rendered_tokens), numerals=numerals_eff)
    _emit(on_event, {"stage": "render", "output_len": len(out)})
    return out


class CodeMixPipeline:
    """
    First-class pipeline API:
      normalize -> tokenize -> LID -> transliterate -> render
    """

    def __init__(self, *, config: Optional[CodeMixConfig] = None, on_event: Optional[EventHook] = None):
        self.config = (config or CodeMixConfig()).normalized()
        self.on_event = on_event

    def run(self, text: str) -> CodeMixPipelineResult:
        raw = text or ""
        cfg = self.config
        lex_res = (
            _load_user_lexicon_cached(cfg.user_lexicon_path)  # type: ignore[arg-type]
            if cfg.user_lexicon_path
            else LexiconLoadResult(mappings={}, source="none")
        )
        lex = lex_res.mappings
        lex_keys = set(lex.keys()) if lex else None

        norm = normalize_stage(raw, config=cfg, on_event=self.on_event)
        if not norm:
            backend = transliteration_backend_configured(preferred=cfg.translit_backend)
            cs = compute_code_switch_metrics([])
            d = detect_dialect_from_tagged_tokens([])
            _emit(
                self.on_event,
                {
                    "stage": "done",
                    "empty_input": True,
                    "backend": backend,
                    "user_lexicon": lex_res.source,
                    "cmi": cs.cmi,
                    "switch_points": cs.n_switch_points,
                    "dialect": d.dialect.value,
                },
            )
            return CodeMixPipelineResult(
                raw=raw,
                normalized="",
                tokens=[],
                tagged_tokens=[],
                codeswitch=cs,
                dialect=d,
                rendered_tokens=[],
                codemix="",
                n_tokens=0,
                n_en_tokens=0,
                n_gu_native_tokens=0,
                n_gu_roman_tokens=0,
                n_gu_roman_transliterated=0,
                transliteration_backend=backend,
            )

        toks = tokenize_stage(norm, on_event=self.on_event)
        tagged = lid_stage(
            toks,
            lexicon_keys=lex_keys,
            fasttext_model_path=cfg.fasttext_model_path,
            user_lexicon_source=lex_res.source,
            on_event=self.on_event,
        )
        cs = compute_code_switch_metrics(tagged)
        d = detect_dialect_from_tagged_tokens(tagged)

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

        rendered, n_transliterated = transliterate_stage(
            tagged, config=cfg, lexicon=lex, on_event=self.on_event
        )
        out = render_stage(rendered, config=cfg, on_event=self.on_event)

        backend = transliteration_backend_configured(preferred=cfg.translit_backend)
        _emit(
            self.on_event,
            {
                "stage": "done",
                "backend": backend,
                "n_tokens": len(toks),
                "n_gu_roman_tokens": n_gu_roman,
                "n_gu_roman_transliterated": n_transliterated,
                "user_lexicon": lex_res.source,
                "cmi": cs.cmi,
                "switch_points": cs.n_switch_points,
                "dialect": d.dialect.value,
            },
        )

        return CodeMixPipelineResult(
            raw=raw,
            normalized=norm,
            tokens=toks,
            tagged_tokens=tagged,
            codeswitch=cs,
            dialect=d,
            rendered_tokens=rendered,
            codemix=out,
            n_tokens=len(toks),
            n_en_tokens=n_en,
            n_gu_native_tokens=n_gu_native,
            n_gu_roman_tokens=n_gu_roman,
            n_gu_roman_transliterated=n_transliterated,
            transliteration_backend=backend,
        )

