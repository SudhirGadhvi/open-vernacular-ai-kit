from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import regex as re

from .config import CodeMixConfig
from .normalize import normalize_text
from .rendering import render_tokens
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import translit_gu_roman_to_native_configured, transliteration_backend

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)

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


def lid_stage(tokens: list[str], *, on_event: Optional[EventHook] = None) -> list[Token]:
    tagged = tag_tokens(tokens)
    counts = {k.value: 0 for k in TokenLang}
    for t in tagged:
        counts[t.lang.value] = counts.get(t.lang.value, 0) + 1
    _emit(on_event, {"stage": "lid", "counts": counts})
    return tagged


def transliterate_stage(
    tagged: list[Token], *, config: CodeMixConfig, on_event: Optional[EventHook] = None
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

        j = i
        run: list[Token] = []
        while j < len(tagged) and tagged[j].lang == TokenLang.GU_ROMAN:
            run.append(tagged[j])
            j += 1

        phrase = " ".join(t.text for t in run)
        cands = translit_gu_roman_to_native_configured(
            phrase,
            topk=cfg.topk,
            preserve_case=cfg.preserve_case,
            aggressive_normalize=cfg.aggressive_normalize,
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
                topk=cfg.topk,
                preserve_case=cfg.preserve_case,
                aggressive_normalize=cfg.aggressive_normalize,
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

        norm = normalize_stage(raw, config=cfg, on_event=self.on_event)
        if not norm:
            backend = transliteration_backend()
            _emit(self.on_event, {"stage": "done", "empty_input": True, "backend": backend})
            return CodeMixPipelineResult(
                raw=raw,
                normalized="",
                tokens=[],
                tagged_tokens=[],
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
        tagged = lid_stage(toks, on_event=self.on_event)

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

        rendered, n_transliterated = transliterate_stage(tagged, config=cfg, on_event=self.on_event)
        out = render_stage(rendered, config=cfg, on_event=self.on_event)

        backend = transliteration_backend()
        _emit(
            self.on_event,
            {
                "stage": "done",
                "backend": backend,
                "n_tokens": len(toks),
                "n_gu_roman_tokens": n_gu_roman,
                "n_gu_roman_transliterated": n_transliterated,
            },
        )

        return CodeMixPipelineResult(
            raw=raw,
            normalized=norm,
            tokens=toks,
            tagged_tokens=tagged,
            rendered_tokens=rendered,
            codemix=out,
            n_tokens=len(toks),
            n_en_tokens=n_en,
            n_gu_native_tokens=n_gu_native,
            n_gu_roman_tokens=n_gu_roman,
            n_gu_roman_transliterated=n_transliterated,
            transliteration_backend=backend,
        )

