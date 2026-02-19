from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .codeswitch import CodeSwitchMetrics
from .config import CodeMixConfig
from .dialects import DialectDetection
from .pipeline import CodeMixPipeline, CodeMixPipelineResult, EventHook


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

    # v0.4 additions: code-switching + dialect heuristics (MVP).
    codeswitch: CodeSwitchMetrics
    dialect: DialectDetection

    n_tokens: int
    n_en_tokens: int
    n_gu_native_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int
    pct_gu_roman_transliterated: float

    transliteration_backend: str


def _result_to_analysis(result: CodeMixPipelineResult) -> CodeMixAnalysis:
    pct = (
        (result.n_gu_roman_transliterated / result.n_gu_roman_tokens)
        if result.n_gu_roman_tokens
        else 0.0
    )
    return CodeMixAnalysis(
        raw=result.raw,
        normalized=result.normalized,
        codemix=result.codemix,
        codeswitch=result.codeswitch,
        dialect=result.dialect,
        n_tokens=result.n_tokens,
        n_en_tokens=result.n_en_tokens,
        n_gu_native_tokens=result.n_gu_native_tokens,
        n_gu_roman_tokens=result.n_gu_roman_tokens,
        n_gu_roman_transliterated=result.n_gu_roman_transliterated,
        pct_gu_roman_transliterated=pct,
        transliteration_backend=result.transliteration_backend,
    )


def analyze_codemix_with_config(
    text: str,
    *,
    config: CodeMixConfig,
    on_event: Optional[EventHook] = None,
) -> CodeMixAnalysis:
    """
    Analyze + render CodeMix using a stable `CodeMixConfig`.
    """

    res = CodeMixPipeline(config=config, on_event=on_event).run(text)
    return _result_to_analysis(res)


def render_codemix_with_config(
    text: str,
    *,
    config: CodeMixConfig,
    on_event: Optional[EventHook] = None,
) -> str:
    """Render CodeMix using a stable `CodeMixConfig`."""

    return CodeMixPipeline(config=config, on_event=on_event).run(text).codemix


def analyze_codemix(
    text: str,
    *,
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
    translit_backend: str = "auto",
    user_lexicon_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
) -> CodeMixAnalysis:
    """
    Analyze + render CodeMix in one pass.

    The primary "success metric" is `pct_gu_roman_transliterated`, i.e. the estimated fraction of
    Gujarati-roman (Gujlish) tokens that were converted into Gujarati script.
    """
    cfg = CodeMixConfig(
        topk=topk,
        numerals=numerals,  # type: ignore[arg-type]
        preserve_case=preserve_case,
        preserve_numbers=preserve_numbers,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,  # type: ignore[arg-type]
        translit_backend=translit_backend,  # type: ignore[arg-type]
        user_lexicon_path=user_lexicon_path,
        fasttext_model_path=fasttext_model_path,
    )
    return analyze_codemix_with_config(text, config=cfg)


def render_codemix(
    text: str,
    *,
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
    translit_backend: str = "auto",
    user_lexicon_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
) -> str:
    """
    Convert mixed Gujarati/English text into a stable code-mix representation:

    - Gujarati stays in Gujarati script
    - English stays in Latin
    - Romanized Gujarati tokens are transliterated to Gujarati script if possible
    """
    cfg = CodeMixConfig(
        topk=topk,
        numerals=numerals,  # type: ignore[arg-type]
        preserve_case=preserve_case,
        preserve_numbers=preserve_numbers,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,  # type: ignore[arg-type]
        translit_backend=translit_backend,  # type: ignore[arg-type]
        user_lexicon_path=user_lexicon_path,
        fasttext_model_path=fasttext_model_path,
    )
    return render_codemix_with_config(text, config=cfg)
 
