from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence

import regex as re

from .config import CodeMixConfig
from .dialects import (
    DialectNormalizationResult,
    GujaratiDialect,
    normalize_dialect_tagged_tokens,
)
from .errors import InvalidConfigError, OfflinePolicyError, OptionalDependencyError
from .token_lid import Token, TokenLang, tokenize

DialectNormalizerName = Literal["auto", "heuristic", "seq2seq", "none"]

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]", flags=re.VERSION1)
_LATIN_RE = re.compile(r"[\p{Latin}]", flags=re.VERSION1)


class DialectNormalizerBackend(Protocol):
    name: str

    def normalize(
        self, *, tagged_tokens: Sequence[Token], dialect: GujaratiDialect, config: CodeMixConfig
    ) -> DialectNormalizationResult: ...


@dataclass(frozen=True)
class HeuristicDialectNormalizer:
    name: str = "heuristic"

    def normalize(
        self, *, tagged_tokens: Sequence[Token], dialect: GujaratiDialect, config: CodeMixConfig
    ) -> DialectNormalizationResult:
        res = normalize_dialect_tagged_tokens(tagged_tokens, dialect=dialect)
        return DialectNormalizationResult(
            dialect=res.dialect,
            changed=res.changed,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
            backend=self.name,
        )


def _local_model_path(p: str | None) -> Optional[Path]:
    if not p:
        return None
    try:
        path = Path(p).expanduser()
        if path.exists():
            return path
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class Seq2SeqDialectNormalizer:
    """
    Seq2seq dialect normalization backend.

    Design goals:
    - Never rewrite English/OTHER tokens (we only run seq2seq on Gujarati-ish spans)
    - Be conservative: if output looks invalid (no Gujarati script), keep the input span.
    """

    model_id_or_path: str
    name: str = "seq2seq"

    def normalize(
        self, *, tagged_tokens: Sequence[Token], dialect: GujaratiDialect, config: CodeMixConfig
    ) -> DialectNormalizationResult:
        # Short-circuit: no-op on unknown dialect.
        if dialect in {GujaratiDialect.UNKNOWN, GujaratiDialect.STANDARD}:
            tokens_in = [t.text for t in tagged_tokens]
            return DialectNormalizationResult(
                dialect=dialect,
                changed=False,
                tokens_in=tokens_in,
                tokens_out=tokens_in,
                backend=self.name,
            )

        cfg = config.normalized()

        # Enforce offline-by-default policy.
        local = _local_model_path(self.model_id_or_path)
        if local is None and not cfg.allow_remote_models:
            raise OfflinePolicyError(
                "Remote model downloads are disabled. Provide a local path for "
                "`dialect_normalizer_model_id_or_path` or set allow_remote_models=True."
            )

        try:
            import torch
            import transformers  # type: ignore
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError(
                "Seq2seq dialect normalizer requires extras. Install: `pip install -e '.[dialect-ml]'`"
            ) from e
        _ = transformers  # silence unused import warnings

        tok, model = _load_transformers_seq2seq(self.model_id_or_path)

        tokens_in = [t.text for t in tagged_tokens]
        out: list[str] = []
        i = 0

        def flush_span(span: list[Token]) -> None:
            if not span:
                return
            # Only Gujarati-ish spans are passed in.
            span_text = " ".join(t.text for t in span).strip()
            if not span_text:
                return
            inputs = tok(span_text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=64,
                )
            norm = tok.decode(gen[0], skip_special_tokens=True).strip()

            # Conservative output validation: require *some* Gujarati script in output,
            # otherwise keep original.
            if not _GUJARATI_RE.search(norm):
                out.extend([t.text for t in span])
                return

            # If it outputs mostly Latin, it's likely a failure mode. Keep original.
            if _LATIN_RE.search(norm) and not _GUJARATI_RE.search(norm):
                out.extend([t.text for t in span])
                return

            out.extend(tokenize(norm))

        span: list[Token] = []
        while i < len(tagged_tokens):
            t = tagged_tokens[i]
            # Dialect-to-standard normalization is most reliable on Gujarati script.
            # Romanized Gujarati can be handled via transliteration + rules.
            if t.lang == TokenLang.GU_NATIVE:
                span.append(t)
                i += 1
                continue
            flush_span(span)
            span = []
            out.append(t.text)
            i += 1
        flush_span(span)

        return DialectNormalizationResult(
            dialect=dialect,
            changed=out != tokens_in,
            tokens_in=tokens_in,
            tokens_out=out,
            backend=self.name,
        )


@lru_cache(maxsize=2)
def _load_transformers_seq2seq(model_id_or_path: str) -> tuple[object, object]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
    model.eval()
    return tok, model


@dataclass(frozen=True)
class RulesThenSeq2SeqNormalizer:
    """
    Rules-first + seq2seq fallback.
    """

    seq2seq_model_id_or_path: str
    name: str = "auto"

    def normalize(
        self, *, tagged_tokens: Sequence[Token], dialect: GujaratiDialect, config: CodeMixConfig
    ) -> DialectNormalizationResult:
        rules = HeuristicDialectNormalizer()
        r = rules.normalize(tagged_tokens=tagged_tokens, dialect=dialect, config=config)

        # Re-tag so seq2seq can safely avoid English/OTHER spans.
        from .token_lid import tag_tokens  # local import to keep module import lightweight

        tagged_eff = tag_tokens(r.tokens_out, fasttext_model_path=config.fasttext_model_path)
        seq = Seq2SeqDialectNormalizer(model_id_or_path=self.seq2seq_model_id_or_path)
        s = seq.normalize(tagged_tokens=tagged_eff, dialect=dialect, config=config)

        # If seq2seq was a no-op, keep rule outputs but preserve "changed" from rules.
        if s.tokens_out == r.tokens_out:
            return DialectNormalizationResult(
                dialect=dialect,
                changed=r.changed,
                tokens_in=r.tokens_in,
                tokens_out=r.tokens_out,
                backend=r.backend,
            )

        return DialectNormalizationResult(
            dialect=dialect,
            changed=(s.tokens_out != r.tokens_in),
            tokens_in=r.tokens_in,
            tokens_out=s.tokens_out,
            backend=s.backend,
        )


def get_dialect_normalizer(config: CodeMixConfig) -> Optional[DialectNormalizerBackend]:
    cfg = config.normalized()
    name: DialectNormalizerName = cfg.dialect_normalizer_backend
    if name == "none":
        return None
    if name == "heuristic":
        return HeuristicDialectNormalizer()
    if name == "seq2seq":
        if not cfg.dialect_normalizer_model_id_or_path:
            raise InvalidConfigError(
                "dialect_normalizer_backend='seq2seq' requires dialect_normalizer_model_id_or_path"
            )
        return Seq2SeqDialectNormalizer(model_id_or_path=cfg.dialect_normalizer_model_id_or_path)

    # auto: rules + optional seq2seq if model configured.
    if cfg.dialect_normalizer_model_id_or_path:
        return RulesThenSeq2SeqNormalizer(seq2seq_model_id_or_path=cfg.dialect_normalizer_model_id_or_path)
    return HeuristicDialectNormalizer()

