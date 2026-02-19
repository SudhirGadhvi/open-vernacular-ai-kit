from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

NumeralsMode = Literal["keep", "ascii"]
TranslitMode = Literal["token", "sentence"]
TranslitBackend = Literal["auto", "ai4bharat", "sanscript", "none"]
DialectBackend = Literal["auto", "heuristic", "transformers", "llm", "none"]
DialectNormalizerBackend = Literal["auto", "heuristic", "seq2seq", "none"]


@dataclass(frozen=True)
class CodeMixConfig:
    """
    Stable, SDK-first configuration for CodeMix processing.

    The CLI should map flags -> this object; the SDK should accept this object directly.
    """

    # Text normalization
    numerals: NumeralsMode = "keep"
    preserve_numbers: bool = True
    preserve_case: bool = True

    # Transliteration
    topk: int = 1
    aggressive_normalize: bool = False
    translit_mode: TranslitMode = "token"
    translit_backend: TranslitBackend = "auto"
    user_lexicon_path: Optional[str] = None

    # Determinism hook for future stochastic components.
    seed: Optional[int] = None

    # Optional LID signal for Latin tokens (fastText model path for lid.176.ftz).
    fasttext_model_path: Optional[str] = None

    # Dialect utilities (v0.4.x+): detection + optional normalization.
    dialect_backend: DialectBackend = "auto"
    dialect_model_id_or_path: Optional[str] = None
    dialect_min_confidence: float = 0.70
    dialect_normalize: bool = False
    dialect_force: Optional[str] = None
    dialect_normalizer_backend: DialectNormalizerBackend = "auto"
    dialect_normalizer_model_id_or_path: Optional[str] = None

    # Safety: offline-first behavior. When False, any HF model-id usage should be considered
    # an error unless the model is already present on disk.
    allow_remote_models: bool = False

    def numerals_effective(self) -> NumeralsMode:
        """
        Effective numerals behavior after considering `preserve_numbers`.

        For backward compatibility, `preserve_numbers=False` forces ASCII numerals even if
        `numerals="keep"`.
        """

        if not self.preserve_numbers:
            return "ascii"
        return self.numerals

    def normalized(self) -> "CodeMixConfig":
        """Return a defensively normalized config (types/constraints)."""

        topk = max(1, int(self.topk))
        numerals: NumeralsMode = self.numerals
        if numerals not in ("keep", "ascii"):
            raise ValueError("numerals must be one of: keep, ascii")
        if self.translit_mode not in ("token", "sentence"):
            raise ValueError("translit_mode must be one of: token, sentence")
        if self.translit_backend not in ("auto", "ai4bharat", "sanscript", "none"):
            raise ValueError("translit_backend must be one of: auto, ai4bharat, sanscript, none")

        if self.dialect_backend not in ("auto", "heuristic", "transformers", "llm", "none"):
            raise ValueError(
                "dialect_backend must be one of: auto, heuristic, transformers, llm, none"
            )
        if self.dialect_normalizer_backend not in ("auto", "heuristic", "seq2seq", "none"):
            raise ValueError(
                "dialect_normalizer_backend must be one of: auto, heuristic, seq2seq, none"
            )
        if self.dialect_force is not None:
            v = str(self.dialect_force).strip().lower().replace("-", "_").replace(" ", "_")
            allowed = {
                "unknown",
                "standard",
                "kathiawadi",
                "surati",
                "charotar",
                "north_gujarat",
            }
            if v not in allowed:
                raise ValueError(f"dialect_force must be one of: {', '.join(sorted(allowed))}")
        dialect_min_conf = float(self.dialect_min_confidence)
        if not (0.0 <= dialect_min_conf <= 1.0):
            raise ValueError("dialect_min_confidence must be in [0.0, 1.0]")

        seed = None if self.seed is None else int(self.seed)
        return CodeMixConfig(
            numerals=numerals,
            preserve_numbers=bool(self.preserve_numbers),
            preserve_case=bool(self.preserve_case),
            topk=topk,
            aggressive_normalize=bool(self.aggressive_normalize),
            translit_mode=self.translit_mode,
            translit_backend=self.translit_backend,
            user_lexicon_path=None if self.user_lexicon_path is None else str(self.user_lexicon_path),
            seed=seed,
            fasttext_model_path=(
                None if self.fasttext_model_path is None else str(self.fasttext_model_path)
            ),
            dialect_backend=self.dialect_backend,
            dialect_model_id_or_path=(
                None if self.dialect_model_id_or_path is None else str(self.dialect_model_id_or_path)
            ),
            dialect_min_confidence=dialect_min_conf,
            dialect_normalize=bool(self.dialect_normalize),
            dialect_force=(None if self.dialect_force is None else str(self.dialect_force)),
            dialect_normalizer_backend=self.dialect_normalizer_backend,
            dialect_normalizer_model_id_or_path=(
                None
                if self.dialect_normalizer_model_id_or_path is None
                else str(self.dialect_normalizer_model_id_or_path)
            ),
            allow_remote_models=bool(self.allow_remote_models),
        )

    def to_dict(self) -> dict[str, object]:
        # Keep it JSON-friendly (no callables, regex, etc.)
        return dict(asdict(self))

