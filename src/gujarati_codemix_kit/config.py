from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

NumeralsMode = Literal["keep", "ascii"]
TranslitMode = Literal["token", "sentence"]


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

    # Determinism hook for future stochastic components.
    seed: Optional[int] = None

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

        seed = None if self.seed is None else int(self.seed)
        return CodeMixConfig(
            numerals=numerals,
            preserve_numbers=bool(self.preserve_numbers),
            preserve_case=bool(self.preserve_case),
            topk=topk,
            aggressive_normalize=bool(self.aggressive_normalize),
            translit_mode=self.translit_mode,
            seed=seed,
        )

    def to_dict(self) -> dict[str, object]:
        # Keep it JSON-friendly (no callables, regex, etc.)
        return dict(asdict(self))

