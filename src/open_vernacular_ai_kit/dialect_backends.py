from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence

from .config import CodeMixConfig
from .dialects import DialectDetection, GujaratiDialect, detect_dialect_from_tagged_tokens
from .errors import InvalidConfigError, OfflinePolicyError, OptionalDependencyError
from .token_lid import Token

DialectBackendName = Literal["auto", "heuristic", "transformers", "llm", "none"]


class DialectBackend(Protocol):
    name: str

    def detect(self, *, text: str, tagged_tokens: Sequence[Token], config: CodeMixConfig) -> DialectDetection: ...


def _confidence_from_scores(scores: dict[str, int]) -> float:
    """
    Convert sparse marker counts into a conservative confidence.

    This intentionally avoids outputting 0.99 on a single hit; it is meant to be used as a
    gating signal (e.g., only normalize if confidence >= threshold).
    """

    vals = sorted((int(v) for v in scores.values()), reverse=True)
    if not vals or vals[0] <= 0:
        return 0.0
    top = float(vals[0])
    second = float(vals[1]) if len(vals) > 1 else 0.0
    if top == second:
        return 0.5
    # 1 vs 0 => 0.5, 3 vs 0 => 0.75, 3 vs 1 => 0.5
    return max(0.0, min(1.0, (top - second) / (top + second + 1.0)))


@dataclass(frozen=True)
class HeuristicDialectBackend:
    name: str = "heuristic"

    def detect(self, *, text: str, tagged_tokens: Sequence[Token], config: CodeMixConfig) -> DialectDetection:
        det = detect_dialect_from_tagged_tokens(tagged_tokens)
        conf = _confidence_from_scores(det.scores)
        return DialectDetection(
            dialect=det.dialect,
            scores=det.scores,
            markers_found=det.markers_found,
            backend=self.name,
            confidence=conf,
        )


@dataclass(frozen=True)
class TransformersDialectBackend:
    """
    Transformers-based dialect classifier.

    This expects a *fine-tuned* sequence classification model. We do not ship weights in the SDK.
    Users can provide:
    - a local directory path (recommended for fully offline use), or
    - a HuggingFace model id (requires network + cache).
    """

    model_id_or_path: str
    name: str = "transformers"

    def detect(self, *, text: str, tagged_tokens: Sequence[Token], config: CodeMixConfig) -> DialectDetection:
        cfg = config.normalized()
        # Enforce offline-by-default policy.
        try:
            p = Path(self.model_id_or_path).expanduser()
            is_local = p.exists()
        except Exception:
            is_local = False
        if not is_local and not cfg.allow_remote_models:
            raise OfflinePolicyError(
                "Remote model downloads are disabled. Provide a local path for "
                "`dialect_model_id_or_path` or set allow_remote_models=True."
            )

        try:
            import torch
            import transformers  # type: ignore
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError(
                "Transformers dialect backend requires extras. Install: `pip install -e '.[dialect-ml]'`"
            ) from e
        _ = transformers  # silence unused import warnings

        tok, model = _load_transformers_seq_cls(self.model_id_or_path)

        inputs = tok(text or "", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0]
            probs = torch.softmax(logits, dim=-1)
            best_idx = int(torch.argmax(probs).item())
            best_p = float(probs[best_idx].item())

        # Map predicted label to our canonical enums.
        id2label = getattr(model.config, "id2label", None) or {}
        raw_label = str(id2label.get(best_idx, best_idx)).strip().lower()
        raw_label = raw_label.replace(" ", "_").replace("-", "_")
        if raw_label in {"kathiawadi", "surati", "standard", "unknown", "charotar", "north_gujarat"}:
            d = GujaratiDialect(raw_label)  # type: ignore[arg-type]
        else:
            # Keep conservative: treat anything else as unknown.
            d = GujaratiDialect.UNKNOWN

        # Preserve the existing `scores: dict[str,int]` shape for compatibility.
        scores = {raw_label: int(round(best_p * 1000))}
        return DialectDetection(
            dialect=d,
            scores=scores,
            markers_found={},
            backend=self.name,
            confidence=best_p,
        )


@lru_cache(maxsize=2)
def _load_transformers_seq_cls(model_id_or_path: str) -> tuple[object, object]:
    """
    Cache HF tokenizer/model per process.
    """

    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    model.eval()
    return tok, model


def get_dialect_backend(config: CodeMixConfig) -> Optional[DialectBackend]:
    cfg = config.normalized()
    name: DialectBackendName = cfg.dialect_backend
    if name == "none":
        return None
    if name == "heuristic":
        return HeuristicDialectBackend()
    if name == "transformers":
        if not cfg.dialect_model_id_or_path:
            raise InvalidConfigError("dialect_backend='transformers' requires dialect_model_id_or_path")
        return TransformersDialectBackend(model_id_or_path=cfg.dialect_model_id_or_path)
    if name == "llm":
        raise NotImplementedError("LLM dialect backend is not implemented yet.")

    # auto: try Transformers if configured, else heuristic.
    if cfg.dialect_model_id_or_path:
        return TransformersDialectBackend(model_id_or_path=cfg.dialect_model_id_or_path)
    return HeuristicDialectBackend()

