from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

EmbedTextsFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class RagDocument:
    """
    One document/snippet for retrieval.
    """

    doc_id: str
    text: str
    meta: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class RagQuery:
    """
    One query with known relevant documents (for eval).
    """

    query: str
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class RagSearchResult:
    doc_id: str
    score: float
    text: str
    meta: dict[str, Any]


def _require_torch_and_transformers(context: str) -> tuple[Any, Any]:
    """
    Import torch + transformers lazily.

    The core SDK stays lightweight; embeddings-based RAG is optional.
    """

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"{context} requires torch. Install with: pip install -e '.[rag-embeddings]'"
        ) from e

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"{context} requires transformers. Install with: pip install -e '.[rag-embeddings]'"
        ) from e

    return torch, (AutoModel, AutoTokenizer)


def _is_local_path(model_id_or_path: str) -> bool:
    try:
        p = Path(model_id_or_path).expanduser()
        return p.exists()
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load_hf_encoder(model_id_or_path: str) -> tuple[Any, Any]:
    _, (AutoModel, AutoTokenizer) = _require_torch_and_transformers("HF embedder")
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModel.from_pretrained(model_id_or_path)
    model.eval()
    return tok, model


def make_hf_embedder(
    *,
    model_id_or_path: str,
    allow_remote_models: bool = False,
    batch_size: int = 16,
    max_length: int = 192,
) -> EmbedTextsFn:
    """
    Create an `embed_texts(texts) -> embeddings` function using a HF encoder model.

    This uses mean pooling over `last_hidden_state` with `attention_mask`, and returns L2-normalized
    embeddings as python lists (so callers do not need numpy).

    Offline-first policy:
      - If `model_id_or_path` is not a local path and `allow_remote_models=False`, this raises.
    """

    mid = str(model_id_or_path or "").strip()
    if not mid:
        raise ValueError("model_id_or_path is required")
    if not _is_local_path(mid) and not bool(allow_remote_models):
        raise RuntimeError(
            "Remote model downloads are disabled. Provide a local path for `model_id_or_path` "
            "or set allow_remote_models=True."
        )

    bs = max(1, int(batch_size))
    ml = max(16, int(max_length))

    def embed_texts(texts: Sequence[str]) -> list[list[float]]:
        torch, _ = _require_torch_and_transformers("HF embedder")
        tok, model = _load_hf_encoder(mid)
        out: list[list[float]] = []
        if not texts:
            return out

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = [t if t is not None else "" for t in texts[i : i + bs]]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=ml,
                    return_tensors="pt",
                )
                res = model(**enc)
                last_hidden = res.last_hidden_state  # (b, t, d)
                mask = enc["attention_mask"].unsqueeze(-1)  # (b, t, 1)
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / denom
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.extend(emb.cpu().tolist())

        # Ensure it is JSON-friendly floats (not numpy scalars, etc).
        return [[float(x) for x in row] for row in out]

    return embed_texts


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return float(sum(float(a[i]) * float(b[i]) for i in range(n)))


def _l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(float(sum(float(x) * float(x) for x in v)))


def _l2_normalize(v: Sequence[float]) -> list[float]:
    denom = _l2_norm(v)
    if denom <= 0.0:
        return [0.0 for _ in v]
    return [float(x) / denom for x in v]


@dataclass(frozen=True)
class RagIndex:
    """
    A tiny embeddings index suitable for small curated corpora.

    Design goals:
      - no mandatory dependencies (pure Python)
      - deterministic behavior
      - serializable (JSON) for easy caching/shipping
    """

    docs: list[RagDocument]
    doc_embeddings: list[list[float]]  # L2-normalized
    embedding_model: Optional[str] = None

    @classmethod
    def build(
        cls,
        *,
        docs: Sequence[RagDocument],
        embed_texts: EmbedTextsFn,
        embedding_model: Optional[str] = None,
    ) -> "RagIndex":
        texts = [d.text for d in docs]
        emb = embed_texts(texts)
        if len(emb) != len(texts):
            raise ValueError("embed_texts returned embeddings with different length than docs")
        emb_norm = [_l2_normalize(v) for v in emb]
        return cls(docs=list(docs), doc_embeddings=emb_norm, embedding_model=embedding_model)

    def search(
        self,
        *,
        query: str,
        embed_texts: EmbedTextsFn,
        topk: int = 5,
    ) -> list[RagSearchResult]:
        k = max(1, int(topk))
        q_embs = embed_texts([query])
        if not q_embs:
            return []
        q = _l2_normalize(q_embs[0])

        scored: list[tuple[float, RagDocument]] = []
        for i, d in enumerate(self.docs):
            dv = self.doc_embeddings[i] if i < len(self.doc_embeddings) else []
            scored.append((_dot(q, dv), d))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[RagSearchResult] = []
        for score, d in scored[:k]:
            out.append(
                RagSearchResult(
                    doc_id=d.doc_id,
                    score=float(score),
                    text=d.text,
                    meta=dict(d.meta or {}),
                )
            )
        return out

    def recall_at_k(self, *, queries: Sequence[RagQuery], embed_texts: EmbedTextsFn, k: int = 5) -> float:
        if not queries:
            return 0.0
        kk = max(1, int(k))
        hits = 0
        for q in queries:
            ranked = self.search(query=q.query, embed_texts=embed_texts, topk=kk)
            ranked_ids = [r.doc_id for r in ranked]
            if any(d in ranked_ids for d in (q.relevant_doc_ids or [])):
                hits += 1
        return hits / len(queries)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "docs": [
                {"doc_id": d.doc_id, "text": d.text, "meta": (d.meta or {})} for d in (self.docs or [])
            ],
            "doc_embeddings": self.doc_embeddings,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "RagIndex":
        docs_raw = data.get("docs") if isinstance(data.get("docs"), list) else []
        docs: list[RagDocument] = []
        for rec in docs_raw:
            if not isinstance(rec, dict):
                continue
            docs.append(
                RagDocument(
                    doc_id=str(rec.get("doc_id", "") or ""),
                    text=str(rec.get("text", "") or ""),
                    meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else {}),
                )
            )
        emb_raw = data.get("doc_embeddings") if isinstance(data.get("doc_embeddings"), list) else []
        emb: list[list[float]] = []
        for row in emb_raw:
            if not isinstance(row, list):
                continue
            emb.append([float(x) for x in row])
        return cls(
            docs=docs,
            doc_embeddings=emb,
            embedding_model=(str(data.get("embedding_model")) if data.get("embedding_model") else None),
        )

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "RagIndex":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid index JSON (expected object at top-level)")
        return cls.from_json_dict(data)

