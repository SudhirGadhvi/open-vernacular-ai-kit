from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import regex as re

from .codemix_render import analyze_codemix, render_codemix
from .dialect_datasets import (
    load_dialect_id_jsonl,
    load_dialect_normalization_jsonl,
    packaged_data_path,
)
from .normalize import normalize_text
from .rag_datasets import load_gujarat_facts_tiny
from .rendering import render_tokens
from .token_lid import tokenize
from .transliterate import transliteration_backend
from .errors import DownloadError, InvalidConfigError, OptionalDependencyError

_GUJARATI_RE = re.compile(r"[\p{Gujarati}]")

_DEFAULT_EMBEDDING_MODEL = "ai4bharat/indic-bert"
# Fallback used when IndicBERT is gated on Hugging Face (no auth token).
_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    text_column: str


@dataclass(frozen=True)
class GoldenTranslitCase:
    """One hand-validated Gujlish -> Gujarati expected output case."""

    gujlish: str
    expected_any_of: list[str]
    requires_backend: bool = False


_GUJLISH_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="in22",
        url="https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20IN22.csv",
        text_column="guj",
    ),
    DatasetSpec(
        name="xnli",
        url="https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20XNLI.csv",
        text_column="guj",
    ),
]


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "gujarati-codemix-kit"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "requests is required for eval harness; install with: pip install -e '.[eval]'"
        ) from e

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:  # pragma: no cover
        raise DownloadError(f"Failed to download: {url}") from e


def _iter_texts_from_csv(path: Path, *, text_column: str) -> Iterable[str]:
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "pandas is required for eval harness; install with: pip install -e '.[eval]'"
        ) from e

    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise InvalidConfigError(f"Missing expected column '{text_column}' in {path.name}")
    for v in df[text_column].astype(str).tolist():
        yield v


def _has_gujarati(text: str) -> bool:
    return bool(_GUJARATI_RE.search(text))


def _analyze_one(text: str, *, topk: int = 1) -> dict[str, Any]:
    raw = text
    a = analyze_codemix(raw, topk=topk)
    norm = a.normalized
    out = a.codemix

    return {
        "raw": raw,
        "normalized": norm,
        "codemix": out,
        "has_gujarati_raw": _has_gujarati(raw),
        "has_gujarati_codemix": _has_gujarati(out),
        "n_tokens": a.n_tokens,
        "n_gu_roman_tokens": a.n_gu_roman_tokens,
        "n_gu_roman_tokens_changed_est": a.n_gu_roman_transliterated,
        "transliteration_backend": a.transliteration_backend,
    }


def _sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="replace")).hexdigest()


def _accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    ok = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return ok / len(y_true)


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str], *, labels: Sequence[str]) -> float:
    """
    Simple macro-F1 without external deps.
    """

    if not labels:
        return 0.0
    label_set = list(dict.fromkeys(labels))
    f1s: list[float] = []
    for lab in label_set:
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == lab and y_pred[i] == lab)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] != lab and y_pred[i] == lab)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == lab and y_pred[i] != lab)
        prec = (tp / (tp + fp)) if (tp + fp) else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _chrf(hyp: str, ref: str, *, n: int = 6, beta: float = 2.0) -> float:
    """
    Tiny chrF-style character n-gram F-score (0..100).

    This is not a full sacrebleu implementation; it's sufficient for regression tracking.
    """

    h = (hyp or "").strip()
    r = (ref or "").strip()
    if not h and not r:
        return 100.0
    if not h or not r:
        return 0.0

    def ngrams(s: str, k: int) -> dict[str, int]:
        out: dict[str, int] = {}
        if len(s) < k:
            return out
        for i in range(len(s) - k + 1):
            g = s[i : i + k]
            out[g] = out.get(g, 0) + 1
        return out

    scores: list[float] = []
    for k in range(1, n + 1):
        hn = ngrams(h, k)
        rn = ngrams(r, k)
        if not hn or not rn:
            scores.append(0.0)
            continue
        overlap = 0
        for g, c in hn.items():
            overlap += min(c, rn.get(g, 0))
        prec = overlap / sum(hn.values()) if hn else 0.0
        rec = overlap / sum(rn.values()) if rn else 0.0
        if prec == 0.0 and rec == 0.0:
            scores.append(0.0)
            continue
        beta2 = beta * beta
        f = (1 + beta2) * prec * rec / (beta2 * prec + rec) if (beta2 * prec + rec) else 0.0
        scores.append(f)

    return 100.0 * (sum(scores) / len(scores))


def run_dialect_id_eval(
    *,
    dataset_path: Optional[str] = None,
    dialect_backend: str = "heuristic",
    dialect_model_id_or_path: Optional[str] = None,
    allow_remote_models: bool = False,
    max_rows: Optional[int] = 2000,
) -> dict[str, Any]:
    """
    Evaluate dialect identification on a labeled JSONL file.
    """

    path = Path(dataset_path) if dataset_path else packaged_data_path("dialect_id_samples.jsonl")
    if not path.exists():
        raise InvalidConfigError(
            f"Dialect eval dataset not found: {path}. "
            "Pass --dialect-dataset (CLI) / dialect_dataset_path (SDK) to a JSONL file."
        )
    rows = load_dialect_id_jsonl(path)
    if max_rows is not None:
        rows = rows[: int(max_rows)]

    y_true: list[str] = []
    y_pred: list[str] = []
    examples: list[dict[str, Any]] = []

    for ex in rows:
        a = analyze_codemix(
            ex.text,
            dialect_backend=dialect_backend,
            dialect_model_id_or_path=dialect_model_id_or_path,
            allow_remote_models=allow_remote_models,
        )
        y_true.append(ex.dialect.value)
        y_pred.append(a.dialect.dialect.value)
        if len(examples) < 8:
            examples.append(
                {
                    "text": ex.text,
                    "dialect_true": ex.dialect.value,
                    "dialect_pred": a.dialect.dialect.value,
                    "confidence": float(getattr(a.dialect, "confidence", 0.0)),
                    "backend": getattr(a.dialect, "backend", "unknown"),
                }
            )

    labels = sorted(set(y_true) | set(y_pred))
    return {
        "dataset": "dialect_id",
        "path": str(path),
        "n_rows": len(rows),
        "dialect_backend": dialect_backend,
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred, labels=labels),
        "labels": labels,
        "examples": examples,
    }


def run_dialect_normalization_eval(
    *,
    dataset_path: Optional[str] = None,
    dialect_normalizer_backend: str = "heuristic",
    dialect_normalizer_model_id_or_path: Optional[str] = None,
    allow_remote_models: bool = False,
    max_rows: Optional[int] = 2000,
) -> dict[str, Any]:
    """
    Evaluate dialect normalization on a labeled JSONL file.

    We force dialect per-row via `dialect_force` so results are deterministic.
    """

    path = Path(dataset_path) if dataset_path else packaged_data_path("dialect_norm_samples.jsonl")
    if not path.exists():
        raise InvalidConfigError(
            f"Dialect normalization eval dataset not found: {path}. "
            "Pass --dialect-norm-dataset (CLI) / dialect_norm_dataset_path (SDK) to a JSONL file."
        )
    rows = load_dialect_normalization_jsonl(path)
    if max_rows is not None:
        rows = rows[: int(max_rows)]

    n = len(rows) or 1
    em_ok = 0
    chrf_vals: list[float] = []
    examples: list[dict[str, Any]] = []

    for ex in rows:
        a = analyze_codemix(
            ex.input,
            dialect_force=ex.dialect.value,
            dialect_normalize=True,
            dialect_min_confidence=0.0,
            dialect_normalizer_backend=dialect_normalizer_backend,
            dialect_normalizer_model_id_or_path=dialect_normalizer_model_id_or_path,
            allow_remote_models=allow_remote_models,
        )
        hyp_toks = list(getattr(a.dialect_normalization, "tokens_out", []))
        ref_toks = tokenize(ex.expected)
        if hyp_toks == ref_toks:
            em_ok += 1

        hyp = render_tokens(hyp_toks)
        ref = render_tokens(ref_toks)
        chrf_vals.append(_chrf(hyp, ref))

        if len(examples) < 8:
            examples.append(
                {
                    "input": ex.input,
                    "dialect": ex.dialect.value,
                    "expected": ex.expected,
                    "pred_tokens": hyp_toks,
                    "pred": hyp,
                    "backend": getattr(a.dialect_normalization, "backend", "unknown"),
                    "chrf": chrf_vals[-1],
                    "exact_match": hyp_toks == ref_toks,
                }
            )

    return {
        "dataset": "dialect_normalization",
        "path": str(path),
        "n_rows": len(rows),
        "dialect_normalizer_backend": dialect_normalizer_backend,
        "exact_match": em_ok / n,
        "chrf": sum(chrf_vals) / len(chrf_vals) if chrf_vals else 0.0,
        "examples": examples,
    }


def _require_torch_and_transformers(context: str) -> tuple[Any, Any]:
    """
    Import torch + transformers lazily.

    We keep the core package lightweight; these deps are intentionally optional.
    """
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            f"{context} requires torch. Install with: pip install -e \".[eval]\""
        ) from e

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            f"{context} requires transformers. Install with: pip install -e \".[eval]\""
        ) from e

    return torch, (AutoModel, AutoTokenizer)


@lru_cache(maxsize=2)
def _get_hf_tokenizer_and_model(model_name: str) -> tuple[Any, Any]:
    torch, (AutoModel, AutoTokenizer) = _require_torch_and_transformers("Embedding eval")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    _ = torch  # keep a reference for type checkers
    return tok, model


def _looks_like_gated_hf_repo_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        "gated repo" in msg
        or "cannot access gated repo" in msg
        or "you must have access" in msg
        or "401" in msg
        or "forbidden" in msg
    )


def _get_tokenizer_and_model_with_fallback(model_name: str) -> tuple[str, Any, Any]:
    """
    Load tokenizer+model, with a best-effort fallback for gated repos.

    If `model_name` is IndicBERT and HF access is gated, we fall back to a public multilingual
    sentence-transformers model so the eval can still run without extra auth steps.
    """
    try:
        tok, model = _get_hf_tokenizer_and_model(model_name)
        return model_name, tok, model
    except OSError as e:
        if model_name == _DEFAULT_EMBEDDING_MODEL and _looks_like_gated_hf_repo_error(e):
            tok, model = _get_hf_tokenizer_and_model(_FALLBACK_EMBEDDING_MODEL)
            return _FALLBACK_EMBEDDING_MODEL, tok, model
        raise


def _embed_texts_with_model(
    texts: Sequence[str],
    *,
    tok: Any,
    model: Any,
    batch_size: int = 16,
    max_length: int = 192,
) -> Any:
    """
    Compute normalized sentence embeddings for Gujarati (and mixed) text.

    Uses mean pooling over last_hidden_state with attention_mask.
    Returns a CPU tensor of shape (n, d).
    """
    torch, _ = _require_torch_and_transformers("Embedding eval")

    out_batches: list[Any] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if t is not None else "" for t in texts[i : i + batch_size]]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            res = model(**enc)
            last_hidden = res.last_hidden_state  # (b, t, d)
            mask = enc["attention_mask"].unsqueeze(-1)  # (b, t, 1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / denom
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out_batches.append(emb.cpu())

    return torch.cat(out_batches, dim=0) if out_batches else None


def _cosine_sim_matrix(emb: Any) -> list[list[float]]:
    """emb is assumed L2-normalized. Returns a python list matrix."""
    torch, _ = _require_torch_and_transformers("Embedding eval")
    if emb is None:
        return []
    sims = (emb @ emb.T).cpu().tolist()
    return [[float(x) for x in row] for row in sims]


_GOLDEN_TRANSLIT_CASES: list[GoldenTranslitCase] = [
    # Exceptions-only (works even when transliteration_backend() == "none")
    GoldenTranslitCase("hu", ["હું"]),
    GoldenTranslitCase("tu", ["તું"]),
    GoldenTranslitCase("tame", ["તમે"]),
    GoldenTranslitCase("ame", ["અમે"]),
    GoldenTranslitCase("maru", ["મારું"]),
    GoldenTranslitCase("mare", ["મારે"]),
    GoldenTranslitCase("tamaro", ["તમારો"]),
    GoldenTranslitCase("tamari", ["તમારી"]),
    GoldenTranslitCase("chhe", ["છે"]),
    GoldenTranslitCase("nathi", ["નથી"]),
    GoldenTranslitCase("shu", ["શું"]),
    GoldenTranslitCase("kem", ["કેમ"]),
    GoldenTranslitCase("aaje", ["આજે"]),
    GoldenTranslitCase("kaale", ["કાલે"]),
    GoldenTranslitCase("naam", ["નામ"]),
    GoldenTranslitCase("hu aaje", ["હું આજે"]),
    GoldenTranslitCase("tame kem", ["તમે કેમ"]),
    # Backends required (sanscript / ai4bharat). We skip these if backend == "none".
    GoldenTranslitCase("ahmedabad", ["અમદાવાદ"], requires_backend=True),
    GoldenTranslitCase("gujarat", ["ગુજરાત"], requires_backend=True),
    GoldenTranslitCase("sarkar", ["સરકાર"], requires_backend=True),
    GoldenTranslitCase("vyapar", ["વ્યાપાર"], requires_backend=True),
]


def run_golden_translit_eval(
    *,
    topk: int = 1,
    translit_mode: str = "token",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
) -> dict[str, Any]:
    """
    Golden Gujlish->Gujarati evaluation.

    This is intentionally tiny and hand-validated. It is meant as a regression guard
    for common function words and a few content words (when a backend is available).
    """
    backend = transliteration_backend()
    rows: list[dict[str, Any]] = []
    n_total = 0
    n_ok = 0
    n_skipped = 0

    for c in _GOLDEN_TRANSLIT_CASES:
        if c.requires_backend and backend == "none":
            n_skipped += 1
            continue

        n_total += 1
        got = render_codemix(
            c.gujlish,
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
        got_norm = normalize_text(got)
        expected_norms = [normalize_text(x) for x in c.expected_any_of]
        ok = got_norm in expected_norms
        if ok:
            n_ok += 1

        rows.append(
            {
                "input": c.gujlish,
                "output": got,
                "expected_any_of": c.expected_any_of,
                "ok": ok,
                "requires_backend": c.requires_backend,
            }
        )

    return {
        "dataset": "golden_translit",
        "transliteration_backend": backend,
        "topk": int(topk),
        "translit_mode": translit_mode,
        "preserve_case": bool(preserve_case),
        "preserve_numbers": bool(preserve_numbers),
        "aggressive_normalize": bool(aggressive_normalize),
        "n_cases": int(n_total),
        "n_ok": int(n_ok),
        "accuracy": (n_ok / n_total) if n_total else 0.0,
        "n_skipped": int(n_skipped),
        "examples_fail": [r for r in rows if not r["ok"]][:10],
        "examples_ok": [r for r in rows if r["ok"]][:10],
    }


def run_retrieval_eval(
    *,
    k_values: Sequence[int] = (1, 3, 5),
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    preprocess_query: bool = True,
) -> dict[str, Any]:
    """
    Tiny retrieval benchmark: IndicBERT embeddings + top-k recall on curated snippets.
    """
    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not k_values:
        raise InvalidConfigError("k_values must contain at least one positive integer")

    ds = load_gujarat_facts_tiny()
    docs = ds.docs
    queries = ds.queries

    doc_texts = [d.text for d in docs]
    doc_ids = [d.doc_id for d in docs]

    q_texts: list[str] = []
    for q in queries:
        s = q.query
        if preprocess_query:
            # Allows running gujlish-ish queries through the same normalization pipeline.
            s = render_codemix(normalize_text(s))
        q_texts.append(s)

    used_model, tok, model = _get_tokenizer_and_model_with_fallback(embedding_model)
    doc_emb = _embed_texts_with_model(doc_texts, tok=tok, model=model)
    q_emb = _embed_texts_with_model(q_texts, tok=tok, model=model)

    torch, _ = _require_torch_and_transformers("Retrieval eval")
    sims = (q_emb @ doc_emb.T)  # (nq, nd)

    k_max = max(k_values)
    topk = torch.topk(sims, k=k_max, dim=1).indices.cpu().tolist()

    per_query: list[dict[str, Any]] = []
    for qi, q in enumerate(queries):
        ranked_ids = [doc_ids[j] for j in topk[qi]]
        per_query.append(
            {
                "query_raw": q.query,
                "query_used": q_texts[qi],
                "relevant_doc_ids": q.relevant_doc_ids,
                "ranked_doc_ids": ranked_ids,
            }
        )

    recall_at_k: dict[str, float] = {}
    for k in k_values:
        hits = 0
        for qi, q in enumerate(queries):
            ranked_ids = per_query[qi]["ranked_doc_ids"][:k]
            if any(d in ranked_ids for d in q.relevant_doc_ids):
                hits += 1
        recall_at_k[str(k)] = hits / (len(queries) or 1)

    return {
        "dataset": "retrieval",
        "retrieval_dataset": ds.name,
        "retrieval_dataset_source": ds.source,
        "embedding_model_requested": embedding_model,
        "embedding_model_used": used_model,
        "k_values": list(k_values),
        "preprocess_query": bool(preprocess_query),
        "n_docs": len(docs),
        "n_queries": len(queries),
        "recall_at_k": recall_at_k,
        "examples": per_query[:6],
    }


def _prompt_variants(base_question_gu: str, *, n_variants: int) -> list[str]:
    """
    Deterministic prompt variants (no LLM) to test model sensitivity.
    """
    templates = [
        "આ પ્રશ્નનું જવાબ ગુજરાતીમાં આપો: {q}",
        "કૃપા કરીને ગુજરાતીમાં ટૂંકું ઉત્તર આપો: {q}",
        "માત્ર 3 મુદ્દામાં જવાબ આપો (ગુજરાતીમાં): {q}",
        "ગુજરાતીમાં સમજાવો: {q}",
        "સીધો અને સ્પષ્ટ જવાબ ગુજરાતીમાં આપો: {q}",
        "યુઝરનો પ્રશ્ન: {q}\nજવાબ ગુજરાતીમાં આપો.",
        "નીચેના પ્રશ્નનો ઉત્તર આપો (ગુજરાતીમાં): {q}",
        "પ્રશ્ન: {q}\nકૃપા કરીને જવાબ ગુજરાતીમાં લખો.",
        "ગુજરાતીમાં જવાબ આપો અને અતિશય વિગત ન આપો: {q}",
        "મહેરબાની કરીને જવાબ ગુજરાતીમાં જ રાખો: {q}",
    ]
    out: list[str] = []
    for i in range(max(1, int(n_variants))):
        t = templates[i % len(templates)]
        out.append(t.format(q=base_question_gu))
    return out


def _cache_load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run_prompt_stability_eval(
    *,
    model: str = "sarvam-m",
    n_variants: int = 10,
    base_question_gu: str = "અમદાવાદમાં શિયાળામાં કઈ ખાસ વાનગી લોકપ્રિય છે?",
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    preprocess: bool = True,
) -> dict[str, Any]:
    """
    Prompt-stability benchmark:
      - create N deterministic prompt variants
      - call Sarvam-M for each variant
      - compute semantic similarity between outputs via IndicBERT embeddings

    Responses are cached in ~/.cache/gujarati-codemix-kit to avoid repeated API calls.
    """
    try:
        from .sarvam_adapters import sarvam_chat
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "Prompt-stability eval requires Sarvam integration. Install with: "
            "pip install -e \".[sarvam,eval]\""
        ) from e

    # Fail fast with a clear message (otherwise you get a deeper stack trace from sarvam_adapters).
    import os

    api_key_value = api_key or os.environ.get("SARVAM_API_KEY")
    if not api_key_value:
        raise InvalidConfigError(
            "Missing SARVAM_API_KEY. Set it in your shell (export SARVAM_API_KEY=...) "
            "or pass --api-key."
        )

    cache_root = cache_dir or (_default_cache_dir() / "eval-cache" / "prompt-stability")
    prompts = _prompt_variants(base_question_gu, n_variants=n_variants)

    outputs: list[str] = []
    used_cache = 0
    for p in prompts:
        cache_key = _sha256_hex(f"{model}\n{preprocess}\n{p}")
        path = cache_root / f"{cache_key}.json"
        cached = _cache_load_json(path)
        if (
            cached
            and cached.get("prompt") == p
            and cached.get("model") == model
            and bool(cached.get("preprocess", bool(preprocess))) == bool(preprocess)
        ):
            outputs.append(str(cached.get("output", "")))
            used_cache += 1
            continue

        out = sarvam_chat(p, model=model, api_key=api_key_value, preprocess=preprocess)
        outputs.append(out)
        _cache_write_json(
            path,
            {
                "created_at_unix": int(time.time()),
                "model": model,
                "preprocess": bool(preprocess),
                "prompt": p,
                "output": out,
            },
        )

    used_model, tok, model_obj = _get_tokenizer_and_model_with_fallback(embedding_model)
    emb = _embed_texts_with_model(outputs, tok=tok, model=model_obj)
    sims = _cosine_sim_matrix(emb)

    # Off-diagonal stats (pairwise stability).
    vals: list[float] = []
    for i in range(len(sims)):
        for j in range(len(sims)):
            if i == j:
                continue
            vals.append(float(sims[i][j]))

    ref_sims = [float(sims[0][j]) for j in range(1, len(sims))] if len(sims) > 1 else []
    return {
        "dataset": "prompt_stability",
        "model": model,
        "n_variants": int(n_variants),
        "base_question_gu": base_question_gu,
        "embedding_model_requested": embedding_model,
        "embedding_model_used": used_model,
        "cache_dir": str(cache_root),
        "used_cache_n": int(used_cache),
        "pairwise_similarity": {
            "mean_offdiag": (sum(vals) / len(vals)) if vals else 0.0,
            "min_offdiag": min(vals) if vals else 0.0,
            "max_offdiag": max(vals) if vals else 0.0,
            "ref_mean": (sum(ref_sims) / len(ref_sims)) if ref_sims else 0.0,
            "ref_min": min(ref_sims) if ref_sims else 0.0,
        },
        "examples": [
            {"prompt": prompts[i], "output": outputs[i]} for i in range(min(len(prompts), 5))
        ],
    }


def run_eval(
    dataset: str = "gujlish",
    *,
    topk: int = 1,
    max_rows: Optional[int] = 2000,
    translit_mode: str = "token",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    k: int = 5,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    sarvam_model: str = "sarvam-m",
    n_variants: int = 10,
    api_key: Optional[str] = None,
    preprocess: bool = True,
    dialect_dataset_path: Optional[str] = None,
    dialect_norm_dataset_path: Optional[str] = None,
    dialect_backend: str = "heuristic",
    dialect_model_id_or_path: Optional[str] = None,
    dialect_normalizer_backend: str = "heuristic",
    dialect_normalizer_model_id_or_path: Optional[str] = None,
    allow_remote_models: bool = False,
) -> dict[str, Any]:
    """
    Run a lightweight eval that answers: does codemix rendering produce Gujarati script
    from Gujlish inputs, and how often?

    This is not a translation-quality benchmark. It's an MVP harness that helps show
    measurable normalization effects.
    """
    if dataset in {"golden", "golden_translit", "golden-translit"}:
        return run_golden_translit_eval(
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
    if dataset in {"retrieval"}:
        return run_retrieval_eval(
            k_values=(1, 3, int(k)),
            embedding_model=embedding_model,
            preprocess_query=preprocess,
        )
    if dataset in {"prompt_stability", "prompt-stability"}:
        return run_prompt_stability_eval(
            model=sarvam_model,
            n_variants=n_variants,
            embedding_model=embedding_model,
            api_key=api_key,
            preprocess=preprocess,
        )
    if dataset in {"dialect_id", "dialect-id"}:
        return run_dialect_id_eval(
            dataset_path=dialect_dataset_path,
            dialect_backend=dialect_backend,
            dialect_model_id_or_path=dialect_model_id_or_path,
            allow_remote_models=allow_remote_models,
            max_rows=max_rows,
        )
    if dataset in {"dialect_normalization", "dialect-normalization", "dialect_norm", "dialect-norm"}:
        return run_dialect_normalization_eval(
            dataset_path=dialect_norm_dataset_path,
            dialect_normalizer_backend=dialect_normalizer_backend,
            dialect_normalizer_model_id_or_path=dialect_normalizer_model_id_or_path,
            allow_remote_models=allow_remote_models,
            max_rows=max_rows,
        )
    if dataset != "gujlish":
        raise InvalidConfigError(
            "Unsupported dataset. Try one of: gujlish, golden_translit, retrieval, prompt_stability, dialect_id, dialect_normalization"
        )

    cache_dir = _default_cache_dir()

    per_split: dict[str, Any] = {}
    for spec in _GUJLISH_SPECS:
        csv_path = cache_dir / f"{dataset}-{spec.name}.csv"
        _download(spec.url, csv_path)

        rows: list[dict[str, Any]] = []
        for i, text in enumerate(_iter_texts_from_csv(csv_path, text_column=spec.text_column)):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(_analyze_one(text, topk=topk))

        n = len(rows) or 1
        has_gu_raw = sum(1 for r in rows if r["has_gujarati_raw"])
        has_gu_out = sum(1 for r in rows if r["has_gujarati_codemix"])
        n_gu_roman = sum(int(r["n_gu_roman_tokens"]) for r in rows)
        n_changed = sum(int(r["n_gu_roman_tokens_changed_est"]) for r in rows)

        per_split[spec.name] = {
            "n_rows": len(rows),
            "pct_has_gujarati_raw": has_gu_raw / n,
            "pct_has_gujarati_codemix": has_gu_out / n,
            "n_gu_roman_tokens": n_gu_roman,
            "pct_gu_roman_tokens_changed_est": (n_changed / n_gu_roman) if n_gu_roman else 0.0,
            "examples": rows[:8],
        }

    return {
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "topk": int(topk),
        "max_rows": None if max_rows is None else int(max_rows),
        "splits": per_split,
    }

