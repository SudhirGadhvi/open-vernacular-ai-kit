from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any, Sequence

from .eval_harness import _DEFAULT_EMBEDDING_MODEL, run_retrieval_uplift_eval


def _compact_retrieval_uplift(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_query_pack": str(result["retrieval_query_pack"]),
        "embedding_model_requested": str(result["embedding_model_requested"]),
        "embedding_model_used": str(result["embedding_model_used"]),
        "k_values": list(result["k_values"]),
        "n_queries": int(result["raw_eval"]["n_queries"]),
        "raw_recall_at_k": dict(result["raw_eval"]["recall_at_k"]),
        "normalized_recall_at_k": dict(result["normalized_eval"]["recall_at_k"]),
        "recall_uplift": dict(result["recall_uplift"]),
    }


def snapshot_downstream_uplift(
    *,
    retrieval_query_packs: Sequence[str] = ("default", "codemix", "codemix_hard"),
    k_values: Sequence[int] = (1, 3, 5),
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    packs = [str(x).strip() for x in retrieval_query_packs if str(x).strip()]
    if not packs:
        raise ValueError("retrieval_query_packs must contain at least one pack")

    retrieval_snapshots: dict[str, Any] = {}
    for pack in packs:
        result = run_retrieval_uplift_eval(
            k_values=tuple(int(k) for k in k_values),
            embedding_model=embedding_model,
            retrieval_query_pack=pack,
        )
        retrieval_snapshots[pack] = _compact_retrieval_uplift(result)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "metric_definitions": {
            "retrieval_uplift": (
                "Top-k retrieval recall delta from run_retrieval_uplift_eval: compares raw "
                "queries vs OVAK-normalized queries on packaged retrieval query packs."
            ),
        },
        "snapshot_config": {
            "retrieval_query_packs": packs,
            "k_values": [int(k) for k in k_values],
            "embedding_model_requested": embedding_model,
        },
        "downstream_uplift_metrics": {
            "retrieval_uplift": retrieval_snapshots,
        },
    }
