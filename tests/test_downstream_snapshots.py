from __future__ import annotations

from open_vernacular_ai_kit import downstream_snapshots


def test_snapshot_downstream_uplift_collects_all_requested_packs(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_run_retrieval_uplift_eval(*, k_values, embedding_model, retrieval_query_pack):
        calls.append(retrieval_query_pack)
        return {
            "dataset": "retrieval_uplift",
            "retrieval_query_pack": retrieval_query_pack,
            "embedding_model_requested": embedding_model,
            "embedding_model_used": "test-model",
            "k_values": list(k_values),
            "raw_eval": {"n_queries": 6, "recall_at_k": {"1": 0.5, "3": 1.0}},
            "normalized_eval": {"n_queries": 6, "recall_at_k": {"1": 0.75, "3": 1.0}},
            "recall_uplift": {
                "1": {"raw": 0.5, "normalized": 0.75, "absolute_uplift": 0.25},
                "3": {"raw": 1.0, "normalized": 1.0, "absolute_uplift": 0.0},
            },
        }

    monkeypatch.setattr(downstream_snapshots, "run_retrieval_uplift_eval", _fake_run_retrieval_uplift_eval)

    payload = downstream_snapshots.snapshot_downstream_uplift(
        retrieval_query_packs=("default", "codemix_hard"),
        k_values=(1, 3),
        embedding_model="test-embed",
    )

    assert calls == ["default", "codemix_hard"]
    assert payload["snapshot_config"]["retrieval_query_packs"] == ["default", "codemix_hard"]
    assert payload["snapshot_config"]["k_values"] == [1, 3]
    assert payload["downstream_uplift_metrics"]["retrieval_uplift"]["default"]["n_queries"] == 6
    assert (
        payload["downstream_uplift_metrics"]["retrieval_uplift"]["codemix_hard"]["recall_uplift"]["1"][
            "absolute_uplift"
        ]
        == 0.25
    )


def test_snapshot_downstream_uplift_rejects_empty_pack_list() -> None:
    try:
        downstream_snapshots.snapshot_downstream_uplift(retrieval_query_packs=())
    except ValueError as exc:
        assert "at least one pack" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_snapshot_downstream_uplift_can_include_prompt_stability(monkeypatch) -> None:
    monkeypatch.setattr(
        downstream_snapshots,
        "run_retrieval_uplift_eval",
        lambda **kwargs: {
            "dataset": "retrieval_uplift",
            "retrieval_query_pack": kwargs["retrieval_query_pack"],
            "embedding_model_requested": kwargs["embedding_model"],
            "embedding_model_used": "test-model",
            "k_values": list(kwargs["k_values"]),
            "raw_eval": {"n_queries": 6, "recall_at_k": {"1": 1.0}},
            "normalized_eval": {"n_queries": 6, "recall_at_k": {"1": 1.0}},
            "recall_uplift": {"1": {"raw": 1.0, "normalized": 1.0, "absolute_uplift": 0.0}},
        },
    )
    monkeypatch.setattr(
        downstream_snapshots,
        "run_prompt_stability_uplift_eval",
        lambda **kwargs: {
            "dataset": "prompt_stability_uplift",
            "model": kwargs["model"],
            "n_variants": kwargs["n_variants"],
            "embedding_model_requested": kwargs["embedding_model"],
            "embedding_model_used": "test-model",
            "raw_eval": {
                "used_cache_n": 2,
                "pairwise_similarity": {"mean_offdiag": 0.5, "ref_mean": 0.55},
            },
            "normalized_eval": {
                "used_cache_n": 2,
                "pairwise_similarity": {"mean_offdiag": 0.7, "ref_mean": 0.75},
            },
            "pairwise_similarity_uplift": {
                "mean_offdiag": {"raw": 0.5, "normalized": 0.7, "absolute_uplift": 0.2},
                "ref_mean": {"raw": 0.55, "normalized": 0.75, "absolute_uplift": 0.2},
            },
        },
    )

    payload = downstream_snapshots.snapshot_downstream_uplift(
        retrieval_query_packs=("default",),
        include_prompt_stability=True,
        prompt_model="sarvam-m",
        prompt_n_variants=8,
        prompt_base_question_gu="શું હાલ સ્થિતિ સારી છે?",
    )

    assert payload["snapshot_config"]["prompt_stability"]["included"] is True
    assert payload["snapshot_config"]["prompt_stability"]["model"] == "sarvam-m"
    assert (
        payload["downstream_uplift_metrics"]["prompt_stability_uplift"]["pairwise_similarity_uplift"][
            "mean_offdiag"
        ]["absolute_uplift"]
        == 0.2
    )
