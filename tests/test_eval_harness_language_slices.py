from __future__ import annotations

from open_vernacular_ai_kit.eval_harness import run_eval, run_golden_translit_eval


def test_golden_translit_eval_supports_language_slices() -> None:
    res = run_golden_translit_eval(language="all", translit_mode="sentence")
    assert res["dataset"] == "golden_translit"
    assert res["language"] == "all"
    assert "gu" in res["language_slices"]
    assert "hi" in res["language_slices"]
    assert int(res["language_slices"]["gu"]["n_cases"]) > 0
    assert int(res["language_slices"]["hi"]["n_cases"]) > 0


def test_golden_eval_falls_back_to_default_for_unknown_language() -> None:
    res = run_eval(dataset="golden_translit", language="unknown-lang")
    assert res["dataset"] == "golden_translit"
    assert res["language"] == "gu"
    assert res["language_requested"] == "unknown-lang"
