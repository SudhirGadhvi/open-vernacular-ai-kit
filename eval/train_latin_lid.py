"""Train a small latin-token classifier: EN vs GU_ROMAN.

Output: src/open_vernacular_ai_kit/_data/latin_lid.joblib

This script is intentionally separate from the library so that:
- the core package stays lightweight
- training deps stay optional
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        import joblib
        import numpy as np
        import wordfreq
        from datasets import load_dataset
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception as e:
        raise SystemExit(
            "Missing deps. Install: pip install -e '.[ml,eval]' "
            "(eval extra provides datasets/wordfreq; ml provides sklearn/joblib/numpy)"
        ) from e

    rng = np.random.default_rng(7)

    # Positive class: romanized Gujarati tokens from Aksharantar.
    # We stream the global dataset and filter to Gujarati script examples.
    import regex as re

    gu_re = re.compile(r"[\p{Gujarati}]+")

    stream = load_dataset("ai4bharat/aksharantar", streaming=True)["train"]
    roman_set: set[str] = set()
    for row in stream:
        native = row.get("native word")
        roman_tok = row.get("english word")
        if not isinstance(native, str) or not isinstance(roman_tok, str):
            continue
        if not gu_re.search(native):
            continue
        roman_tok = roman_tok.strip().lower()
        if 2 <= len(roman_tok) <= 24 and roman_tok.isascii() and roman_tok.isalpha():
            roman_set.add(roman_tok)
        if len(roman_set) >= 80000:
            break

    roman = list(roman_set)
    rng.shuffle(roman)
    roman = roman[:60000]

    # Negative class: frequent English words.
    en_words = [w for w in wordfreq.top_n_list("en", 120000) if w.isalpha()]
    rng.shuffle(en_words)
    en_words = en_words[:60000]

    X = en_words + roman
    y = np.array([0] * len(en_words) + [1] * len(roman), dtype=np.int64)

    pipeline = Pipeline(
        steps=[
            ("vec", CountVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True)),
            ("clf", LogisticRegression(max_iter=300, n_jobs=1)),
        ]
    )

    pipeline.fit(X, y)

    out_path = _repo_root() / "src/open_vernacular_ai_kit/_data/latin_lid.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
 
