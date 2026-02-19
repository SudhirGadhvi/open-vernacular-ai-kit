# Benchmarks

This repo includes a tiny, dependency-light micro-benchmark:

```bash
python3 scripts/bench.py --mode run_many --n 50
python3 scripts/bench.py --mode run --n 50
python3 scripts/bench.py --mode render --n 50
```

Notes:

- This is meant for regression tracking, not absolute performance claims.
- Real-world throughput depends on optional backends (transliteration engines, transformers models, etc).

## Quality / Coverage (Public Gujlish Eval)

This project also includes a lightweight, reproducible "coverage-style" eval on public Gujlish
datasets. It answers the question:

> After `codemix` rendering, how often does the output contain Gujarati script (i.e., did we convert
> romanized Gujarati tokens into Gujarati)?

Run:

```bash
gck eval --dataset gujlish --report eval/out/report.json
```

Key fields in the JSON report:

- `pct_has_gujarati_codemix`: fraction of rows where output contains Gujarati script
- `pct_gu_roman_tokens_changed_est`: fraction of detected Gujlish tokens that were transliterated

Important caveat:

- This is not a translation benchmark; it measures normalization / script conversion effects.

Example (from one local run with `topk=1`, `max_rows=2000`):

- Split `in22`: `pct_has_gujarati_codemix` ~= `0.987`
- Split `xnli`: `pct_has_gujarati_codemix` ~= `0.956`


