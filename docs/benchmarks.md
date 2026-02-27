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

## North-Star Baseline Snapshot

For release tracking, generate the 3-metric baseline snapshot:

```bash
python3 scripts/snapshot_north_star_metrics.py --output docs/data/north_star_metrics_snapshot.json --iterations 200
```

Current snapshot (`2026-02-27T19:23:14Z`):

| Metric | Value | Notes |
| --- | --- | --- |
| `transliteration_success` | `1.000` | Golden transliteration accuracy (`17/17`; backend=`none`) |
| `dialect_accuracy` | `0.833` | Heuristic dialect-id accuracy (`5/6`) |
| `p95_latency_ms` | `0.174` | Pipeline p95 latency in ms (`iterations=200`, `n_calls=1200`) |

## Quality / Coverage (Gujarati Baseline Eval)

This project also includes a lightweight, reproducible "coverage-style" eval on public Gujarati
romanization data. It answers the question:

> After `codemix` rendering, how often does the output contain native Gujarati script (i.e., did we
> convert romanized Gujarati tokens into Gujarati)?

Run:

```bash
gck eval --dataset gujlish --report eval/out/report.json
```

Key fields in the JSON report:

- `pct_has_gujarati_codemix`: fraction of rows where output contains Gujarati script
- `pct_gu_roman_tokens_changed_est`: fraction of detected romanized Gujarati tokens that were transliterated

Important caveat:

- This is not a translation benchmark; it measures normalization / script conversion effects.

Example (from one local run with `topk=1`, `max_rows=2000`):

- Split `in22`: `pct_has_gujarati_codemix` ~= `0.987`
- Split `xnli`: `pct_has_gujarati_codemix` ~= `0.956`
