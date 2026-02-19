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

