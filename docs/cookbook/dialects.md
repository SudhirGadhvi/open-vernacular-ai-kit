# Dialects

Dialect utilities are offline-first and pluggable.

Heuristic dialect normalization (rules) gated by confidence:

```python
from gujarati_codemix_kit import analyze_codemix

a = analyze_codemix(
    "kamaad thaalu rakhje",
    dialect_backend="heuristic",
    dialect_normalize=True,
    dialect_min_confidence=0.7,
)
print(a.codemix)
```

Transformers backends require optional extras and usually a local model path:

```bash
pip install -e ".[dialect-ml]"
```

