# SDK

The intended public SDK surface centers on:

- `CodeMixConfig`
- `CodeMixPipeline`
- `render_codemix(...)` / `analyze_codemix(...)`

## Minimal usage

```python
from open_vernacular_ai_kit import render_codemix

out = render_codemix("maru business plan ready chhe!!!", translit_mode="sentence")
print(out)
```

## Pipeline usage

```python
from open_vernacular_ai_kit import CodeMixConfig, CodeMixPipeline

cfg = CodeMixConfig(translit_mode="sentence")
pipe = CodeMixPipeline(config=cfg)
res = pipe.run("hu aaje office jaish!!")
print(res.codemix)
```

## Batch usage

```python
from open_vernacular_ai_kit import CodeMixPipeline

pipe = CodeMixPipeline()
results = pipe.run_many(["maru plan", "hu aaje office"])
print([r.codemix for r in results])
```

