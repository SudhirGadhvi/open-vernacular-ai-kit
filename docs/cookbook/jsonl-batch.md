# JSONL Batch Processing

Batch-process a JSONL input and write an enriched output JSONL.

```python
from open_vernacular_ai_kit import process_jsonl_batch

summ = process_jsonl_batch("in.jsonl", "out.jsonl", text_key="text")
print(summ)
```

