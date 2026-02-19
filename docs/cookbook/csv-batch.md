# CSV Batch Processing

Batch-process a CSV input and write an enriched output CSV.

```python
from gujarati_codemix_kit import process_csv_batch

summ = process_csv_batch("in.csv", "out.csv", text_column="text")
print(summ)
```

