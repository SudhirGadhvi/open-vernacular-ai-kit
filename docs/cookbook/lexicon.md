# User Lexicon

If you have specific roman->Gujarati mappings that must be enforced, provide a user lexicon.

Example JSON:

```json
{
  "mane": "\u0aae\u0aa8\u0ac7",
  "kyare": "\u0a95\u0acd\u0aaf\u0abe\u0ab0\u0ac7"
}
```

Usage:

```python
from gujarati_codemix_kit import CodeMixConfig, CodeMixPipeline

cfg = CodeMixConfig(user_lexicon_path="lex.json")
out = CodeMixPipeline(config=cfg).run("mane ok chhe?").codemix
print(out)
```

