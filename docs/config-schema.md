# Config Schema

`CodeMixConfig` is designed to be stable and JSON-serializable.

## schema_version

Configs serialized to a dict include `schema_version`:

- Older dicts without `schema_version` are accepted.
- Unknown keys are ignored by default to preserve forward compatibility.
- Set `strict=True` to reject unknown keys.

## Roundtrip

```python
from gujarati_codemix_kit import CodeMixConfig

cfg = CodeMixConfig(translit_mode="sentence", topk=2)
d = cfg.to_dict()
cfg2 = CodeMixConfig.from_dict(d)
assert cfg2.to_dict() == cfg.normalized().to_dict()
```

## Offline-first policy

Some features accept HF model ids/paths. By default, remote downloads are blocked:

- `CodeMixConfig(allow_remote_models=False)` (default)
- When blocked, APIs raise `OfflinePolicyError`.

