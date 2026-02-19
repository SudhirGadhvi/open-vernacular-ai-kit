from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import regex as re

# Keep normalization consistent with transliteration exception matching:
# - lower-case
# - keep Latin letters only
_LATIN_ONLY_RE = re.compile(r"[^\p{Latin}]+", flags=re.VERSION1)


def normalize_lexicon_key(key: str) -> str:
    k = (key or "").strip().lower()
    k = _LATIN_ONLY_RE.sub("", k)
    return k


@dataclass(frozen=True)
class LexiconLoadResult:
    mappings: dict[str, str]
    source: str


def _load_yaml_text(text: str) -> object:
    """
    YAML is optional: requires `PyYAML` to be installed.

    We keep the core package lightweight, so this only activates when the user
    installs the optional extra.
    """
    try:
        import yaml  # type: ignore[reportMissingImports]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "YAML lexicon support requires PyYAML. Install with: pip install -e \".[lexicon]\""
        ) from e

    return yaml.safe_load(text)


def load_user_lexicon(path: Optional[str]) -> LexiconLoadResult:
    """
    Load a user lexicon (custom roman->Gujarati mappings) from JSON or YAML.

    Expected file format: a mapping/dict of { "roman_key": "GujaratiValue", ... }.
    Keys are normalized (lower + latin-only) so that lookups remain stable across punctuation.
    """
    if not path:
        return LexiconLoadResult(mappings={}, source="none")

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"User lexicon not found: {p}")
    if not p.is_file():
        raise ValueError(f"User lexicon path is not a file: {p}")

    raw = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    payload: object
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml_text(raw)
    else:
        raise ValueError(f"Unsupported lexicon format: {suffix} (use .json/.yaml/.yml)")

    if not isinstance(payload, dict):
        raise ValueError("User lexicon must be a JSON/YAML object (mapping of key->value).")

    out: dict[str, str] = {}
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        nk = normalize_lexicon_key(k)
        nv = v.strip()
        if not nk or not nv:
            continue
        out[nk] = nv

    return LexiconLoadResult(mappings=out, source=str(p))

