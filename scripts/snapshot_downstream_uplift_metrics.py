from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_vernacular_ai_kit.downstream_snapshots import snapshot_downstream_uplift  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot downstream uplift benchmark metrics.")
    ap.add_argument(
        "--output",
        default="docs/data/downstream_uplift_snapshot.json",
        help="Path to JSON output file.",
    )
    ap.add_argument(
        "--retrieval-query-packs",
        default="default,codemix,codemix_hard",
        help="Comma-separated retrieval query packs to snapshot.",
    )
    ap.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated top-k values for retrieval uplift.",
    )
    ap.add_argument(
        "--embedding-model",
        default="ai4bharat/indic-bert",
        help="Embedding model requested for retrieval uplift.",
    )
    args = ap.parse_args()

    payload = snapshot_downstream_uplift(
        retrieval_query_packs=_parse_csv(args.retrieval_query_packs),
        k_values=_parse_int_csv(args.k_values),
        embedding_model=str(args.embedding_model),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote snapshot: {out_path}")
    print(json.dumps(payload["downstream_uplift_metrics"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
