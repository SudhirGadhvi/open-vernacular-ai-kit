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

from open_vernacular_ai_kit.errors import GckError  # noqa: E402
from open_vernacular_ai_kit.sarvam_teacher import (  # noqa: E402
    dump_sarvam_teacher_records_jsonl,
    load_sarvam_teacher_inputs_jsonl,
    mine_sarvam_teacher_candidate,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mine reviewable Hindi/Gujarati normalization candidates using Sarvam."
    )
    ap.add_argument("--input", required=True, help="Input JSONL with `text` or `input` per row.")
    ap.add_argument("--output", required=True, help="Output JSONL for mined candidate records.")
    ap.add_argument("--model", default="sarvam-m", help="Sarvam model id.")
    ap.add_argument("--api-key", default=None, help="Sarvam API key override.")
    ap.add_argument(
        "--language-hint",
        default=None,
        help="Optional global language hint override (`gu`, `hi`, `mixed`, `unknown`).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional max input rows to process. Use 0 for all rows.",
    )
    ap.add_argument(
        "--exclude-raw-response",
        action="store_true",
        help="Do not write the raw model response into output JSONL.",
    )
    args = ap.parse_args()

    try:
        inputs = load_sarvam_teacher_inputs_jsonl(args.input)
        if args.max_rows and args.max_rows > 0:
            inputs = inputs[: args.max_rows]

        out = []
        for row in inputs:
            rec = mine_sarvam_teacher_candidate(
                row.text,
                model=args.model,
                api_key=args.api_key,
                language_hint=args.language_hint or row.language_hint,
                source=row.source,
                meta=row.meta,
            )
            out.append(rec)

        dump_sarvam_teacher_records_jsonl(
            args.output,
            out,
            include_raw_response=not args.exclude_raw_response,
        )

        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "model": args.model,
                    "n_rows": len(out),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"mine_sarvam_candidates: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
