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
from open_vernacular_ai_kit.sarvam_review import (  # noqa: E402
    dump_reviewed_records_jsonl,
    init_review_records_from_candidates,
)
from open_vernacular_ai_kit.sarvam_teacher import SarvamTeacherCandidateRecord  # noqa: E402


def _load_candidate_rows(path: str | Path) -> list[SarvamTeacherCandidateRecord]:
    import json
    from pathlib import Path

    p = Path(path)
    rows: list[SarvamTeacherCandidateRecord] = []
    from open_vernacular_ai_kit.sarvam_teacher import parse_sarvam_teacher_response

    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        rows.append(
            parse_sarvam_teacher_response(
                json.dumps(
                    {
                        "language_hint": rec.get("language_hint"),
                        "sarvam_native": rec.get("sarvam_native"),
                        "sarvam_canonical": rec.get("sarvam_canonical"),
                        "english_tokens_keep": rec.get("english_tokens_keep", []),
                        "candidate_tokens": rec.get("candidate_tokens", []),
                        "notes": rec.get("notes", ""),
                    },
                    ensure_ascii=False,
                ),
                input_text=str(rec.get("input", "") or ""),
                source=str(rec.get("source", "unknown") or "unknown"),
                model=str(rec.get("model", "sarvam-m") or "sarvam-m"),
                ovak_baseline=str(rec.get("ovak_baseline", "") or ""),
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
                fallback_language_hint=rec.get("language_hint"),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Initialize a reviewed JSONL scaffold from mined Sarvam candidate output."
    )
    ap.add_argument("--input", required=True, help="Mined candidate JSONL input.")
    ap.add_argument("--output", required=True, help="Reviewed JSONL output.")
    ap.add_argument(
        "--default-action",
        default="pending",
        help="Initial review action for all rows.",
    )
    args = ap.parse_args()

    try:
        rows = _load_candidate_rows(args.input)
        reviewed = init_review_records_from_candidates(rows, default_action=args.default_action)
        dump_reviewed_records_jsonl(args.output, reviewed, include_raw_response=False)
        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "n_rows": len(reviewed),
                    "default_action": args.default_action,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"init_sarvam_review: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
