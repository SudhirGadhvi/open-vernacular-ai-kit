from __future__ import annotations

import argparse
import json

from open_vernacular_ai_kit.eval_harness import run_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gujlish")
    p.add_argument("--topk", default=1, type=int)
    p.add_argument("--max-rows", default=2000, type=int)
    args = p.parse_args()

    res = run_eval(dataset=args.dataset, topk=args.topk, max_rows=args.max_rows)
    print(json.dumps(res, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

