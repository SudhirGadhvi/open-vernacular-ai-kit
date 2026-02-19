from __future__ import annotations

import argparse
import statistics
import time

from gujarati_codemix_kit import CodeMixConfig, CodeMixPipeline, render_codemix


def _samples() -> list[str]:
    return [
        "maru business plan ready chhe!!!",
        "hu aaje office jaish!!",
        "kamaad thaalu rakhje",
        "મારું business plan ready છે!!!",
        "mne kal surat javanu chhe, diamond market mate",
        "gujarat ni rajdhani kai che?",
    ]


def _bench_pipeline(n: int, *, mode: str) -> dict[str, float]:
    texts = _samples()
    cfg = CodeMixConfig(translit_mode="sentence")
    pipe = CodeMixPipeline(config=cfg)

    # Warmup.
    for s in texts:
        _ = pipe.run(s).codemix

    times: list[float] = []
    t0 = time.perf_counter()
    if mode == "run_many":
        for _ in range(n):
            _ = pipe.run_many(texts)
    else:
        for _ in range(n):
            for s in texts:
                _ = pipe.run(s).codemix
    t1 = time.perf_counter()

    # Per-iteration timings.
    total = t1 - t0
    per_iter = total / max(1, n)
    times.append(per_iter)

    return {
        "total_s": float(total),
        "per_iter_s": float(per_iter),
        "iters": float(n),
        "n_texts": float(len(texts)),
    }


def _bench_render(n: int) -> dict[str, float]:
    texts = _samples()
    for s in texts:
        _ = render_codemix(s, translit_mode="sentence")

    vals: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        for s in texts:
            _ = render_codemix(s, translit_mode="sentence")
        vals.append(time.perf_counter() - t0)

    return {
        "runs": float(n),
        "n_texts": float(len(texts)),
        "mean_s": float(statistics.mean(vals)) if vals else 0.0,
        "p50_s": float(statistics.median(vals)) if vals else 0.0,
        "min_s": float(min(vals)) if vals else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Micro-benchmark for gujarati-codemix-kit.")
    p.add_argument("--n", type=int, default=50, help="Number of benchmark iterations.")
    p.add_argument(
        "--mode",
        choices=["run", "run_many", "render"],
        default="run_many",
        help="Benchmark pipeline.run, pipeline.run_many, or render_codemix.",
    )
    args = p.parse_args()

    n = max(1, int(args.n))
    if args.mode == "render":
        res = _bench_render(n)
        print(res)
        return
    res = _bench_pipeline(n, mode=args.mode)
    print(res)


if __name__ == "__main__":
    main()

