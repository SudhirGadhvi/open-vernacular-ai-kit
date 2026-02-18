from __future__ import annotations

from gujarati_codemix_kit.eval_harness import _default_cache_dir, _download  # noqa: SLF001


def main() -> None:
    cache = _default_cache_dir()
    urls = [
        (
            "in22",
            "https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20IN22.csv",
        ),
        (
            "xnli",
            "https://raw.githubusercontent.com/mukund302002/Gujlish-English-Translation/main/Evaluation%20Dataset%20XNLI.csv",
        ),
    ]

    for name, url in urls:
        dest = cache / f"gujlish-{name}.csv"
        _download(url, dest)
        print(f"Downloaded: {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

