from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .codemix_render import analyze_codemix, render_codemix
from .normalize import normalize_text

app = typer.Typer(add_completion=False, no_args_is_help=True)
_console = Console()


@app.command()
def normalize(
    text: str = typer.Argument(..., help="Input text (Gujarati / English / code-mixed)."),
    numerals: str = typer.Option("keep", help="Numerals: keep (Gujarati digits) or ascii."),
) -> None:
    """Normalize punctuation, whitespace, and Gujarati script."""
    _console.print(normalize_text(text, numerals=numerals))


@app.command()
def codemix(
    text: str = typer.Argument(..., help="Input text (may include romanized Gujarati/Gujlish)."),
    topk: int = typer.Option(1, help="Top-K transliteration candidates to consider."),
    numerals: str = typer.Option("keep", help="Numerals: keep (Gujarati digits) or ascii."),
    translit_mode: str = typer.Option(
        "token", help="Transliteration mode for Gujlish: token or sentence."
    ),
    preserve_case: bool = typer.Option(
        True, help="Preserve original case for Latin tokens (English + Gujlish)."
    ),
    preserve_numbers: bool = typer.Option(
        True, help="Preserve Gujarati digits (disable to normalize Gujarati digits to ASCII)."
    ),
    aggressive_normalize: bool = typer.Option(
        False, help="Try extra Gujlish spelling variants before transliteration."
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Write CodeMix conversion stats to stderr as JSON (stdout remains the rendered string).",
    ),
) -> None:
    """Render a clean Gujarati-English code-mix string."""
    if stats:
        a = analyze_codemix(
            text,
            topk=topk,
            numerals=numerals,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
        _console.print(a.codemix)
        sys.stderr.write(
            json.dumps(
                {
                    "transliteration_backend": a.transliteration_backend,
                    "n_tokens": a.n_tokens,
                    "n_gu_roman_tokens": a.n_gu_roman_tokens,
                    "n_gu_roman_transliterated": a.n_gu_roman_transliterated,
                    "pct_gu_roman_transliterated": a.pct_gu_roman_transliterated,
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    else:
        _console.print(
            render_codemix(
                text,
                topk=topk,
                numerals=numerals,
                translit_mode=translit_mode,
                preserve_case=preserve_case,
                preserve_numbers=preserve_numbers,
                aggressive_normalize=aggressive_normalize,
            )
        )


@app.command()
def eval(
    dataset: str = typer.Option(
        "gujlish",
        help="Eval dataset/suite: gujlish, golden_translit, retrieval, prompt_stability.",
    ),
    report: Optional[Path] = typer.Option(
        None, help="Write a JSON report to this path (directories auto-created)."
    ),
    topk: int = typer.Option(1, help="Top-K transliteration candidates (gujlish/golden_translit)."),
    max_rows: Optional[int] = typer.Option(
        2000, help="Max rows per split (gujlish). Use 0 for no limit."
    ),
    translit_mode: str = typer.Option(
        "token", help="Transliteration mode: token or sentence (golden_translit)."
    ),
    k: int = typer.Option(5, help="Top-k for retrieval recall (retrieval)."),
    embedding_model: str = typer.Option(
        "ai4bharat/indic-bert",
        help=(
            "HF model for embeddings (retrieval/prompt_stability). "
            "Note: ai4bharat/indic-bert may be gated on HF (the eval will fall back automatically)."
        ),
    ),
    sarvam_model: str = typer.Option("sarvam-m", help="Sarvam chat model (prompt_stability)."),
    n_variants: int = typer.Option(10, help="Number of prompt variants (prompt_stability)."),
    api_key: Optional[str] = typer.Option(None, help="Sarvam API key override (prompt_stability)."),
    preprocess: bool = typer.Option(
        True, help="Preprocess text with normalize+codemix before eval (retrieval/prompt_stability)."
    ),
) -> None:
    """Run a lightweight, reproducible eval harness (downloads data if needed)."""
    try:
        from .eval_harness import run_eval
    except Exception:
        # Keep core library lightweight; eval dependencies are optional.
        _console.print(
            "[red]Eval harness is not installed.[/red] Install with: "
            "`pip install -e \".[eval]\"`"
        )
        raise typer.Exit(code=2)

    if max_rows == 0:
        max_rows = None
    try:
        result = run_eval(
            dataset=dataset,
            topk=topk,
            max_rows=max_rows,
            translit_mode=translit_mode,
            k=k,
            embedding_model=embedding_model,
            sarvam_model=sarvam_model,
            n_variants=n_variants,
            api_key=api_key,
            preprocess=preprocess,
        )
    except Exception as e:
        _console.print(f"[red]Eval failed:[/red] {e}")
        raise typer.Exit(code=1)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        _console.print(f"[green]Wrote report:[/green] {report}")
    else:
        _console.print(json.dumps(result, ensure_ascii=True, indent=2))
 
