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
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Write CodeMix conversion stats to stderr as JSON (stdout remains the rendered string).",
    ),
) -> None:
    """Render a clean Gujarati-English code-mix string."""
    if stats:
        a = analyze_codemix(text, topk=topk, numerals=numerals)
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
        _console.print(render_codemix(text, topk=topk, numerals=numerals))


@app.command()
def eval(
    dataset: str = typer.Option("gujlish", help="Eval dataset name (currently: gujlish)."),
    report: Optional[Path] = typer.Option(
        None, help="Write a JSON report to this path (directories auto-created)."
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

    result = run_eval(dataset=dataset)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        _console.print(f"[green]Wrote report:[/green] {report}")
    else:
        _console.print(json.dumps(result, ensure_ascii=True, indent=2))
 
