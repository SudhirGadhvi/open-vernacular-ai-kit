from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import regex as re

from .config import CodeMixConfig
from .dialects import GujaratiDialect
from .pipeline import CodeMixPipeline


@dataclass(frozen=True)
class WhatsAppMessage:
    timestamp: Optional[str]
    author: Optional[str]
    text: str
    is_system: bool = False


# Common WhatsApp export prefixes:
# Android: "12/31/23, 9:41 PM - Name: message"
# iOS:     "[12/31/23, 9:41:12 PM] Name: message"
_WA_ANDROID_RE = re.compile(
    r"^\s*(?P<date>\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?|\d{1,2}:\d{2})\s*-\s*"
    r"(?P<rest>.*)$",
    flags=re.IGNORECASE,
)
_WA_IOS_RE = re.compile(
    r"^\s*\[(?P<date>\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?|\d{1,2}:\d{2})\]\s*"
    r"(?P<rest>.*)$",
    flags=re.IGNORECASE,
)


def _split_author_and_text(rest: str) -> tuple[Optional[str], str, bool]:
    # System messages typically have no "Author:" prefix.
    if ":" not in rest:
        return None, rest.strip(), True
    before, after = rest.split(":", 1)
    author = before.strip()
    text = after.lstrip()
    # Heuristic: if author is empty, treat as system line.
    if not author:
        return None, rest.strip(), True
    return author, text, False


def parse_whatsapp_export(text: str) -> list[WhatsAppMessage]:
    """
    Parse a WhatsApp exported chat into messages.

    This is intentionally tolerant: formats vary across Android/iOS and locales.
    """

    lines = (text or "").splitlines()
    out: list[WhatsAppMessage] = []
    cur: Optional[WhatsAppMessage] = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        # Trim trailing whitespace introduced by joins.
        out.append(
            WhatsAppMessage(
                timestamp=cur.timestamp,
                author=cur.author,
                text=(cur.text or "").strip(),
                is_system=bool(cur.is_system),
            )
        )
        cur = None

    for line in lines:
        m = _WA_ANDROID_RE.match(line) or _WA_IOS_RE.match(line)
        if m:
            flush()
            ts = f"{m.group('date')}, {m.group('time')}".strip()
            rest = (m.group("rest") or "").strip()
            author, msg, is_system = _split_author_and_text(rest)
            cur = WhatsAppMessage(timestamp=ts, author=author, text=msg, is_system=is_system)
            continue

        # Continuation line for a multi-line message.
        if cur is None:
            # If export starts mid-message or has non-standard preamble, keep as a system line.
            cur = WhatsAppMessage(timestamp=None, author=None, text=line, is_system=True)
        else:
            cur = WhatsAppMessage(
                timestamp=cur.timestamp,
                author=cur.author,
                text=f"{cur.text}\n{line}",
                is_system=cur.is_system,
            )

    flush()
    return out


def clean_whatsapp_chat_text(
    text: str,
    *,
    keep_author: bool = False,
    drop_system_messages: bool = True,
    drop_media_omitted: bool = True,
) -> str:
    """
    Convenience cleaner for WhatsApp exports so downstream NLP sees "just the message content".
    """

    msgs = parse_whatsapp_export(text)
    cleaned: list[str] = []

    for m in msgs:
        if drop_system_messages and m.is_system:
            continue
        msg = (m.text or "").strip()
        if not msg:
            continue
        if drop_media_omitted:
            low = msg.lower()
            if "<media omitted>" in low or "media omitted" in low or "image omitted" in low:
                continue
        if keep_author and m.author:
            cleaned.append(f"{m.author}: {msg}")
        else:
            cleaned.append(msg)

    return "\n".join(cleaned).strip()


@dataclass(frozen=True)
class BatchProcessSummary:
    n_rows_in: int
    n_rows_out: int
    n_errors: int


def process_csv_batch(
    in_path: str | Path,
    out_path: str | Path,
    *,
    text_column: str = "text",
    output_column: str = "codemix",
    config: Optional[CodeMixConfig] = None,
    add_metrics: bool = True,
) -> BatchProcessSummary:
    """
    Batch-process a CSV of text rows using the SDK pipeline.

    Uses only stdlib `csv` to keep core dependencies light.
    """

    in_p = Path(in_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pipe = CodeMixPipeline(config=config or CodeMixConfig())
    n_in = 0
    n_out = 0
    n_err = 0

    with in_p.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row (fieldnames).")
        fieldnames = list(reader.fieldnames)
        if output_column not in fieldnames:
            fieldnames.append(output_column)
        if add_metrics:
            for extra in ("cmi", "switch_points", "dialect"):
                if extra not in fieldnames:
                    fieldnames.append(extra)

        with out_p.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                n_in += 1
                try:
                    txt = str(row.get(text_column, "") or "")
                    res = pipe.run(txt)
                    row[output_column] = res.codemix
                    if add_metrics:
                        row["cmi"] = f"{res.codeswitch.cmi:.2f}"
                        row["switch_points"] = str(res.codeswitch.n_switch_points)
                        row["dialect"] = res.dialect.dialect.value
                except Exception:
                    n_err += 1
                    row[output_column] = ""
                    if add_metrics:
                        row["cmi"] = ""
                        row["switch_points"] = ""
                        row["dialect"] = GujaratiDialect.UNKNOWN.value
                writer.writerow(row)
                n_out += 1

    return BatchProcessSummary(n_rows_in=n_in, n_rows_out=n_out, n_errors=n_err)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            yield json.loads(s)


def process_jsonl_batch(
    in_path: str | Path,
    out_path: str | Path,
    *,
    text_key: str = "text",
    output_key: str = "codemix",
    config: Optional[CodeMixConfig] = None,
    add_metrics: bool = True,
) -> BatchProcessSummary:
    """
    Batch-process a JSONL file, adding `codemix` + optional metrics per record.
    """

    in_p = Path(in_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pipe = CodeMixPipeline(config=config or CodeMixConfig())
    n_in = 0
    n_out = 0
    n_err = 0

    with in_p.open("r", encoding="utf-8") as f_in, out_p.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            s = (line or "").strip()
            if not s:
                continue
            n_in += 1
            try:
                rec = json.loads(s)
                txt = str(rec.get(text_key, "") or "")
                res = pipe.run(txt)
                rec[output_key] = res.codemix
                if add_metrics:
                    rec["cmi"] = res.codeswitch.cmi
                    rec["switch_points"] = res.codeswitch.n_switch_points
                    rec["dialect"] = res.dialect.dialect.value
                f_out.write(json.dumps(rec, ensure_ascii=True) + "\n")
                n_out += 1
            except Exception:
                n_err += 1
                # Best-effort: keep malformed rows out of output to preserve JSONL validity.
                continue

    return BatchProcessSummary(n_rows_in=n_in, n_rows_out=n_out, n_errors=n_err)

