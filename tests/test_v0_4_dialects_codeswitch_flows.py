from __future__ import annotations

import csv
import json

from gujarati_codemix_kit.app_flows import (
    clean_whatsapp_chat_text,
    process_csv_batch,
    process_jsonl_batch,
)
from gujarati_codemix_kit.codeswitch import compute_code_switch_metrics
from gujarati_codemix_kit.dialects import GujaratiDialect, normalize_dialect_tokens
from gujarati_codemix_kit.token_lid import Token, TokenLang, tokenize


def test_codeswitch_metrics_basic() -> None:
    tagged = [
        Token(text="hu", lang=TokenLang.GU_ROMAN),
        Token(text="office", lang=TokenLang.EN),
        Token(text="jaish", lang=TokenLang.GU_ROMAN),
    ]
    cs = compute_code_switch_metrics(tagged)
    assert cs.n_tokens_total == 3
    assert cs.n_tokens_considered == 3
    assert cs.n_gu_tokens == 2
    assert cs.n_en_tokens == 1
    assert cs.n_switch_points == 2
    assert cs.n_spans == 3
    assert 33.0 < cs.cmi < 34.0


def test_dialect_normalization_kathiawadi_token() -> None:
    toks = tokenize("kamaad thaalu rakhje")
    res = normalize_dialect_tokens(toks)
    assert res.dialect in {GujaratiDialect.KATHIAWADI, GujaratiDialect.UNKNOWN}
    # We expect at least "kamaad" -> "દરવાજો" when Kathiawadi is detected.
    if res.dialect == GujaratiDialect.KATHIAWADI:
        assert "દરવાજો" in res.tokens_out


def test_clean_whatsapp_chat_text_strips_system_and_media() -> None:
    raw = "\n".join(
        [
            "12/31/23, 9:41 PM - Alice: maru plan ready chhe",
            "12/31/23, 9:42 PM - Bob: <Media omitted>",
            "12/31/23, 9:43 PM - Messages to this group are now secured with end-to-end encryption.",
            "12/31/23, 9:44 PM - Alice: ok",
            "continued line",
        ]
    )
    cleaned = clean_whatsapp_chat_text(raw)
    assert "Media omitted" not in cleaned
    assert "secured" not in cleaned.lower()
    assert "maru plan ready chhe" in cleaned
    assert "ok\ncontinued line" in cleaned


def test_process_jsonl_batch_adds_codemix_and_metrics(tmp_path) -> None:
    in_p = tmp_path / "in.jsonl"
    out_p = tmp_path / "out.jsonl"
    in_p.write_text(json.dumps({"text": "maru plan ready chhe"}) + "\n", encoding="utf-8")

    summ = process_jsonl_batch(in_p, out_p)
    assert summ.n_rows_in == 1
    assert summ.n_rows_out == 1
    assert summ.n_errors == 0

    rec = json.loads(out_p.read_text(encoding="utf-8").strip())
    assert "codemix" in rec
    assert "cmi" in rec
    assert "switch_points" in rec
    assert "dialect" in rec
    assert "મારું" in rec["codemix"]


def test_process_csv_batch_adds_columns(tmp_path) -> None:
    in_p = tmp_path / "in.csv"
    out_p = tmp_path / "out.csv"
    with in_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text"])
        w.writeheader()
        w.writerow({"text": "maru plan ready chhe"})

    summ = process_csv_batch(in_p, out_p)
    assert summ.n_rows_in == 1
    assert summ.n_rows_out == 1
    assert summ.n_errors == 0

    rows = list(csv.DictReader(out_p.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert "codemix" in rows[0]
    assert "cmi" in rows[0]
    assert "switch_points" in rows[0]
    assert "dialect" in rows[0]
    assert "મારું" in (rows[0]["codemix"] or "")

