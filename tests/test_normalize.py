from __future__ import annotations

from gujarati_codemix_kit.normalize import normalize_text


def test_normalize_basic_whitespace_and_punct() -> None:
    s = "Hello\u00a0world!!!   "
    assert normalize_text(s) == "Hello world!!"


def test_normalize_quotes_and_ellipsis() -> None:
    s = "“test”…  "
    assert normalize_text(s) == '"test"...'


def test_normalize_pipe_danda() -> None:
    s = "આ | તે"
    assert normalize_text(s) == "આ।તે"


def test_normalize_gujarati_digits_ascii() -> None:
    s = "મારો નંબર ૧૨૩ છે"
    assert normalize_text(s, numerals="ascii") == "મારો નંબર 123 છે"

