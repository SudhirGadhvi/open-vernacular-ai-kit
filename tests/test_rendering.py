from __future__ import annotations

from gujarati_codemix_kit.rendering import render_tokens


def test_render_tokens_basic_punct_spacing() -> None:
    assert render_tokens(["Hello", ",", "world", "!"]) == "Hello, world!"
    assert render_tokens(["hi", ".", "ok", "?"]) == "hi. ok?"


def test_render_tokens_parentheses_and_apostrophe() -> None:
    assert render_tokens(["(", "test", ")"]) == "(test)"
    assert render_tokens(["don", "'", "t"]) == "don't"


def test_render_tokens_joiners() -> None:
    assert render_tokens(["test", "@", "abc", ".", "com"]) == "test@abc.com"
    assert render_tokens(["maru", "-", "business"]) == "maru-business"


def test_render_tokens_danda() -> None:
    # Gujarati danda should behave like sentence punctuation.
    assert render_tokens(["હું", "।", "તું"]) == "હું। તું"

