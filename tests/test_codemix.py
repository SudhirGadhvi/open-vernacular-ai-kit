from __future__ import annotations

from gujarati_codemix_kit.codemix_render import render_codemix
from gujarati_codemix_kit.token_lid import TokenLang, detect_token_lang


def test_detect_token_lang_common_gujlish() -> None:
    assert detect_token_lang("maru") == TokenLang.GU_ROMAN
    assert detect_token_lang("chhe") == TokenLang.GU_ROMAN
    assert detect_token_lang("plan") in {TokenLang.EN, TokenLang.GU_ROMAN}  # ambiguous; keep loose


def test_render_codemix_preserves_english_and_gujarati() -> None:
    s = "મારું business plan ready છે!!!"
    out = render_codemix(s)
    assert "business" in out
    assert "plan" in out
    assert "મારું" in out


def test_render_codemix_transliterates_chhe() -> None:
    s = "ready chhe!!!"
    out = render_codemix(s)
    assert "છે" in out

