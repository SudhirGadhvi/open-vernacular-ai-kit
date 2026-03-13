from __future__ import annotations

from open_vernacular_ai_kit.language_packs import get_language_pack


def test_language_profiles_load_from_packaged_data() -> None:
    gu = get_language_pack("gu")
    hi = get_language_pack("hi")

    assert gu.default_exceptions["tamaro"] == "તમારો"
    assert gu.default_exceptions["ma"] == "માં"
    assert "ma" in gu.context_roman_tokens

    assert hi.default_exceptions["mera"] == "मेरा"
    assert hi.default_exceptions["me"] == "में"
    assert "me" in hi.context_roman_tokens

    assert hi.default_exceptions["dijiye"] == "दीजिए"
    assert hi.default_exceptions["madad"] == "मदद"
    assert hi.default_exceptions["bhej"] == "भेज"
    assert hi.default_exceptions["lekin"] == "लेकिन"
    assert gu.default_exceptions["chhiye"] == "છીએ"
    assert gu.default_exceptions["aavo"] == "આવો"
    assert gu.default_exceptions["paisa"] == "પૈસા"
    assert gu.default_exceptions["avse"] == "આવશે"
    assert gu.default_exceptions["tamare"] == "તમારે"
    assert gu.default_exceptions["aa"] == "આ"
    assert gu.default_exceptions["fari"] == "ફરી"
    assert gu.default_exceptions["ni"] == "ની"
    assert gu.default_exceptions["nu"] == "નું"
    assert gu.default_exceptions["vage"] == "વાગે"
    assert "tamare" in gu.common_roman_tokens
    assert "aa" in gu.common_roman_tokens
    assert "fari" in gu.common_roman_tokens
    assert "ni" in gu.context_roman_tokens
    assert "nu" in gu.context_roman_tokens
    assert "lekin" in hi.common_roman_tokens
