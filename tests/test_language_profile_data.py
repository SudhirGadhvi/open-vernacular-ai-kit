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
    assert gu.default_exceptions["jagyae"] == "જગ્યાએ"
    assert gu.default_exceptions["sathe"] == "સાથે"
    assert gu.default_exceptions["ochhu"] == "ઓછું"
    assert gu.default_exceptions["etle"] == "એટલે"
    assert gu.default_exceptions["pela"] == "પહેલા"
    assert gu.default_exceptions["paise"] == "પૈસા"
    assert gu.default_exceptions["malyo"] == "મળ્યો"
    assert gu.default_exceptions["batave"] == "બતાવે"
    assert gu.default_exceptions["avsho"] == "આવશો"
    assert gu.default_exceptions["atki"] == "અટકી"
    assert gu.default_exceptions["moklo"] == "મોકલો"
    assert gu.default_exceptions["karto"] == "કરતો"
    assert gu.default_exceptions["karvu"] == "કરવું"
    assert gu.default_exceptions["kare"] == "કરે"
    assert "tamare" in gu.common_roman_tokens
    assert "aa" in gu.common_roman_tokens
    assert "fari" in gu.common_roman_tokens
    assert "jagyae" in gu.common_roman_tokens
    assert "ochhu" in gu.common_roman_tokens
    assert "paise" in gu.common_roman_tokens
    assert "moklo" in gu.common_roman_tokens
    assert "karto" in gu.common_roman_tokens
    assert "karvu" in gu.common_roman_tokens
    assert "kare" in gu.common_roman_tokens
    assert "ni" in gu.context_roman_tokens
    assert "nu" in gu.context_roman_tokens
    assert "sathe" in gu.context_roman_tokens
    assert "etle" in gu.context_roman_tokens
    assert "pela" in gu.context_roman_tokens
    assert "lekin" in hi.common_roman_tokens
    assert hi.default_exceptions["dobara"] == "दोबारा"
    assert hi.default_exceptions["abhi"] == "अभी"
    assert hi.default_exceptions["galat"] == "गलत"
    assert hi.default_exceptions["batayiye"] == "बताइए"
    assert hi.default_exceptions["kyon"] == "क्यों"
    assert hi.default_exceptions["bhejo"] == "भेजो"
    assert hi.default_exceptions["subah"] == "सुबह"
    assert hi.default_exceptions["mila"] == "मिला"
    assert hi.default_exceptions["niche"] == "नीचे"
    assert "galat" in hi.common_roman_tokens
    assert "batayiye" in hi.common_roman_tokens
    assert "kyon" in hi.common_roman_tokens
    assert "bhejo" in hi.common_roman_tokens
    assert "subah" in hi.common_roman_tokens
    assert "mila" in hi.common_roman_tokens
    assert "dobara" in hi.context_roman_tokens
    assert "abhi" in hi.context_roman_tokens
    assert "niche" in hi.context_roman_tokens
