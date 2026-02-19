from __future__ import annotations

from gujarati_codemix_kit.config import CodeMixConfig
from gujarati_codemix_kit.doctor import collect_doctor_info
from gujarati_codemix_kit.pipeline import CodeMixPipeline
from gujarati_codemix_kit.token_lid import TokenLang, detect_token_lang, tokenize
from gujarati_codemix_kit.transliterate import translit_gu_roman_to_native_configured


def test_tokenize_splits_punct() -> None:
    assert tokenize("hello, world!") == ["hello", ",", "world", "!"]
    assert tokenize("hu।tu") == ["hu", "।", "tu"]


def test_lid_heuristics_basic() -> None:
    assert detect_token_lang("ગુજરાતી") == TokenLang.GU_NATIVE
    assert detect_token_lang("maru") == TokenLang.GU_ROMAN
    assert detect_token_lang("the") == TokenLang.EN
    assert detect_token_lang("123") == TokenLang.OTHER


def test_exception_dictionary_token_and_phrase() -> None:
    assert translit_gu_roman_to_native_configured("hu", topk=1) == ["હું"]
    assert translit_gu_roman_to_native_configured("hu aaje", topk=1) == ["હું આજે"]


def test_pipeline_event_hook_is_non_blocking_and_ordered() -> None:
    events: list[dict[str, object]] = []

    def hook(e: dict[str, object]) -> None:
        events.append(e)

    cfg = CodeMixConfig(translit_mode="sentence")
    out = CodeMixPipeline(config=cfg, on_event=hook).run("hu aaje office jaish!!").codemix
    assert "હું" in out
    assert "આજે" in out

    stages = [str(e.get("stage")) for e in events]
    assert stages[:3] == ["normalize", "tokenize", "lid"]
    assert "transliterate" in stages
    assert "render" in stages
    assert stages[-1] == "done"


def test_pipeline_preserve_numbers_config() -> None:
    cfg = CodeMixConfig(preserve_numbers=False)
    out = CodeMixPipeline(config=cfg).run("મારો નંબર ૧૨૩ છે").codemix
    assert "123" in out


def test_doctor_info_has_expected_keys() -> None:
    info = collect_doctor_info()
    assert "python" in info
    assert "platform" in info
    assert "features" in info
    assert "packages" in info

