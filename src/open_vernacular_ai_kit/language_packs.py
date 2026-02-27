from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import regex as re


@dataclass(frozen=True)
class LanguagePack:
    code: str
    name: str
    native_script_re: re.Pattern
    common_roman_tokens: frozenset[str]
    roman_clusters: tuple[str, ...]
    default_exceptions: Mapping[str, str]
    ai4bharat_lang_code: Optional[str]
    sanscript_target: Optional[str]
    translit_candidate_keys: tuple[str, ...]
    terminal_virama: Optional[str]
    dialect_enabled: bool = False


DEFAULT_LANGUAGE = "gu"
_SUPPORTED_LANGUAGE_CODES: tuple[str, ...] = ("gu", "hi")


_GU_EXCEPTIONS: dict[str, str] = {
    "hu": "હું",
    "tu": "તું",
    "tame": "તમે",
    "ame": "અમે",
    "maru": "મારું",
    "mare": "મારે",
    "taro": "તારો",
    "tari": "તારી",
    "tamaro": "તમારો",
    "tamari": "તમારી",
    "chhe": "છે",
    "che": "છે",
    "nathi": "નથી",
    "hato": "હતો",
    "hati": "હતી",
    "hase": "હશે",
    "shu": "શું",
    "su": "શું",
    "kem": "કેમ",
    "kya": "ક્યાં",
    "kyare": "ક્યારે",
    "aaje": "આજે",
    "kaale": "કાલે",
    "naam": "નામ",
}


_HI_EXCEPTIONS: dict[str, str] = {
    "main": "मैं",
    "mai": "मैं",
    "mein": "में",
    "mera": "मेरा",
    "meri": "मेरी",
    "mere": "मेरे",
    "tum": "तुम",
    "aap": "आप",
    "hum": "हम",
    "kya": "क्या",
    "kaise": "कैसे",
    "hai": "है",
    "ho": "हो",
    "tha": "था",
    "thi": "थी",
    "the": "थे",
    "nahi": "नहीं",
    "haan": "हाँ",
    "aaj": "आज",
    "kal": "कल",
    "naam": "नाम",
    "namaste": "नमस्ते",
}


_LANGUAGE_PACKS: dict[str, LanguagePack] = {
    "gu": LanguagePack(
        code="gu",
        name="Gujarati",
        native_script_re=re.compile(r"[\p{Gujarati}]", flags=re.VERSION1),
        common_roman_tokens=frozenset(
            {
                "hu",
                "tu",
                "tame",
                "ame",
                "maru",
                "mare",
                "taro",
                "tari",
                "tamaro",
                "tamari",
                "chhe",
                "che",
                "nathi",
                "hato",
                "hati",
                "hase",
                "shu",
                "su",
                "kem",
                "kya",
                "kyare",
                "aaje",
                "kaale",
            }
        ),
        roman_clusters=(
            "aa",
            "ee",
            "oo",
            "ai",
            "au",
            "kh",
            "gh",
            "ch",
            "chh",
            "jh",
            "th",
            "dh",
            "ph",
            "bh",
            "sh",
            "gn",
        ),
        default_exceptions=_GU_EXCEPTIONS,
        ai4bharat_lang_code="gu",
        sanscript_target="GUJARATI",
        translit_candidate_keys=("gu", "guj", "gu-Gujr", "Gujarati"),
        terminal_virama="\u0acd",
        dialect_enabled=True,
    ),
    "hi": LanguagePack(
        code="hi",
        name="Hindi",
        native_script_re=re.compile(r"[\p{Devanagari}]", flags=re.VERSION1),
        common_roman_tokens=frozenset(
            {
                "main",
                "mai",
                "mein",
                "mera",
                "meri",
                "mere",
                "tum",
                "aap",
                "hum",
                "kya",
                "kaise",
                "hai",
                "ho",
                "tha",
                "thi",
                "the",
                "nahi",
                "haan",
                "aaj",
                "kal",
                "naam",
                "namaste",
            }
        ),
        roman_clusters=(
            "aa",
            "ee",
            "oo",
            "ai",
            "au",
            "kh",
            "gh",
            "ch",
            "chh",
            "jh",
            "th",
            "dh",
            "ph",
            "bh",
            "sh",
            "tr",
            "kr",
            "ri",
        ),
        default_exceptions=_HI_EXCEPTIONS,
        ai4bharat_lang_code="hi",
        sanscript_target="DEVANAGARI",
        translit_candidate_keys=("hi", "hin", "hi-Deva", "Hindi", "Devanagari"),
        terminal_virama="\u094d",
        dialect_enabled=False,
    ),
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "gu": "gu",
    "gujarati": "gu",
    "g" : "gu",
    "hi": "hi",
    "hindi": "hi",
    "h": "hi",
}


def normalize_language_code(code: Optional[str]) -> str:
    raw = str(code or "").strip().lower()
    if not raw:
        return DEFAULT_LANGUAGE
    return _LANGUAGE_ALIASES.get(raw, raw)


def is_supported_language(code: Optional[str]) -> bool:
    return normalize_language_code(code) in _LANGUAGE_PACKS


def supported_language_codes() -> tuple[str, ...]:
    return _SUPPORTED_LANGUAGE_CODES


def get_language_pack(code: Optional[str]) -> LanguagePack:
    key = normalize_language_code(code)
    return _LANGUAGE_PACKS.get(key, _LANGUAGE_PACKS[DEFAULT_LANGUAGE])
