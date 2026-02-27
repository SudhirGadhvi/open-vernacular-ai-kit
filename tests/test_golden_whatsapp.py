from __future__ import annotations

import pytest

from open_vernacular_ai_kit.codemix_render import render_codemix

_CASES: list[tuple[str, str]] = [
    ("maru plan ready chhe!!!", "મારું plan ready છે!!"),
    ("hu aaje office ma plan ready chhe!!", "હું આજે office ma plan ready છે!!"),
    ("tame aaje ok chhe?", "તમે આજે ok છે?"),
    ("shu che???", "શું છે??"),
    ("maru naam Sudhir chhe.", "મારું નામ Sudhir છે."),
    ("naam shu che?", "નામ શું છે?"),
    ("tame kem chhe?", "તમે કેમ છે?"),
    ("hu nathi ready.", "હું નથી ready."),
    ("hu hato office ma.", "હું હતો office ma."),
    ("hu hase ready!", "હું હશે ready!"),
    ("maru bill ₹500 chhe", "મારું bill ₹500 છે"),
    ("hu | tu", "હું। તું"),
    ("maru-business plan ready chhe", "મારું-business plan ready છે"),
    ("maru_plan ready chhe", "મારું_plan ready છે"),
    ("maru/plan ready chhe", "મારું/plan ready છે"),
    ("hu (maru) plan ready chhe", "હું (મારું) plan ready છે"),
    ("hu, tame aaje plan ready chhe", "હું, તમે આજે plan ready છે"),
    ("maru id test@abc.com chhe", "મારું id test@abc.com છે"),
    ("hu aaje ૧૦ mins ma ready chhe", "હું આજે ૧૦ mins ma ready છે"),
    ("tame aaje to ready chhe", "તમે આજે to ready છે"),
    ("hu aaje in office chhe", "હું આજે in office છે"),
    ("hu aaje plan... ready chhe", "હું આજે plan... ready છે"),
    ("hu aaje 🙂 ready chhe", "હું આજે 🙂 ready છે"),
    ("maru plan - ready chhe", "મારું plan-ready છે"),
    ("hu aaje note: maru plan ready chhe", "હું આજે note: મારું plan ready છે"),
]


@pytest.mark.parametrize("raw,expected", _CASES)
def test_golden_whatsapp_style_sentences(raw: str, expected: str) -> None:
    # Use sentence mode to cover contiguous Gujlish runs.
    assert render_codemix(raw, translit_mode="sentence") == expected

