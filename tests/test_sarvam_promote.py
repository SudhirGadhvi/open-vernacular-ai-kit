from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_promote import (
    LanguageSentenceCaseRecord,
    infer_sentence_case_language,
    promote_sentence_cases_from_review,
)
from open_vernacular_ai_kit.sarvam_review import init_review_record
from open_vernacular_ai_kit.sarvam_teacher import mine_sarvam_teacher_candidate


def _candidate(text: str, language_hint: str, canonical: str):
    def fake_call(_: str) -> str:
        return json.dumps(
            {
                "language_hint": language_hint,
                "sarvam_native": canonical,
                "sarvam_canonical": canonical,
                "english_tokens_keep": [],
                "candidate_tokens": [],
                "notes": "",
            },
            ensure_ascii=False,
        )

    return mine_sarvam_teacher_candidate(text, language_hint=language_hint, call_model=fake_call)


def test_infer_sentence_case_language_handles_mixed_via_script() -> None:
    reviewed = init_review_record(
        _candidate("tamne refund kyare malse", "mixed", "તમને refund ક્યારે મળશે"),
        review_action="accept_sentence_case",
        reviewed_expected="તમને refund ક્યારે મળશે",
        prefer_meta_expected=False,
    )
    assert infer_sentence_case_language(reviewed) == "gu"


def test_promote_sentence_cases_adds_new_and_skips_same() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("meri maa ka naam kya hai", "hi", "मेरी माँ का नाम क्या है"),
            review_action="accept_sentence_case",
            reviewed_expected="मेरी माँ का नाम क्या है",
            prefer_meta_expected=False,
        ),
        init_review_record(
            _candidate("tamne aaje office ma aavu chhe", "gu", "તમને આજે office માં આવવું છે"),
            review_action="accept_sentence_case",
            reviewed_expected="તમને આજે office માં આવવું છે",
            prefer_meta_expected=False,
        ),
        init_review_record(
            _candidate("ignore me", "hi", "ignore me"),
            review_action="reject",
            reviewed_expected="ignore me",
            prefer_meta_expected=False,
        ),
    ]
    existing = [
        LanguageSentenceCaseRecord(
            language="hi",
            raw="meri maa ka naam kya hai",
            expected="मेरी माँ का नाम क्या है",
            source="existing",
        )
    ]
    merged, report = promote_sentence_cases_from_review(
        reviewed_rows, existing_rows=existing, require_pass=False
    )
    assert len(merged) == 2
    assert report["n_added"] == 1
    assert report["n_duplicates_same"] == 1
    assert report["n_duplicates_conflict"] == 0
    assert report["n_skipped_non_sentence"] == 1


def test_promote_sentence_cases_reports_conflicts() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("mara paisa kyare avse", "mixed", "મારા પૈસા ક્યારે આવશે"),
            review_action="accept_sentence_case",
            reviewed_expected="મારા પૈસા ક્યારે આવશે",
            prefer_meta_expected=False,
        )
    ]
    existing = [
        LanguageSentenceCaseRecord(
            language="gu",
            raw="mara paisa kyare avse",
            expected="મારા પૈસા ક્યારે આવસે",
            source="existing",
        )
    ]
    merged, report = promote_sentence_cases_from_review(
        reviewed_rows, existing_rows=existing, require_pass=False
    )
    assert len(merged) == 1
    assert report["n_added"] == 0
    assert report["n_duplicates_conflict"] == 1


def test_promote_sentence_cases_skips_validation_failures_by_default() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("tamne aaje office ma aavu chhe", "gu", "તમને આજે office માં આવવું છે"),
            review_action="accept_sentence_case",
            reviewed_expected="તમને આજે office માં આવવું છે",
            prefer_meta_expected=False,
        )
    ]
    merged, report = promote_sentence_cases_from_review(reviewed_rows, existing_rows=[])
    assert len(merged) == 0
    assert report["n_added"] == 0
    assert report["n_validation_failures"] == 1
