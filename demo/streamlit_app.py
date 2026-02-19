from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from gujarati_codemix_kit import __version__ as gck_version
from gujarati_codemix_kit.codemix_render import analyze_codemix
from gujarati_codemix_kit.normalize import normalize_text
from gujarati_codemix_kit.token_lid import TokenLang, tag_tokens, tokenize
from gujarati_codemix_kit.transliterate import translit_gu_roman_to_native_configured


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _inject_css() -> None:
    # "Dribbble-ish" dark AI landing: gradient hero, glass cards, less Streamlit chrome.
    st.markdown(
        """
<style>
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }
div[data-testid="stDecoration"] { visibility: hidden; height: 0px; }

.main .block-container {
  max-width: 1140px;
  padding-top: 1.7rem;
  padding-bottom: 3rem;
}

.gck-hero {
  border-radius: 26px;
  padding: 1.8rem 1.8rem;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background:
    radial-gradient(900px circle at 12% -10%, rgba(56, 189, 248, 0.22), transparent 45%),
    radial-gradient(700px circle at 95% 10%, rgba(168, 85, 247, 0.18), transparent 40%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(2, 6, 23, 0.96));
  box-shadow: 0 26px 90px rgba(0, 0, 0, 0.40);
}
.gck-pill {
  display: inline-flex;
  gap: 0.45rem;
  align-items: center;
  padding: 0.28rem 0.70rem;
  border-radius: 999px;
  border: 1px solid rgba(56, 189, 248, 0.28);
  background: rgba(56, 189, 248, 0.10);
  color: rgba(226, 232, 240, 0.92);
  font-size: 0.85rem;
}
.gck-hero h1 {
  margin: 0.55rem 0 0 0;
  font-size: 2.35rem;
  line-height: 1.06;
  letter-spacing: -0.02em;
}
.gck-hero p {
  margin: 0.80rem 0 0 0;
  color: rgba(226, 232, 240, 0.82);
  line-height: 1.55;
  font-size: 1.02rem;
}

.gck-card {
  border-radius: 20px;
  padding: 1.05rem 1.05rem;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(2, 6, 23, 0.38);
}
.gck-card h4 { margin: 0 0 0.45rem 0; font-size: 1.06rem; }
.gck-card p { margin: 0; color: rgba(226, 232, 240, 0.78); }

.gck-section-title {
  margin-top: 0.2rem;
  margin-bottom: 0.2rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _sarvam_available() -> bool:
    try:
        import sarvamai  # noqa: F401
    except Exception:
        return False
    return True


def _sarvam_chat(prompt: str, *, api_key: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    """
    Returns content + usage (if provided).

    sarvamai changed signature: some versions accept `model=`, newer ones don't.
    """
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    try:
        resp = client.chat.completions(
            model="sarvam-m",
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            top_p=1,
            max_tokens=int(max_tokens),
        )
    except TypeError:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            top_p=1,
            max_tokens=int(max_tokens),
        )

    usage = None
    try:
        if resp.usage is not None:
            usage = resp.usage.model_dump()
    except Exception:
        usage = None

    return {"content": resp.choices[0].message.content, "usage": usage, "model": getattr(resp, "model", None)}


def _sarvam_translate(text: str, *, api_key: str) -> Any:
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    return client.text.translate(
        input=text,
        source_language_code="auto",
        target_language_code="en-IN",
        model="mayura:v1",
    )


def _extract_translate_output(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("translated_text", "output", "translation", "translatedText"):
            v = resp.get(k)
            if isinstance(v, str) and v:
                return v
    for attr in ("translated_text", "output", "translation", "translatedText"):
        try:
            v = getattr(resp, attr)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v
    try:
        dump = resp.model_dump()  # type: ignore[attr-defined]
    except Exception:
        dump = None
    if isinstance(dump, dict):
        for k in ("translated_text", "output", "translation", "translatedText"):
            v = dump.get(k)
            if isinstance(v, str) and v:
                return v
    return str(resp)


def _examples() -> dict[str, str]:
    return {
        "Support: order update": "mare order update joie chhe... parcel kyare aavse??",
        "Business plan": "maru business plan ready chhe!!!",
        "Mixed Gujarati + English": "મારે tomorrow meeting છે, please confirm.",
        "Delivery / numbers": "kal 2 baje delivery moklo, bill 450 rupiya.",
        "Multi-line": "maru business plan ready chhe!!!\n\nમારું business plan ready છે!!!",
    }


def _transliteration_rows(normalized_text: str, *, topk: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    toks = tokenize(normalized_text or "")
    tagged = tag_tokens(toks)
    for tok in tagged:
        if tok.lang != TokenLang.GU_ROMAN:
            continue
        cands = translit_gu_roman_to_native_configured(
            tok.text,
            topk=topk,
            preserve_case=True,
            aggressive_normalize=False,
        )
        if not cands:
            continue
        best = cands[0]
        if best != tok.text:
            rows.append({"Gujlish token": tok.text, "Converted Gujarati": best})
    return rows


def _lid_counts(text: str) -> dict[str, int]:
    toks = tokenize(normalize_text(text or ""))
    tagged = tag_tokens(toks)
    out = {"gu_native": 0, "gu_roman": 0, "en": 0, "other": 0}
    for t in tagged:
        if t.lang == TokenLang.GU_NATIVE:
            out["gu_native"] += 1
        elif t.lang == TokenLang.GU_ROMAN:
            out["gu_roman"] += 1
        elif t.lang == TokenLang.EN:
            out["en"] += 1
        else:
            out["other"] += 1
    return out


def _token_set(text: str) -> set[str]:
    toks = tokenize(normalize_text(text or ""))
    out: set[str] = set()
    for t in toks:
        if not any(ch.isalnum() for ch in t):
            continue
        out.add(t.lower())
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _kb_corpus() -> list[dict[str, str]]:
    # Stored data should be stable; query changes after canonicalization.
    return [
        {"id": "kb1", "title": "Order delivery status", "text": "મારે order update જોઈએ છે. parcel ક્યારે આવશે?"},
        {"id": "kb2", "title": "Meeting confirmation", "text": "મારે tomorrow meeting છે, please confirm time."},
        {"id": "kb3", "title": "Business plan guidance", "text": "મારું business plan ready છે. Next steps માટે guidance જોઈએ છે."},
        {"id": "kb4", "title": "Invoice and billing", "text": "Bill amount confirm કરો. payment receipt મોકલો."},
    ]


def _best_match(query_tokens: set[str], corpus: list[dict[str, str]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for doc in corpus:
        doc_tokens = _token_set(doc["text"])
        score = _jaccard(query_tokens, doc_tokens)
        overlap = sorted(query_tokens & doc_tokens)
        cand = {
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "score": score,
            "overlap": overlap,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best or {"id": "", "title": "", "text": "", "score": 0.0, "overlap": []}


def _apply_template(tpl: str, text: str) -> str:
    if "{text}" not in tpl:
        return f"{tpl}\n\n{text}".strip()
    return tpl.replace("{text}", text)


def _cell(v: int | None) -> str:
    return "—" if v is None else str(int(v))


def _delta(b: int | None, a: int | None) -> str:
    if b is None or a is None:
        return "—"
    return f"{(int(a) - int(b)):+d}"


def main() -> None:
    _try_load_dotenv()
    st.set_page_config(page_title="Gujarati CodeMix Kit", layout="wide", initial_sidebar_state="collapsed")
    _inject_css()

    st.markdown(
        f"""
<div class="gck-hero">
  <div class="gck-pill">AI-ready Gujarati text <span style="opacity:.65">•</span> v{gck_version}</div>
  <h1>Gujarati CodeMix Kit</h1>
  <p>
    Preprocess Gujarati-English messages before they hit LLMs, search, routing, and analytics.
    We convert Gujlish (romanized Gujarati) into Gujarati script while keeping English as-is.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="gck-card">
  <h4>LLM quality</h4>
  <p>Less ambiguity: Gujlish becomes Gujarati script, improving intent understanding.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="gck-card">
  <h4>Search & retrieval</h4>
  <p>Canonical text matches stored tickets/KB more reliably (Gujarati script vs Latin).</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="gck-card">
  <h4>Routing & analytics</h4>
  <p>Cleaner language signals help queues, dashboards, and monitoring.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("Settings", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            topk = st.number_input("Transliteration top-k", min_value=1, max_value=5, value=1, step=1)
            numerals = st.selectbox("Numerals", options=["keep", "ascii"], index=0)
        with s2:
            sarvam_key = st.text_input("SARVAM_API_KEY (optional)", value=os.environ.get("SARVAM_API_KEY", ""), type="password")
            sarvam_enabled = st.checkbox(
                "Enable Sarvam-M comparison",
                value=False,
                disabled=not bool(sarvam_key) or not _sarvam_available(),
            )
            temperature = st.slider("Sarvam-M temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.1)
            max_out = st.number_input("Max output tokens (Sarvam-M)", min_value=32, max_value=800, value=256, step=16)

            if sarvam_enabled and not _sarvam_available():
                st.warning("Sarvam SDK not available. Install extras: `pip install -e '.[sarvam]'`.")

    topk = int(topk)
    max_out = int(max_out)

    st.markdown("## Live Demo", help="Paste a message, then click Analyze.")

    ex = _examples()
    ex_names = list(ex.keys())

    with st.form("gck_form", clear_on_submit=False):
        f1, f2 = st.columns([3, 1])
        with f1:
            chosen = st.selectbox("Example", options=ex_names, index=0)
        with f2:
            load = st.form_submit_button("Load example")

        if "gck_msg" not in st.session_state:
            st.session_state["gck_msg"] = ex[ex_names[0]]
        if load:
            st.session_state["gck_msg"] = ex[chosen]

        msg = st.text_area("Message", key="gck_msg", height=140)
        analyze_clicked = st.form_submit_button("Analyze", type="primary")

    # Compute only when the user clicks Analyze (or first load).
    if "gck_last_analysis" not in st.session_state or analyze_clicked:
        st.session_state["gck_last_analysis"] = analyze_codemix(msg, topk=topk, numerals=numerals)

    a = st.session_state["gck_last_analysis"]

    out1, out2 = st.columns(2)
    with out1:
        st.subheader("Before")
        st.caption("Raw user message (often Gujlish + English).")
        st.code(a.raw or "")
    with out2:
        st.subheader("After")
        st.caption("Canonical text for downstream systems.")
        st.code(a.codemix or "")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gujlish tokens", a.n_gu_roman_tokens)
    m2.metric("Converted", a.n_gu_roman_transliterated)
    m3.metric("Conversion rate", f"{a.pct_gu_roman_transliterated * 100:.1f}%")
    m4.metric("Backend", a.transliteration_backend)

    st.markdown("## What Changed")
    rows = _transliteration_rows(a.normalized, topk=topk)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No Gujlish conversions detected for this input.")

    st.divider()

    st.markdown("## Impact Without AI")
    st.caption("These are measurable improvements you can explain to a product team without talking about tokens.")

    impact_left, impact_right = st.columns(2)
    with impact_left:
        st.subheader("Search / retrieval (simulation)")
        st.caption("Stored KB/ticket text stays the same. Only the query changes after canonicalization.")
        corpus = _kb_corpus()
        q_before = _token_set(a.raw)
        q_after = _token_set(a.codemix)
        bm_before = _best_match(q_before, corpus)
        bm_after = _best_match(q_after, corpus)

        st.metric("Top match score (Before)", f"{bm_before['score']*100:.1f}")
        st.metric("Top match score (After)", f"{bm_after['score']*100:.1f}", delta=f"{(bm_after['score']-bm_before['score'])*100:+.1f}")

        st.markdown("**Query tokens (Before)**")
        st.code(" ".join(sorted(q_before)) if q_before else "(empty)")
        st.markdown("**Query tokens (After)**")
        st.code(" ".join(sorted(q_after)) if q_after else "(empty)")

        st.markdown("**Overlap tokens (Before)**")
        st.code(", ".join(bm_before["overlap"]) if bm_before["overlap"] else "(none)")
        st.markdown("**Overlap tokens (After)**")
        st.code(", ".join(bm_after["overlap"]) if bm_after["overlap"] else "(none)")

        with st.expander("Stored KB/ticket example (unchanged)"):
            st.write(bm_after["title"] or bm_before["title"])
            st.code(bm_after["text"] or bm_before["text"])

    with impact_right:
        st.subheader("Routing / analytics signal")
        st.caption("Token-level language mix becomes cleaner after Gujlish conversion.")
        c_before = _lid_counts(a.raw)
        c_after = _lid_counts(a.codemix)
        st.dataframe(
            [
                {"Lang": "gu_native", "Before": c_before["gu_native"], "After": c_after["gu_native"], "Delta": c_after["gu_native"] - c_before["gu_native"]},
                {"Lang": "gu_roman", "Before": c_before["gu_roman"], "After": c_after["gu_roman"], "Delta": c_after["gu_roman"] - c_before["gu_roman"]},
                {"Lang": "en", "Before": c_before["en"], "After": c_after["en"], "Delta": c_after["en"] - c_before["en"]},
                {"Lang": "other", "Before": c_before["other"], "After": c_after["other"], "Delta": c_after["other"] - c_before["other"]},
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()

    st.markdown("## Optional: AI Comparison (Sarvam-M)")
    st.caption("Use this to demonstrate answer quality and stability. Token savings is not the primary promise.")

    prompt_template = st.text_area(
        "Prompt template (use {text})",
        value="Please respond to the message below:\n\n{text}",
        height=90,
    )

    if not sarvam_enabled:
        st.info("Enable Sarvam-M comparison in Settings to run this section.")
        compare: dict[str, Any] | None = None
    elif not sarvam_key:
        st.warning("Missing SARVAM_API_KEY.")
        compare = None
    elif not _sarvam_available():
        st.warning("Sarvam SDK is not installed/available.")
        compare = None
    else:
        run = st.button("Run Sarvam-M", type="primary")
        if run:
            p_before = _apply_template(prompt_template, a.raw)
            p_after = _apply_template(prompt_template, a.codemix)

            with st.spinner("Calling Sarvam-M (Before)..."):
                out_before = _sarvam_chat(p_before, api_key=sarvam_key, temperature=float(temperature), max_tokens=max_out)
            with st.spinner("Calling Sarvam-M (After)..."):
                out_after = _sarvam_chat(p_after, api_key=sarvam_key, temperature=float(temperature), max_tokens=max_out)

            with st.spinner("Calling Mayura translate (After -> en-IN)..."):
                try:
                    tr = _sarvam_translate(a.codemix, api_key=sarvam_key)
                    mayura_text = _extract_translate_output(tr)
                except Exception:
                    mayura_text = ""

            st.session_state["gck_compare"] = {
                "created_at": _utc_now_iso(),
                "prompt_before": p_before,
                "prompt_after": p_after,
                "usage_before": out_before.get("usage"),
                "usage_after": out_after.get("usage"),
                "answer_before": out_before.get("content", ""),
                "answer_after": out_after.get("content", ""),
                "mayura_translation_en_in": mayura_text,
            }

        compare = st.session_state.get("gck_compare")

    if compare:
        u1 = compare.get("usage_before") if isinstance(compare.get("usage_before"), dict) else None
        u2 = compare.get("usage_after") if isinstance(compare.get("usage_after"), dict) else None

        pt_b = int(u1.get("prompt_tokens", 0)) if u1 else None
        pt_a = int(u2.get("prompt_tokens", 0)) if u2 else None
        ct_b = int(u1.get("completion_tokens", 0)) if u1 else None
        ct_a = int(u2.get("completion_tokens", 0)) if u2 else None
        tt_b = int(u1.get("total_tokens", 0)) if u1 else None
        tt_a = int(u2.get("total_tokens", 0)) if u2 else None

        st.subheader("Token usage (Sarvam)")
        st.dataframe(
            [
                {"Metric": "Prompt tokens", "Before": _cell(pt_b), "After": _cell(pt_a), "Delta": _delta(pt_b, pt_a)},
                {"Metric": "Completion tokens", "Before": _cell(ct_b), "After": _cell(ct_a), "Delta": _delta(ct_b, ct_a)},
                {"Metric": "Total tokens", "Before": _cell(tt_b), "After": _cell(tt_a), "Delta": _delta(tt_b, tt_a)},
            ],
            width="stretch",
            hide_index=True,
        )

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Before answer")
            with st.expander("Prompt", expanded=False):
                st.code(compare["prompt_before"])
            st.write(compare["answer_before"])
        with r2:
            st.subheader("After answer")
            with st.expander("Prompt", expanded=False):
                st.code(compare["prompt_after"])
            st.write(compare["answer_after"])

        if compare.get("mayura_translation_en_in"):
            st.subheader("Mayura translation (After → en-IN)")
            st.write(compare["mayura_translation_en_in"])

    st.divider()

    st.markdown("## Export")
    export_payload = {
        "created_at": _utc_now_iso(),
        "gck_version": gck_version,
        "analysis": asdict(a),
        "sarvam_compare": compare,
    }
    export_json = json.dumps(export_payload, ensure_ascii=False, indent=2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    st.download_button(
        "Download report (JSON)",
        data=export_json.encode("utf-8"),
        file_name=f"gck_demo_results_{ts}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

