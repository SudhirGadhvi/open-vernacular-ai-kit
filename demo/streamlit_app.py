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


def _sarvam_available() -> bool:
    try:
        import sarvamai  # noqa: F401
    except Exception:
        return False
    return True


def _sarvam_chat(
    prompt: str, *, api_key: str, temperature: float, max_tokens: int = 256
) -> dict[str, Any]:
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    # sarvamai SDK signature changed: newer versions do not accept `model=`.
    # Keep compatibility by trying with `model` first and falling back.
    try:
        resp = client.chat.completions(
            model="sarvam-m",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1,
            max_tokens=int(max_tokens),
        )
    except TypeError:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1,
            max_tokens=int(max_tokens),
        )
    usage = None
    try:
        if resp.usage is not None:
            usage = resp.usage.model_dump()
    except Exception:
        usage = None
    return {
        "content": resp.choices[0].message.content,
        "usage": usage,
        "model": getattr(resp, "model", None),
    }


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
    # Keep defensive: SDK output shape can evolve.
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("translated_text", "output", "translation", "translatedText"):
            if k in resp and isinstance(resp[k], str):
                return resp[k]
    # Pydantic models (sarvamai returns typed objects).
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


def _example_inputs() -> dict[str, str]:
    # Keep these simple and client-friendly.
    return {
        "Business plan (Gujlish + English)": "maru business plan ready chhe!!!",
        "Mixed Gujarati + English": "મારે tomorrow meeting છે, please confirm.",
        "Delivery / numbers": "kal 2 baje delivery moklo, bill 450 rupiya.",
        "Support (WhatsApp style)": "mare order update joie chhe... parcel kyare aavse??",
        "Multi-line WhatsApp style": "maru business plan ready chhe!!!\n\nમારું business plan ready છે!!!",
    }


def _lid_counts(text: str) -> dict[str, int]:
    toks = tokenize(normalize_text(text or ""))
    tagged = tag_tokens(toks)
    out = {TokenLang.EN.value: 0, TokenLang.GU_NATIVE.value: 0, TokenLang.GU_ROMAN.value: 0, TokenLang.OTHER.value: 0}
    for t in tagged:
        out[t.lang.value] = out.get(t.lang.value, 0) + 1
    return out


def _token_set_for_search(text: str) -> set[str]:
    """
    Very small, explainable search-normalization:
    normalize -> tokenize -> keep only letter/number tokens -> lowercase.
    """
    toks = tokenize(normalize_text(text or ""))
    out: set[str] = set()
    for t in toks:
        # Keep words/numbers only (drop punctuation).
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


def _demo_corpus() -> list[dict[str, str]]:
    """
    Mini "support desk / product" corpus that looks like what an app stores:
    canonical Gujarati script for Gujarati words, English preserved.
    """
    return [
        {
            "title": "Business plan readiness",
            "text": "મારું business plan ready છે. Next steps માટે guidance જોઈએ છે.",
        },
        {
            "title": "Delivery status",
            "text": "મારે order update જોઈએ છે. parcel ક્યારે આવશે?",
        },
        {
            "title": "Meeting confirmation",
            "text": "મારે tomorrow meeting છે, please confirm time.",
        },
        {
            "title": "Invoice and billing",
            "text": "Bill amount confirm કરો. payment receipt મોકલો.",
        },
    ]


def _transliteration_rows(normalized_text: str, *, topk: int) -> list[dict[str, str]]:
    """
    Build a simple, explainable table for clients:
    Gujlish token -> Gujarati output (best candidate).
    """
    rows: list[dict[str, str]] = []
    try:
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
                rows.append({"Gujlish (romanized Gujarati)": tok.text, "Gujarati (converted)": best})
    except Exception:
        # If optional backends aren't available, keep the UI resilient.
        return []
    return rows


def _inject_ai_ui_css() -> None:
    # Streamlit doesn't provide a stable "hide chrome" API; CSS is best-effort.
    st.markdown(
        """
<style>
/* Hide Streamlit chrome for a "real webpage" feel */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }
div[data-testid="stDecoration"] { visibility: hidden; height: 0px; }
div[data-testid="stStatusWidget"] { visibility: hidden; }

/* Layout */
.main .block-container {
  padding-top: 2.0rem;
  padding-bottom: 3.0rem;
  max-width: 1180px;
}

/* Hero */
.gck-hero {
  border-radius: 28px;
  padding: 2.2rem 2.2rem;
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.40);
  background:
    radial-gradient(1200px circle at 12% 0%, rgba(56, 189, 248, 0.22), transparent 45%),
    radial-gradient(900px circle at 95% 10%, rgba(168, 85, 247, 0.16), transparent 40%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(2, 6, 23, 0.96));
}
.gck-hero-title {
  margin: 0;
  font-size: 2.55rem;
  line-height: 1.06;
  letter-spacing: -0.02em;
}
.gck-hero-subtitle {
  margin-top: 0.8rem;
  color: rgba(226, 232, 240, 0.82);
  font-size: 1.05rem;
  line-height: 1.5;
}
.gck-badge {
  display: inline-block;
  padding: 0.30rem 0.65rem;
  border-radius: 999px;
  border: 1px solid rgba(56, 189, 248, 0.28);
  background: rgba(56, 189, 248, 0.10);
  color: rgba(226, 232, 240, 0.92);
  font-size: 0.90rem;
}

/* Cards */
.gck-card {
  border-radius: 20px;
  padding: 1.1rem 1.1rem;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(2, 6, 23, 0.35);
}
.gck-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.05rem;
}
.gck-card p {
  margin: 0;
  color: rgba(226, 232, 240, 0.78);
}

/* Tighten selectbox label spacing a bit */
label { color: rgba(226, 232, 240, 0.88) !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    _try_load_dotenv()

    st.set_page_config(
        page_title="Gujarati CodeMix Kit",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None, "About": None},
    )
    _inject_ai_ui_css()

    st.markdown(
        f"""
<div class="gck-hero">
  <div class="gck-badge">AI-ready Gujarati text • v{gck_version}</div>
  <h1 class="gck-hero-title">Gujarati CodeMix Kit</h1>
  <p class="gck-hero-subtitle">
    Convert messy Gujarati-English messages into a stable format for LLMs, search, and analytics.
    Gujarati stays in Gujarati script. English stays in Latin. Gujlish gets converted when possible.
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
  <h4>Problem</h4>
  <p>Gujarati typed in Latin (Gujlish) is hard for models and search to interpret consistently.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="gck-card">
  <h4>Solution</h4>
  <p>Normalize + transliterate Gujarati tokens to Gujarati script while preserving English.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="gck-card">
  <h4>Outcome</h4>
  <p>Cleaner prompts, better retrieval, and more stable downstream behavior.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    # Defaults (overridable in Settings).
    topk = 1
    numerals = "keep"
    sarvam_key = os.environ.get("SARVAM_API_KEY", "")
    sarvam_enabled = False
    temperature = 0.2
    sarvam_max_tokens = 256

    with st.expander("Settings", expanded=False):
        st.caption(
            "Tip: This demo is styled like a webpage, so the Streamlit sidebar toggle may be hidden. "
            "All settings are available here."
        )

        s_left, s_right = st.columns(2)
        with s_left:
            st.markdown("**Text processing**")
            topk = st.number_input("Transliteration top-k", min_value=1, max_value=5, value=int(topk), step=1)
            numerals = st.selectbox("Numerals", options=["keep", "ascii"], index=0)

        with s_right:
            st.markdown("**AI comparison (optional)**")
            sarvam_key = st.text_input(
                "SARVAM_API_KEY",
                value=sarvam_key,
                key="sarvam_api_key",
                type="password",
                help="Stored only in your browser session. Do not commit keys to git.",
            )
            sarvam_enabled = st.checkbox(
                "Enable Sarvam calls",
                value=False,
                disabled=not bool(sarvam_key) or not _sarvam_available(),
                help="Requires SARVAM_API_KEY and the Sarvam SDK.",
            )
            temperature = st.slider(
                "Sarvam-M temperature", min_value=0.0, max_value=1.2, value=float(temperature), step=0.1
            )
            sarvam_max_tokens = st.number_input(
                "Max output tokens (Sarvam-M)",
                min_value=32,
                max_value=800,
                value=int(sarvam_max_tokens),
                step=16,
                help="Caps response length (controls completion tokens). Useful for fair cost comparisons.",
            )

            if not _sarvam_available():
                st.warning("Sarvam SDK not available (install extras: `pip install -e '.[sarvam]'`).")
            elif not sarvam_key:
                st.info("Add `SARVAM_API_KEY` to enable AI comparison.")

    topk = int(topk)
    temperature = float(temperature)
    sarvam_max_tokens = int(sarvam_max_tokens)

    examples = _example_inputs()
    example_names = list(examples.keys())
    tab_try, tab_value, tab_how, tab_ai, tab_export = st.tabs(
        ["Try it", "Business value", "How it works", "AI impact", "Export"]
    )

    with tab_try:
        cex1, cex2 = st.columns([3, 1])
        with cex1:
            selected_example = st.selectbox("Try an example", options=example_names, index=0)
        with cex2:
            load = st.button("Load example")

        if "client_input" not in st.session_state:
            st.session_state["client_input"] = examples[example_names[0]]
        if load:
            st.session_state["client_input"] = examples[selected_example]

        raw = st.text_area("Message", key="client_input", height=160)

        a = analyze_codemix(raw, topk=int(topk), numerals=numerals)
        norm = a.normalized
        codemixed = a.codemix

        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Before")
            st.caption("What the user wrote (often Gujlish + English).")
            st.code(raw or "")
        with b2:
            st.subheader("After")
            st.caption("Stable text for LLMs/search (Gujarati in Gujarati script, English preserved).")
            st.code(codemixed or "")

        with st.expander("Behind the scenes (normalized text)", expanded=False):
            if (raw or "") == (norm or ""):
                st.caption("No visible normalization changes for this input (it is already clean).")
            else:
                st.caption("This is an internal cleanup step before transliteration.")
            st.code(norm or "")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gujlish tokens", a.n_gu_roman_tokens)
        m2.metric("Converted", a.n_gu_roman_transliterated)
        m3.metric("Conversion rate", f"{a.pct_gu_roman_transliterated * 100:.1f}%")
        m4.metric("Backend", a.transliteration_backend)

        st.subheader("What changed?")
        rows = _transliteration_rows(norm, topk=int(topk))
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info(
                "No Gujlish-to-Gujarati conversions detected in this input (or transliteration backend unavailable)."
            )

        # Keep these around for other tabs.
        st.session_state["_last_analysis"] = a
        st.session_state["_last_norm"] = norm
        st.session_state["_last_codemix"] = codemixed

    with tab_value:
        st.subheader("Why teams buy this")
        st.write(
            "This SDK is a **preprocessor**: it makes Gujarati-English messages consistent before they hit "
            "LLMs, search, analytics, routing, or any downstream NLP."
        )

        a = st.session_state.get("_last_analysis") or analyze_codemix("", topk=int(topk), numerals=numerals)
        raw = a.raw or ""
        after = a.codemix or ""

        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(
                """
<div class="gck-card">
  <h4>Better AI responses</h4>
  <p>Less ambiguity: Gujlish becomes Gujarati script, so the model understands intent faster.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        with v2:
            st.markdown(
                """
<div class="gck-card">
  <h4>Better search & retrieval</h4>
  <p>Canonical text improves matching against stored tickets/KB content that is in Gujarati script.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        with v3:
            st.markdown(
                """
<div class="gck-card">
  <h4>Better routing & analytics</h4>
  <p>Cleaner language signals enable language-based routing, tagging, and reporting.</p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.divider()

        st.subheader("Search impact (toy example)")
        st.caption(
            "Simulates support-desk search: matching a user query against stored canonical messages. "
            "After cleanup, more tokens align with stored Gujarati script."
        )

        corpus = _demo_corpus()
        q_before = _token_set_for_search(raw)
        q_after = _token_set_for_search(after)

        rows_before: list[dict[str, Any]] = []
        rows_after: list[dict[str, Any]] = []
        for doc in corpus:
            dset = _token_set_for_search(doc["text"])
            rows_before.append(
                {"Title": doc["title"], "Score": round(_jaccard(q_before, dset) * 100, 1), "Text": doc["text"]}
            )
            rows_after.append(
                {"Title": doc["title"], "Score": round(_jaccard(q_after, dset) * 100, 1), "Text": doc["text"]}
            )

        rows_before.sort(key=lambda r: r["Score"], reverse=True)
        rows_after.sort(key=lambda r: r["Score"], reverse=True)

        sb1, sb2 = st.columns(2)
        with sb1:
            st.markdown("**Search results using Before query**")
            st.dataframe(rows_before[:3], width="stretch", hide_index=True)
        with sb2:
            st.markdown("**Search results using After query**")
            st.dataframe(rows_after[:3], width="stretch", hide_index=True)

        st.divider()

        st.subheader("Routing/analytics signal")
        st.caption("Token-level language ID becomes more confident after Gujlish is converted.")
        c_before = _lid_counts(raw)
        c_after = _lid_counts(after)
        lang_rows = [
            {"Lang": "gu_native", "Before": c_before.get("gu_native", 0), "After": c_after.get("gu_native", 0)},
            {"Lang": "gu_roman", "Before": c_before.get("gu_roman", 0), "After": c_after.get("gu_roman", 0)},
            {"Lang": "en", "Before": c_before.get("en", 0), "After": c_after.get("en", 0)},
            {"Lang": "other", "Before": c_before.get("other", 0), "After": c_after.get("other", 0)},
        ]
        st.dataframe(lang_rows, width="stretch", hide_index=True)

    with tab_how:
        st.subheader("What the kit does")
        st.write(
            "The canonical output is a code-mix representation: Gujarati stays Gujarati script; "
            "English stays Latin; romanized Gujarati (Gujlish) tokens are converted when possible."
        )
        st.subheader("Drop-in SDK usage")
        st.code(
            """from gujarati_codemix_kit import analyze_codemix

incoming = "maru business plan ready chhe!!!"
a = analyze_codemix(incoming)

# Use this for LLM/search indexing:
clean_text = a.codemix

# Use these for dashboards/QA:
stats = {
  "gujlish_tokens": a.n_gu_roman_tokens,
  "converted": a.n_gu_roman_transliterated,
  "conversion_rate": a.pct_gu_roman_transliterated,
  "backend": a.transliteration_backend,
}""",
            language="python",
        )
        st.write(
            "This reduces ambiguity for downstream systems (LLMs, embeddings, search indexing, analytics) "
            "because Gujarati words are represented consistently."
        )
        st.subheader("Where it helps")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(
                """
<div class="gck-card">
  <h4>Customer support</h4>
  <p>Cleaner tickets and chat history improve routing and answer quality.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                """
<div class="gck-card">
  <h4>Search & retrieval</h4>
  <p>Better matching for Gujarati content that was typed in Latin.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                """
<div class="gck-card">
  <h4>Analytics</h4>
  <p>More stable tokenization and language segmentation for reporting.</p>
</div>
""",
                unsafe_allow_html=True,
            )

    with tab_ai:
        st.subheader("AI impact (optional)")
        st.caption('Compare model output on raw vs "After" text. Requires `SARVAM_API_KEY`.')

        # Pull latest state from Try tab (works even if user switches tabs).
        a = st.session_state.get("_last_analysis") or analyze_codemix("", topk=int(topk), numerals=numerals)
        raw = a.raw
        codemixed = a.codemix

        prompt_template = st.text_area(
            "Prompt template",
            value="Please respond to the message below:\n\n{text}",
            height=100,
            help="Use {text} as a placeholder for either the raw or After text.",
        )

        def _apply_template(tpl: str, text: str) -> str:
            if "{text}" not in tpl:
                return f"{tpl}\n\n{text}".strip()
            return tpl.replace("{text}", text)

        prompt_raw = _apply_template(prompt_template, raw)
        prompt_codemix = _apply_template(prompt_template, codemixed)

        compare_key = json.dumps(
            {
                "raw": raw,
                "codemix": codemixed,
                "topk": int(topk),
                "numerals": numerals,
                "prompt_template": prompt_template,
                "temperature": float(temperature),
                "sarvam_max_tokens": int(sarvam_max_tokens),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        if st.session_state.get("sarvam_compare_key") != compare_key:
            st.session_state.pop("sarvam_compare", None)
            st.session_state["sarvam_compare_key"] = compare_key

        can_run_sarvam = bool(sarvam_enabled) and bool(sarvam_key) and _sarvam_available()
        run = st.button("Run Sarvam-M comparison", type="primary", disabled=not can_run_sarvam)

        if run:
            if not can_run_sarvam:
                st.error("Enable Sarvam calls and provide SARVAM_API_KEY.")
            else:
                with st.spinner("Calling Sarvam-M (Before)..."):
                    try:
                        answer_raw = _sarvam_chat(
                            prompt_raw,
                            api_key=sarvam_key,
                            temperature=float(temperature),
                            max_tokens=int(sarvam_max_tokens),
                        )
                    except Exception as e:
                        st.error(f"Sarvam-M (Before) call failed: {e}")
                        return

                with st.spinner("Calling Sarvam-M (After)..."):
                    try:
                        answer_codemix = _sarvam_chat(
                            prompt_codemix,
                            api_key=sarvam_key,
                            temperature=float(temperature),
                            max_tokens=int(sarvam_max_tokens),
                        )
                    except Exception as e:
                        st.error(f"Sarvam-M (After) call failed: {e}")
                        return

                mayura_out: str | None = None
                with st.spinner("Calling Mayura translate (After text -> en-IN)..."):
                    try:
                        tr = _sarvam_translate(codemixed, api_key=sarvam_key)
                    except Exception as e:
                        st.warning(f"Translate failed: {e}")
                    else:
                        mayura_out = _extract_translate_output(tr)

                st.session_state["sarvam_compare"] = {
                    "created_at": _utc_now_iso(),
                    "model": answer_codemix.get("model") or answer_raw.get("model") or "sarvam-m",
                    "temperature": float(temperature),
                    "sarvam_max_tokens": int(sarvam_max_tokens),
                    "prompt_template": prompt_template,
                    "prompt_raw": prompt_raw,
                    "prompt_codemix": prompt_codemix,
                    "answer_raw": answer_raw.get("content", ""),
                    "answer_codemix": answer_codemix.get("content", ""),
                    "usage_raw": answer_raw.get("usage"),
                    "usage_codemix": answer_codemix.get("usage"),
                    "mayura_translation_en_in": mayura_out,
                }

        if not sarvam_enabled:
            st.info("Enable Sarvam calls in Settings to run the AI comparison.")
        elif not sarvam_key:
            st.warning("Missing SARVAM_API_KEY (set it in Settings).")
        elif sarvam_enabled and not _sarvam_available():
            st.warning("Sarvam SDK not available (install extras: `pip install -e '.[sarvam]'`).")

        compare = st.session_state.get("sarvam_compare")
        if compare:
            # Efficiency panel: one compact table for before/after.
            st.subheader("Efficiency (Before vs After)")
            st.caption(
                "Prompt tokens affect input cost. Completion tokens vary with answer length. "
                "Total tokens = prompt + completion."
            )

            def _cell(v: int | None) -> str:
                return "—" if v is None else str(int(v))

            def _delta(a: int | None, b: int | None) -> str:
                if a is None or b is None:
                    return "—"
                d = int(b) - int(a)
                return f"{d:+d}"

            p_raw = compare.get("prompt_raw") or ""
            p_after = compare.get("prompt_codemix") or ""

            u_raw = compare.get("usage_raw")
            u_after = compare.get("usage_codemix")
            if not isinstance(u_raw, dict):
                u_raw = None
            if not isinstance(u_after, dict):
                u_after = None

            pt_b = int(u_raw.get("prompt_tokens", 0)) if u_raw else None
            pt_a = int(u_after.get("prompt_tokens", 0)) if u_after else None
            ct_b = int(u_raw.get("completion_tokens", 0)) if u_raw else None
            ct_a = int(u_after.get("completion_tokens", 0)) if u_after else None
            tt_b = int(u_raw.get("total_tokens", 0)) if u_raw else None
            tt_a = int(u_after.get("total_tokens", 0)) if u_after else None

            rows = [
                {
                    "Metric": "Prompt chars",
                    "Before": _cell(len(p_raw)),
                    "After": _cell(len(p_after)),
                    "Delta": _delta(len(p_raw), len(p_after)),
                },
                {
                    "Metric": "Prompt words",
                    "Before": _cell(len(p_raw.split())),
                    "After": _cell(len(p_after.split())),
                    "Delta": _delta(len(p_raw.split()), len(p_after.split())),
                },
                {
                    "Metric": "Prompt tokens",
                    "Before": _cell(pt_b),
                    "After": _cell(pt_a),
                    "Delta": _delta(pt_b, pt_a),
                },
                {
                    "Metric": "Completion tokens",
                    "Before": _cell(ct_b),
                    "After": _cell(ct_a),
                    "Delta": _delta(ct_b, ct_a),
                },
                {
                    "Metric": "Total tokens",
                    "Before": _cell(tt_b),
                    "After": _cell(tt_a),
                    "Delta": _delta(tt_b, tt_a),
                },
            ]

            st.dataframe(rows, width="stretch", hide_index=True)
            if u_raw is None or u_after is None:
                st.caption("Token usage not provided by the API for this run (only chars/words shown).")

            left, right = st.columns(2)
            with left:
                st.subheader("Before: model sees raw")
                with st.expander("Prompt", expanded=False):
                    st.code(compare["prompt_raw"])
                st.write(compare["answer_raw"])
            with right:
                st.subheader("After: model sees cleaned text")
                with st.expander("Prompt", expanded=False):
                    st.code(compare["prompt_codemix"])
                st.write(compare["answer_codemix"])

            if compare.get("mayura_translation_en_in"):
                st.subheader("Mayura translation (After text, auto -> en-IN)")
                st.write(compare["mayura_translation_en_in"])
        else:
            st.caption("Run the comparison to populate the Before/After outputs.")

    with tab_export:
        st.subheader("Download a report")
        st.caption("Useful for sharing internally or attaching to a client/email thread.")

        a = st.session_state.get("_last_analysis") or analyze_codemix("", topk=int(topk), numerals=numerals)
        compare = st.session_state.get("sarvam_compare")
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

