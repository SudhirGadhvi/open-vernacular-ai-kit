from __future__ import annotations

import os
from typing import Any

import streamlit as st

from gujarati_codemix_kit.codemix_render import analyze_codemix


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


def _sarvam_chat(prompt: str, *, api_key: str, temperature: float) -> str:
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    resp = client.chat.completions(
        model="sarvam-m",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=1,
        max_tokens=800,
    )
    return resp.choices[0].message.content


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
    return str(resp)


def main() -> None:
    _try_load_dotenv()

    st.set_page_config(page_title="Gujarati CodeMix Kit Demo", layout="wide")
    st.title("Gujarati CodeMix Kit")
    st.caption("Normalize Gujarati-English code-mixed text (incl. romanized Gujarati / Gujlish).")

    with st.sidebar:
        st.subheader("Options")
        topk = st.number_input("Transliteration top-k", min_value=1, max_value=5, value=1, step=1)
        numerals = st.selectbox("Numerals", options=["keep", "ascii"], index=0)
        st.divider()
        st.subheader("Sarvam (optional)")
        sarvam_key = st.text_input(
            "SARVAM_API_KEY",
            value=os.environ.get("SARVAM_API_KEY", ""),
            type="password",
            help="If set, demo can call Sarvam-M and Mayura. Do not commit keys to git.",
        )
        sarvam_enabled = st.checkbox(
            "Enable Sarvam calls", value=False, disabled=not bool(sarvam_key) or not _sarvam_available()
        )
        temperature = st.slider("Sarvam-M temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.1)

        if sarvam_enabled and not _sarvam_available():
            st.warning("Install Sarvam SDK: `pip install -e '.[sarvam]'`")

    default = "maru business plan ready chhe!!!\n\nમારું business plan ready છે!!!"
    raw = st.text_area("Input", value=default, height=180)

    a = analyze_codemix(raw, topk=int(topk), numerals=numerals)
    norm = a.normalized
    codemixed = a.codemix

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Raw")
        st.code(raw)
    with c2:
        st.subheader("Normalized")
        st.code(norm)
    with c3:
        st.subheader("CodeMix Rendered")
        st.code(codemixed)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gujlish tokens", a.n_gu_roman_tokens)
    m2.metric("Transliterated", a.n_gu_roman_transliterated)
    m3.metric("Conversion rate", f"{a.pct_gu_roman_transliterated * 100:.1f}%")
    m4.metric("Backend", a.transliteration_backend)

    st.divider()

    if not sarvam_enabled:
        st.info("Enable Sarvam calls in the sidebar to see end-to-end impact.")
        return

    if not sarvam_key:
        st.error("Missing SARVAM_API_KEY")
        return

    prompt = st.text_area("Prompt to Sarvam-M", value=codemixed, height=120)
    run = st.button("Run Sarvam-M", type="primary")

    if run:
        with st.spinner("Calling Sarvam-M..."):
            try:
                answer = _sarvam_chat(prompt, api_key=sarvam_key, temperature=float(temperature))
            except Exception as e:
                st.error(f"Sarvam-M call failed: {e}")
                return
        st.subheader("Sarvam-M response")
        st.write(answer)

        with st.spinner("Calling Mayura translate (to English)..."):
            try:
                tr = _sarvam_translate(codemixed, api_key=sarvam_key)
            except Exception as e:
                st.warning(f"Translate failed: {e}")
            else:
                st.subheader("Mayura translation (auto -> en-IN)")
                st.write(_extract_translate_output(tr))


if __name__ == "__main__":
    main()

