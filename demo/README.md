# Web Demo (Streamlit)

This folder contains a web demo for Gujarati CodeMix Kit.

The demo is designed for non-technical users too:

- Paste a message (WhatsApp-style mixed Gujarati + English, including Gujlish).
- See "Before vs After" (what a user wrote vs what we send to AI/search).
- Optionally compare Sarvam-M outputs (requires `SARVAM_API_KEY`).

## Run locally

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[demo,indic]"
streamlit run demo/streamlit_app.py
```

Optional (to enable AI comparison):

```bash
.venv/bin/pip install -e ".[sarvam]"
export SARVAM_API_KEY="..."
streamlit run demo/streamlit_app.py
```

## Host as a web page

Recommended: Streamlit Community Cloud or HuggingFace Spaces (Streamlit).

Key setup notes:

- Entry file: `demo/streamlit_app.py`
- No API key is required to demonstrate the core transformation.
- If you want live AI before/after, set `SARVAM_API_KEY` as a secret/environment variable.

