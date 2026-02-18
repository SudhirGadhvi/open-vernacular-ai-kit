 # Gujarati CodeMix Kit
 
 `gujarati-codemix-kit` is a small SDK + CLI for cleaning up Gujarati-English code-mixed text,
 especially messy WhatsApp-style inputs where Gujarati might appear in:
 
 - Gujarati script (ગુજરાતી)
 - Romanized Gujarati (Gujlish)
 - Mixed script in the same sentence
 
 The goal is to normalize text *before* sending it to downstream models (Sarvam-M / Mayura /
 Sarvam-Translate), and to postprocess certain outputs (e.g., Saaras `codemix`).
 
 ## Install
 
 For full functionality (recommended):
 
 ```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[indic,ml,eval,demo,dev]"
 ```
 
 Minimal (CLI + basic heuristics only):
 
 ```bash
 pip install -e .
 ```
 
 ## CLI
 
 Normalize text:
 
 ```bash
 gck normalize "મારું business plan ready છે!!!"
 ```
 
 Render clean code-mix (Gujarati tokens in Gujarati script, English preserved):
 
 ```bash
 gck codemix "maru business plan ready chhe!!!"
 ```
 
Run eval (downloads public Gujlish eval CSVs into `~/.cache/gujarati-codemix-kit`):

```bash
gck eval --dataset gujlish --report eval/out/report.json
```

 ## Demo (Streamlit)
 
 ```bash
 streamlit run demo/streamlit_app.py
 ```
 
 If you export `SARVAM_API_KEY`, the demo can optionally call Sarvam APIs.
 
 ## Disclaimer
 
 This is an early MVP. Token-level language ID and romanized Gujarati handling is heuristic-first
 with an optional lightweight ML classifier.
 
