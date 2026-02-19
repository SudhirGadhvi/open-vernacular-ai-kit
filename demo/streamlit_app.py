from __future__ import annotations

# ruff: noqa: E402
import hashlib
import inspect
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure the demo uses the local SDK code when run from the repo, even if an older
# version of `gujarati_codemix_kit` is installed elsewhere in the environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))
    # If Streamlit re-runs the script, `gujarati_codemix_kit` may already be in sys.modules
    # from an older import path. Drop it so we re-import from local `src/`.
    for name in list(sys.modules.keys()):
        if name == "gujarati_codemix_kit" or name.startswith("gujarati_codemix_kit."):
            sys.modules.pop(name, None)

from gujarati_codemix_kit import __version__ as gck_version
from gujarati_codemix_kit.app_flows import (
    clean_whatsapp_chat_text,
    process_csv_batch,
    process_jsonl_batch,
)
from gujarati_codemix_kit.codemix_render import analyze_codemix, render_codemix
from gujarati_codemix_kit.codeswitch import compute_code_switch_metrics
from gujarati_codemix_kit.dialects import detect_dialect_from_tagged_tokens
from gujarati_codemix_kit.lexicon import load_user_lexicon
from gujarati_codemix_kit.normalize import normalize_text
from gujarati_codemix_kit.token_lid import TokenLang, tag_tokens, tokenize
from gujarati_codemix_kit.transliterate import translit_gu_roman_to_native_configured

try:
    # v0.5: RAG helpers (optional UI section).
    from gujarati_codemix_kit import RagIndex, load_gujarat_facts_tiny, make_hf_embedder

    _RAG_AVAILABLE = True
except Exception:  # pragma: no cover
    _RAG_AVAILABLE = False
    RagIndex = None  # type: ignore[assignment]
    load_gujarat_facts_tiny = None  # type: ignore[assignment]
    make_hf_embedder = None  # type: ignore[assignment]


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


def _rag_keyword_embed(texts: list[str]) -> list[list[float]]:
    # Deterministic "embedding" for the demo when ML deps are not installed.
    #
    # It is intentionally small and only supports the tiny packaged dataset queries well.
    keys = [
        "અમદાવાદ",
        "રાજધાની",
        "ગાંધીનગર",
        "નવરાત્રી",
        "ગરબા",
        "શિયાળ",
        "ઉંધીયુ",
        "ગિર",
        "સિંહ",
        "ડાયમંડ",
        "સુરત",
    ]
    out: list[list[float]] = []
    for t in texts:
        s = t or ""
        out.append([1.0 if k in s else 0.0 for k in keys])
    return out


@st.cache_resource(show_spinner=False)
def _build_rag_index_cached(
    *,
    embedding_mode: str,
    hf_model_id_or_path: str,
    allow_remote_models: bool,
) -> object:
    """
    Cache the index across Streamlit reruns.

    Returns a `RagIndex` instance when available, else raises.
    """

    if not _RAG_AVAILABLE:
        raise RuntimeError("RAG utilities are not available in this environment.")
    ds = load_gujarat_facts_tiny()

    mode = (embedding_mode or "").strip().lower()
    if mode == "hf":
        mid = (hf_model_id_or_path or "").strip()
        if not mid:
            raise RuntimeError("Missing HF model id/path for embeddings.")
        embed = make_hf_embedder(  # type: ignore[misc]
            model_id_or_path=mid,
            allow_remote_models=bool(allow_remote_models),
        )
        idx = RagIndex.build(docs=ds.docs, embed_texts=embed, embedding_model=mid)  # type: ignore[union-attr]
        return idx

    # Default: keyword mode (no ML deps).
    idx = RagIndex.build(docs=ds.docs, embed_texts=_rag_keyword_embed, embedding_model="keywords")  # type: ignore[union-attr]
    return idx


def _rag_context_block(rag_payload: dict[str, Any], *, n_docs: int = 3) -> str:
    """
    Render a compact context block from the demo's stored RAG payload.
    """

    if not isinstance(rag_payload, dict):
        return ""
    rows = rag_payload.get("results")
    if not isinstance(rows, list) or not rows:
        return ""

    k = max(1, int(n_docs))
    lines: list[str] = []
    for r in rows[:k]:
        if not isinstance(r, dict):
            continue
        doc_id = str(r.get("doc_id", "") or "").strip()
        text = str(r.get("text", "") or "").strip()
        if not text:
            continue
        if doc_id:
            lines.append(f"- ({doc_id}) {text}")
        else:
            lines.append(f"- {text}")
    ctx = "\n".join(lines).strip()
    if not ctx:
        return ""

    q_used = str(rag_payload.get("query_used", "") or "").strip()
    q_used_block = f"\n\nQuestion:\n{q_used}" if q_used else ""
    return (
        "Use the context below to answer.\n\n"
        f"Context:\n{ctx}"
        f"{q_used_block}\n\n"
        "If the context is insufficient, say so briefly."
    ).strip()


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


def _jsonable(x: Any) -> Any:
    """
    Convert nested dataclasses/objects into JSON-serializable structures.
    """

    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _examples() -> dict[str, str]:
    return {
        "Support: order update": "mare order update joie chhe... parcel kyare aavse??",
        "Business plan": "maru business plan ready chhe!!!",
        "Mixed Gujarati + English": "મારે tomorrow meeting છે, please confirm.",
        "Delivery / numbers": "kal 2 baje delivery moklo, bill 450 rupiya.",
        "Multi-line": "maru business plan ready chhe!!!\n\nમારું business plan ready છે!!!",
    }


def _write_uploaded_file_to_tmp(*, filename: str, data: bytes) -> str:
    # Streamlit reruns the script often; keep a content-addressed copy on disk so we can
    # pass a stable `user_lexicon_path` to the SDK.
    h = hashlib.sha1(data).hexdigest()[:14]
    ext = Path(filename or "").suffix.lower() or ".bin"
    out_dir = Path(tempfile.gettempdir()) / "gck_demo_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{h}{ext}"
    p.write_bytes(data)
    return str(p)


def _transliteration_rows(
    normalized_text: str,
    *,
    topk: int,
    aggressive_normalize: bool,
    translit_backend: str,
    lexicon: dict[str, str] | None,
    lexicon_keys: set[str] | None,
    fasttext_model_path: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    toks = tokenize(normalized_text or "")
    tagged = tag_tokens(toks, lexicon_keys=lexicon_keys, fasttext_model_path=fasttext_model_path)
    for tok in tagged:
        if tok.lang != TokenLang.GU_ROMAN:
            continue
        cands = translit_gu_roman_to_native_configured(
            tok.text,
            topk=topk,
            preserve_case=True,
            aggressive_normalize=aggressive_normalize,
            exceptions=lexicon,
            backend=translit_backend,  # type: ignore[arg-type]
        )
        if not cands:
            continue
        best = cands[0]
        if best != tok.text:
            rows.append({"Gujlish token": tok.text, "Converted Gujarati": best})
    return rows


def _lid_counts(
    text: str, *, lexicon_keys: set[str] | None, fasttext_model_path: str | None
) -> dict[str, int]:
    toks = tokenize(normalize_text(text or ""))
    tagged = tag_tokens(toks, lexicon_keys=lexicon_keys, fasttext_model_path=fasttext_model_path)
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
            translit_mode = st.selectbox(
                "Transliteration mode",
                options=["sentence", "token"],
                index=0,
                help="Sentence mode unlocks phrase/joiner improvements for Gujlish runs.",
            )
            translit_backend = st.selectbox(
                "Transliteration backend",
                options=["auto", "ai4bharat", "sanscript", "none"],
                index=0,
                help="auto picks best available. ai4bharat requires optional install.",
            )
            aggressive_normalize = st.checkbox(
                "Aggressive Gujlish normalization",
                value=False,
                help="Try extra Gujlish spelling variants before transliteration.",
            )
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

        st.markdown("### Advanced (v0.3/v0.4.x)")
        a1, a2 = st.columns(2)
        with a1:
            lex_upload = st.file_uploader(
                "User lexicon (JSON/YAML) to force specific roman→Gujarati mappings",
                type=["json", "yaml", "yml"],
                help="Example JSON: {\"mane\": \"મને\", \"kyare\": \"ક્યારે\"}",
            )
        with a2:
            fasttext_model_path = st.text_input(
                "fastText model path (lid.176.ftz) for optional LID fallback",
                value=os.environ.get("GCK_FASTTEXT_MODEL_PATH", ""),
                help="Optional. If provided + fastText installed + file exists, it can help English detection.",
            )

        st.markdown("### Dialects (v0.4.x)")
        d1, d2 = st.columns(2)
        with d1:
            dialect_backend = st.selectbox(
                "Dialect backend",
                options=["auto", "heuristic", "transformers", "none"],
                index=0,
                help=(
                    "auto uses Transformers if a model is provided, else heuristic. "
                    "transformers expects a fine-tuned HF seq-classification model (path or id)."
                ),
            )
            dialect_normalize = st.checkbox(
                "Apply dialect normalization",
                value=False,
                help="Only applies when dialect confidence >= threshold. Never rewrites English tokens.",
            )
        with d2:
            dialect_model_id_or_path = st.text_input(
                "Dialect model id/path (Transformers)",
                value="",
                help="Local path (offline) or HF model id. Only used for backend=transformers or auto.",
            )
            dialect_min_confidence = st.slider(
                "Dialect min confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.05,
                help="Normalization is gated on this threshold.",
            )

        n1, n2 = st.columns(2)
        with n1:
            dialect_normalizer_backend = st.selectbox(
                "Dialect normalizer backend",
                options=["auto", "heuristic", "seq2seq", "none"],
                index=0,
                help="auto = rules-first + optional seq2seq if a model is provided.",
            )
            allow_remote_models = st.checkbox(
                "Allow remote model downloads",
                value=False,
                help="Off by default. Enable only if you want HF model-id downloads/caching.",
            )
        with n2:
            dialect_normalizer_model_id_or_path = st.text_input(
                "Dialect normalizer model id/path (seq2seq)",
                value="",
                help="Optional. Local path (offline) or HF id for a seq2seq dialect->standard normalizer.",
            )

    topk = int(topk)
    max_out = int(max_out)

    lexicon: dict[str, str] | None = None
    lexicon_path: str | None = None
    lexicon_keys: set[str] | None = None
    lexicon_source = "none"
    if "lex_upload" in locals() and lex_upload is not None:
        try:
            data = lex_upload.getvalue()
            lexicon_path = _write_uploaded_file_to_tmp(filename=lex_upload.name, data=data)
            lex_res = load_user_lexicon(lexicon_path)
            lexicon = lex_res.mappings
            lexicon_keys = set(lexicon.keys()) if lexicon else None
            lexicon_source = lex_res.source
            st.caption(f"Lexicon loaded: {len(lexicon)} entries")
        except Exception as e:
            st.warning(f"Could not load lexicon: {e}")
            lexicon = None
            lexicon_path = None
            lexicon_keys = None
            lexicon_source = "error"

    ft_path = (fasttext_model_path or "").strip()
    if not ft_path:
        ft_path = None
    else:
        try:
            if not Path(ft_path).expanduser().exists():
                st.caption("fastText model path set, but file not found (will be ignored).")
        except Exception:
            pass

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
        whatsapp_cleanup = st.checkbox(
            "WhatsApp export cleanup (v0.4)",
            value=False,
            help="Remove timestamps/system lines from exported chat logs; keeps just message text.",
        )
        analyze_clicked = st.form_submit_button("Analyze", type="primary")

    # Compute only when the user clicks Analyze (or first load).
    if "gck_last_analysis" not in st.session_state or analyze_clicked:
        raw_input = msg
        msg_to_analyze = msg
        if whatsapp_cleanup:
            try:
                cleaned = clean_whatsapp_chat_text(msg or "")
            except Exception:
                cleaned = ""
            if cleaned:
                msg_to_analyze = cleaned

        st.session_state["gck_last_raw_input"] = raw_input
        st.session_state["gck_last_preprocessed_input"] = msg_to_analyze

        # Keep the demo resilient if an older SDK version is imported in some environments.
        # We only pass supported kwargs based on signature inspection.
        desired_kwargs: dict[str, Any] = {
            "topk": topk,
            "numerals": numerals,
            "translit_mode": translit_mode,
            "translit_backend": translit_backend,
            "aggressive_normalize": aggressive_normalize,
            "user_lexicon_path": lexicon_path,
            "fasttext_model_path": ft_path,
            "dialect_backend": dialect_backend,
            "dialect_model_id_or_path": (dialect_model_id_or_path or "").strip() or None,
            "dialect_min_confidence": float(dialect_min_confidence),
            "dialect_normalize": bool(dialect_normalize),
            "dialect_normalizer_backend": dialect_normalizer_backend,
            "dialect_normalizer_model_id_or_path": (dialect_normalizer_model_id_or_path or "").strip()
            or None,
            "allow_remote_models": bool(allow_remote_models),
        }
        try:
            supported = set(inspect.signature(analyze_codemix).parameters.keys())
            filtered = {k: v for k, v in desired_kwargs.items() if k in supported}
            dropped = sorted(set(desired_kwargs.keys()) - set(filtered.keys()))
            if dropped:
                st.warning(
                    "Some v0.3 demo options aren't supported by the imported SDK. "
                    f"Ignoring: {', '.join(dropped)}. "
                    "If you're running from the repo, restart Streamlit to pick up latest code."
                )
            st.session_state["gck_last_analysis"] = analyze_codemix(msg_to_analyze, **filtered)
        except TypeError as e:
            st.warning(
                "SDK/demo mismatch while calling analyze_codemix(). "
                "Restart Streamlit (and ensure editable install / local src is used). "
                f"Error: {e}"
            )
            st.session_state["gck_last_analysis"] = analyze_codemix(
                msg_to_analyze, topk=topk, numerals=numerals
            )

    a = st.session_state["gck_last_analysis"]

    out1, out2 = st.columns(2)
    with out1:
        st.subheader("Before")
        st.caption("Raw user message (often Gujlish + English).")
        st.code(st.session_state.get("gck_last_raw_input", a.raw) or "")
    with out2:
        st.subheader("After")
        st.caption("Canonical text for downstream systems.")
        st.code(a.codemix or "")

    pre = st.session_state.get("gck_last_preprocessed_input", "") or ""
    raw = st.session_state.get("gck_last_raw_input", "") or ""
    if pre and raw and pre.strip() != raw.strip():
        st.caption("Preprocessed input (v0.4):")
        st.code(pre)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gujlish tokens", a.n_gu_roman_tokens)
    m2.metric("Converted", a.n_gu_roman_transliterated)
    m3.metric("Conversion rate", f"{a.pct_gu_roman_transliterated * 100:.1f}%")
    m4.metric("Backend", a.transliteration_backend)

    # v0.4 metrics (be defensive if an older SDK is imported).
    try:
        cs = a.codeswitch  # type: ignore[attr-defined]
    except Exception:
        toks = tokenize(a.normalized or "")
        tagged = tag_tokens(toks, lexicon_keys=lexicon_keys, fasttext_model_path=ft_path)
        cs = compute_code_switch_metrics(tagged)

    try:
        d = a.dialect  # type: ignore[attr-defined]
    except Exception:
        toks = tokenize(a.normalized or "")
        tagged = tag_tokens(toks, lexicon_keys=lexicon_keys, fasttext_model_path=ft_path)
        d = detect_dialect_from_tagged_tokens(tagged)

    try:
        dn = a.dialect_normalization  # type: ignore[attr-defined]
    except Exception:
        dn = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CMI", f"{cs.cmi:.1f}")
    c2.metric("Switch points", cs.n_switch_points)
    c3.metric("Dialect", d.dialect.value)
    c4.metric("Dialect confidence", f"{getattr(d, 'confidence', 0.0):.2f}")

    st.markdown("## What Changed")
    rows = _transliteration_rows(
        a.normalized,
        topk=topk,
        aggressive_normalize=aggressive_normalize,
        translit_backend=translit_backend,
        lexicon=lexicon,
        lexicon_keys=lexicon_keys,
        fasttext_model_path=ft_path,
    )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No Gujlish conversions detected for this input.")

    with st.expander("Token LID (v0.3: confidence + reason)", expanded=False):
        toks = tokenize(a.normalized or "")
        tagged = tag_tokens(toks, lexicon_keys=lexicon_keys, fasttext_model_path=ft_path)
        st.caption(
            "This is the token-level language ID used by the pipeline. "
            "Lexicon + optional fastText can influence Latin tokens."
        )
        st.dataframe(
            [
                {
                    "token": t.text,
                    "lang": t.lang.value,
                    "confidence": round(float(t.confidence), 3),
                    "reason": t.reason,
                }
                for t in tagged[:120]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Lexicon source: {lexicon_source}")

    with st.expander("Code-switching + dialect (v0.4)", expanded=False):
        st.caption("Heuristic metrics to quantify how mixed the input is (Gujarati vs English).")
        dialect_norm_applied = bool(getattr(dn, "changed", False)) if dn is not None else False
        dialect_norm_backend = getattr(dn, "backend", "none") if dn is not None else "none"
        st.dataframe(
            [
                {
                    "Metric": "CMI (0..100)",
                    "Value": round(float(cs.cmi), 2),
                },
                {"Metric": "Switch points", "Value": int(cs.n_switch_points)},
                {"Metric": "Gujarati tokens", "Value": int(cs.n_gu_tokens)},
                {"Metric": "English tokens", "Value": int(cs.n_en_tokens)},
                {"Metric": "Lexical tokens considered", "Value": int(cs.n_tokens_considered)},
                {"Metric": "Dialect guess", "Value": d.dialect.value},
                {"Metric": "Dialect backend", "Value": getattr(d, "backend", "heuristic")},
                {"Metric": "Dialect confidence", "Value": round(float(getattr(d, "confidence", 0.0)), 3)},
                {"Metric": "Dialect normalized", "Value": dialect_norm_applied},
                {"Metric": "Dialect normalizer backend", "Value": dialect_norm_backend},
            ],
            width="stretch",
            hide_index=True,
        )
        if getattr(d, "markers_found", None):
            st.caption("Dialect markers found (debug):")
            st.json(d.markers_found)
        if dn is not None and getattr(dn, "changed", False):
            st.caption("Dialect normalization output (debug):")
            try:
                st.code(" ".join(list(getattr(dn, "tokens_out", []))[:80]))
            except Exception:
                pass

    with st.expander("Batch helpers (CSV / JSONL) (v0.4)", expanded=False):
        st.caption("Upload a file, run preprocessing, download enriched output.")

        b1, b2 = st.columns(2)
        with b1:
            csv_up = st.file_uploader("CSV upload", type=["csv"], key="gck_csv_upload")
            csv_text_col = st.text_input("CSV text column", value="text", key="gck_csv_text_col")
            if csv_up is not None and st.button("Process CSV", key="gck_process_csv"):
                in_p = _write_uploaded_file_to_tmp(filename=csv_up.name, data=csv_up.getvalue())
                out_p = str(Path(in_p).with_suffix(".out.csv"))
                summ = process_csv_batch(in_p, out_p, text_column=csv_text_col)
                st.success(
                    f"Processed {summ.n_rows_out} rows (errors: {summ.n_errors})."
                )
                st.download_button(
                    "Download processed CSV",
                    data=Path(out_p).read_bytes(),
                    file_name=Path(out_p).name,
                    mime="text/csv",
                )

        with b2:
            jsonl_up = st.file_uploader("JSONL upload", type=["jsonl"], key="gck_jsonl_upload")
            jsonl_text_key = st.text_input("JSONL text key", value="text", key="gck_jsonl_text_key")
            if jsonl_up is not None and st.button("Process JSONL", key="gck_process_jsonl"):
                in_p = _write_uploaded_file_to_tmp(filename=jsonl_up.name, data=jsonl_up.getvalue())
                out_p = str(Path(in_p).with_suffix(".out.jsonl"))
                summ = process_jsonl_batch(in_p, out_p, text_key=jsonl_text_key)
                st.success(
                    f"Processed {summ.n_rows_out} rows (errors: {summ.n_errors})."
                )
                st.download_button(
                    "Download processed JSONL",
                    data=Path(out_p).read_bytes(),
                    file_name=Path(out_p).name,
                    mime="application/x-ndjson",
                )

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
        c_before = _lid_counts(a.raw, lexicon_keys=lexicon_keys, fasttext_model_path=ft_path)
        c_after = _lid_counts(a.codemix, lexicon_keys=lexicon_keys, fasttext_model_path=ft_path)
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

    st.markdown("## RAG (v0.5): Gujarat Facts Mini-KB")
    st.caption(
        "A tiny packaged dataset + retrieval helper. Use this to demo how canonicalization improves "
        "search/retrieval inputs (Gujarati script vs Gujlish)."
    )

    if not _RAG_AVAILABLE:
        st.info("RAG section unavailable in this environment (older install or missing module).")
        st.caption("If you are running from the repo, restart Streamlit to pick up v0.5 code.")
        rag_payload: dict[str, Any] | None = None
    else:
        ds = load_gujarat_facts_tiny()  # type: ignore[misc]
        q_examples = [q.query for q in ds.queries]
        default_q = q_examples[0] if q_examples else "ગુજરાતની રાજધાની કઈ છે?"

        with st.form("gck_rag_form", clear_on_submit=False):
            r1, r2 = st.columns([2, 1])
            with r1:
                ex = st.selectbox("Example query", options=q_examples or [default_q], index=0)
            with r2:
                use_after = st.checkbox(
                    "Use current 'After' text as query",
                    value=False,
                    help="Useful if your message is itself a question.",
                )

            rag_query = st.text_area(
                "Query",
                value=(a.codemix if use_after else ex),
                height=80,
                help="Try Gujlish too (e.g., 'gujarat ni rajdhani kai che?').",
            )

            preprocess_query = st.checkbox(
                "Preprocess query with CodeMix",
                value=True,
                help="Runs normalize + Gujlish-to-Gujarati conversion before retrieval.",
            )

            embedding_mode = st.radio(
                "Embeddings mode",
                options=["keyword", "hf"],
                index=0,
                help="keyword requires no extra deps. hf uses torch+transformers (optional).",
                horizontal=True,
            )

            hf_model_id_or_path = ""
            if embedding_mode == "hf":
                hf_model_id_or_path = st.text_input(
                    "HF model id/path (embeddings)",
                    value="",
                    help=(
                        "Recommended: a local path for offline-first use. "
                        "If you provide a HF id, you must enable 'Allow remote model downloads' in Settings."
                    ),
                )
                if not allow_remote_models:
                    st.caption("Remote model downloads are disabled (Settings). Local paths will work.")

            rag_topk = st.slider("Top-k", min_value=1, max_value=8, value=3, step=1)
            retrieve_clicked = st.form_submit_button("Retrieve", type="primary")

        if not retrieve_clicked:
            rag_payload = st.session_state.get("gck_last_rag")
        else:
            q_raw = rag_query or ""
            q_used = q_raw
            if preprocess_query:
                # Keep retrieval input in the same canonical format used elsewhere.
                q_used = render_codemix(
                    normalize_text(q_used),
                    topk=topk,
                    numerals=numerals,
                    translit_mode=translit_mode,
                    translit_backend=translit_backend,
                    aggressive_normalize=aggressive_normalize,
                    user_lexicon_path=lexicon_path,
                    fasttext_model_path=ft_path,
                    preserve_case=True,
                    preserve_numbers=True,
                )

            try:
                idx = _build_rag_index_cached(
                    embedding_mode=embedding_mode,
                    hf_model_id_or_path=hf_model_id_or_path,
                    allow_remote_models=bool(allow_remote_models),
                )
                if embedding_mode == "hf":
                    embed = make_hf_embedder(  # type: ignore[misc]
                        model_id_or_path=hf_model_id_or_path,
                        allow_remote_models=bool(allow_remote_models),
                    )
                else:
                    embed = _rag_keyword_embed

                results = idx.search(query=q_used, embed_texts=embed, topk=int(rag_topk))  # type: ignore[union-attr]
                recall1 = idx.recall_at_k(queries=ds.queries, embed_texts=embed, k=1)  # type: ignore[union-attr]
                recall3 = idx.recall_at_k(queries=ds.queries, embed_texts=embed, k=3)  # type: ignore[union-attr]

                rag_payload = {
                    "dataset": ds.name,
                    "source": ds.source,
                    "embedding_mode": embedding_mode,
                    "hf_model_id_or_path": hf_model_id_or_path,
                    "allow_remote_models": bool(allow_remote_models),
                    "query_raw": q_raw,
                    "query_used": q_used,
                    "topk": int(rag_topk),
                    "recall_at_1_tinyset": float(recall1),
                    "recall_at_3_tinyset": float(recall3),
                    "results": [
                        {
                            "doc_id": r.doc_id,
                            "score": float(r.score),
                            "domain": (r.meta or {}).get("domain", ""),
                            "tags": ", ".join(list((r.meta or {}).get("tags", []))[:6]),
                            "text": r.text,
                        }
                        for r in results
                    ],
                }
                st.session_state["gck_last_rag"] = rag_payload
            except Exception as e:
                rag_payload = {"error": str(e), "query_raw": rag_query, "query_used": q_used}
                st.session_state["gck_last_rag"] = rag_payload

        if rag_payload and rag_payload.get("error"):
            st.warning(f"RAG retrieval failed: {rag_payload.get('error')}")
        elif rag_payload:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Recall@1 (tiny set)", f"{float(rag_payload.get('recall_at_1_tinyset', 0.0)):.2f}")
            with m2:
                st.metric("Recall@3 (tiny set)", f"{float(rag_payload.get('recall_at_3_tinyset', 0.0)):.2f}")
            with m3:
                st.metric("Top-k", str(int(rag_payload.get('topk', 0) or 0)))

            st.markdown("**Query (raw)**")
            st.code(rag_payload.get("query_raw", "") or "")
            st.markdown("**Query (used for retrieval)**")
            st.code(rag_payload.get("query_used", "") or "")

            rows = rag_payload.get("results") if isinstance(rag_payload.get("results"), list) else []
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info("No results.")

            with st.expander("How to use results (prompt pattern)", expanded=False):
                st.caption("A minimal pattern for RAG-style prompting (you can paste into the Sarvam section).")
                ctx = "\n".join(
                    [f"- ({r['doc_id']}) {r['text']}" for r in rows[:3] if isinstance(r, dict)]
                ).strip()
                st.code(
                    (
                        "Use the context below to answer the question.\n\n"
                        f"Context:\n{ctx}\n\n"
                        f"Question:\n{rag_payload.get('query_used','')}\n\n"
                        "Answer in Gujarati."
                    ).strip()
                )

    st.divider()

    st.markdown("## Optional: AI Comparison (Sarvam-M)")
    st.caption("Use this to demonstrate answer quality and stability. Token savings is not the primary promise.")

    prompt_template = st.text_area(
        "Prompt template (use {text})",
        value="Please respond to the message below:\n\n{text}",
        height=90,
    )

    rag_last = st.session_state.get("gck_last_rag")
    rag_ok = (
        isinstance(rag_last, dict)
        and isinstance(rag_last.get("results"), list)
        and len(list(rag_last.get("results") or [])) > 0
        and not rag_last.get("error")
    )
    use_rag_context = st.checkbox(
        "Include RAG context (from last retrieval)",
        value=False,
        disabled=not rag_ok,
        help="Run the RAG section first to populate retrieved snippets.",
    )
    rag_apply_to_before = st.checkbox(
        "Also apply RAG context to Before prompt",
        value=False,
        disabled=not bool(use_rag_context),
        help="By default, context is applied only to the After prompt.",
    )
    rag_n_docs = st.slider(
        "RAG docs to include",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        disabled=not bool(use_rag_context),
    )
    use_rag_query_as_text_after = st.checkbox(
        "Use RAG query as {text} (After)",
        value=False,
        disabled=not rag_ok,
        help="Useful for direct QA. If off, {text}=the canonicalized message (After).",
    )
    if not rag_ok:
        st.caption("Tip: run a retrieval in the RAG section to enable context injection here.")
    elif use_rag_context:
        with st.expander("RAG context that will be injected", expanded=False):
            st.code(_rag_context_block(rag_last, n_docs=int(rag_n_docs)))

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
            text_before = a.raw
            text_after = a.codemix

            if rag_ok and use_rag_query_as_text_after:
                q_used = str(rag_last.get("query_used", "") or "").strip()
                if q_used:
                    text_after = q_used

            p_before = _apply_template(prompt_template, text_before)
            p_after = _apply_template(prompt_template, text_after)

            rag_ctx = ""
            if rag_ok and use_rag_context:
                rag_ctx = _rag_context_block(rag_last, n_docs=int(rag_n_docs))
                if rag_ctx:
                    p_after = f"{rag_ctx}\n\n{p_after}".strip()
                    if rag_apply_to_before:
                        p_before = f"{rag_ctx}\n\n{p_before}".strip()

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
                "rag_enabled": bool(use_rag_context),
                "rag_apply_to_before": bool(rag_apply_to_before),
                "rag_n_docs": int(rag_n_docs),
                "rag_query_used": (str(rag_last.get("query_used", "") or "") if rag_ok else ""),
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
        "analysis": _jsonable(asdict(a)),
        "rag": st.session_state.get("gck_last_rag"),
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

