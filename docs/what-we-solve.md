# What We Solve

`open-vernacular-ai-kit` solves a specific production gap:

Teams building AI apps for India frequently receive messy mixed-script, mixed-language user text
(for example, Gujarati script + Gujlish + English in one sentence). Downstream LLMs and retrieval
systems become less stable when this input is not normalized first.

This project is the normalization layer before chat, RAG, classification, or translation pipelines.

## Problem Pattern

- User text arrives in inconsistent script and spelling.
- Token-level language boundaries are unclear.
- Romanized vernacular words degrade retrieval and generation quality.
- App teams need deterministic preprocessing they can version and test.

## What This Kit Does

- Normalize punctuation/whitespace and Indic text forms.
- Tag token language (`en`, `gu_native`, `gu_roman`, `other`).
- Transliterate romanized Gujarati into Gujarati script when possible.
- Preserve canonical code-mix output:
  - Gujarati-native stays Gujarati script
  - English stays Latin
  - Gujlish converts to Gujarati script when possible
- Provide optional dialect detection/normalization and code-switch metrics.

## Ideal Users

- Conversational AI teams handling WhatsApp/support/chat inputs.
- Retrieval and search teams indexing vernacular user text.
- Builders who need SDK + CLI primitives, not full model training infrastructure.

## Landscape Matrix

| Category / Project | Primary Layer | Strengths | Gaps For This Use Case | Position vs OVAK |
| --- | --- | --- | --- | --- |
| Broad research NLP toolkits | Research NLP toolkit | Mature NLP utilities and broad foundational coverage | Not focused on product-grade code-mix canonicalization workflow | Complementary foundation; OVAK is product-oriented normalization layer |
| Large model-and-dataset ecosystems | Models + datasets | Large-scale model/data infrastructure | Different scope: model infrastructure, not lightweight app-preprocessing SDK | Complementary; OVAK sits upstream in app pipelines |
| Single-language code-mix normalizers | Narrow normalization | Useful for focused language-pair experiments | Usually narrow language scope and limited production packaging | OVAK focuses on reusable SDK/CLI pipeline with governance and eval harness |
| Open Vernacular AI Kit | App preprocessing layer | Deterministic pipeline, offline-first defaults, SDK + CLI + eval + demo | Early-stage language coverage today (Gujarati-first) | Positioning: production-ready vernacular normalization layer |

## Positioning Statement

`open-vernacular-ai-kit` is the production-facing vernacular normalization layer for India-focused AI
applications: lightweight, testable, and pipeline-friendly.
