# Gujarati Support And Ecommerce Case Study

This case study shows how `open-vernacular-ai-kit` fits in front of support, ecommerce, and RAG
workflows that receive Gujarati-English code-mixed text from WhatsApp, chat widgets, ticket forms,
and search boxes.

## Scenario

A support or ecommerce team receives user text such as:

| Raw user text | OVAK canonical code-mix |
| --- | --- |
| `maru order status shu chhe?` | `મારું order status શું છે?` |
| `maru payment nathi thayu` | `મારું payment નથી થયું` |
| `aaje delivery kyare aavse` | `આજે delivery ક્યારે આવશે` |
| `return pickup mate address change karvu chhe` | `return pickup માટે address change કરવું છે` |

These messages are hard for downstream systems because Gujarati intent words often arrive in Latin
script while product terms, order words, and support vocabulary stay in English.

## Why Normalize First

OVAK provides a deterministic preprocessing step before retrieval, LLM prompts, classification, or
ticket routing:

- native Gujarati stays in Gujarati script
- English product and support terms stay in Latin script
- Romanized Gujarati is converted to Gujarati script when possible
- punctuation, spacing, and code-mix rendering stay reproducible

That gives teams a stable text layer they can test locally before adding heavier model calls.

## Measured Snapshot

The committed downstream snapshot in `docs/data/downstream_uplift_snapshot.json` shows the strongest
current evidence for the support/ecommerce use case:

| Benchmark | Raw | Normalized | Absolute uplift |
| --- | ---: | ---: | ---: |
| hard code-mix retrieval `recall@1` | `0.8` | `1.0` | `+0.2` |
| hard code-mix retrieval `recall@3` | `0.9` | `1.0` | `+0.1` |
| prompt stability `mean_offdiag` | `0.8718` | `0.8907` | `+0.0189` |
| prompt stability `ref_min` | `0.8024` | `0.8826` | `+0.0802` |
| answer-quality suite exact match | `0.9545` | `1.0` | `+0.0455` |
| answer-quality suite mean answer similarity | `0.3582` | `0.3685` | `+0.0103` |

Interpret these as release-tracking signals, not universal production guarantees. They are useful
because they are versioned, reproducible from repo commands, and focused on raw-vs-normalized
behavior.

## Implementation Pattern

Normalize user text once at the edge of the application, then pass the canonical code-mix string to
retrieval, routing, or model prompts.

```python
from open_vernacular_ai_kit import render_codemix


def preprocess_support_message(text: str) -> str:
    return render_codemix(
        text,
        language="gu",
        translit_mode="sentence",
    )


raw_message = "maru order status shu chhe?"
clean_message = preprocess_support_message(raw_message)
print(clean_message)
```

For batch ticket cleanup, use the [batch CLI recipes](../cookbook/batch-cli-recipes.md).
For model and RAG integrations, use the [integration examples](../cookbook/integrations.md).

## Rollout Checklist

1. Start with Gujarati support or ecommerce messages where mixed English terms should be preserved.
2. Run `gck codemix --stats` on a small sample to inspect how many romanized vernacular tokens are
   being converted.
3. Add a small golden set of raw input and expected canonical code-mix output for your domain.
4. Preprocess both incoming queries and any comparable indexed text before retrieval.
5. Track retrieval or answer-quality deltas with the same raw-vs-normalized split used in the repo
   benchmarks.

## Boundaries

OVAK is not a translation layer and does not put remote models in the default runtime path. Its job is
to make messy vernacular-English text more canonical, testable, and pipeline-friendly before downstream
systems receive it.
