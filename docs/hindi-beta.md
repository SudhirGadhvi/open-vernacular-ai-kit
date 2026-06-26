# Hindi Beta

Hindi support is currently a beta language profile. It uses the same offline-first pipeline as the
Gujarati profile, but the shipped coverage is intentionally narrower and focused on deterministic
romanized-Hindi normalization.

## Current Scope

The beta profile is designed for common support and ecommerce text where English product terms should
stay in Latin script while Hindi function words, question words, and high-confidence support verbs are
rendered in Devanagari.

| Raw input | Canonical code-mix |
| --- | --- |
| `coupon code bilkul work nhi kr rha` | `coupon code बिल्कुल work नहीं कर रहा` |
| `pickup subah hoga ya dopahar baad?` | `pickup सुबह होगा या दोपहर बाद?` |
| `group me jo support number bheja tha us par koi phone nahi utha raha to dusra number dijiye` | `group में जो support number भेजा था उस पर कोई phone नहीं उठा रहा तो दूसरा number दीजिए` |
| `exchange nahi simple refund chahiye kyunki main same item dubara nahi mangwana chahta` | `exchange नहीं simple refund चाहिए क्योंकि मैं same item दुबारा नहीं मंगवाना चाहता` |

## Regression Coverage

Current Hindi beta validation is intentionally local and deterministic:

- `56` packaged sentence-level Hindi cases in `language_sentences`
- `34` rendered Hindi code-mix quality cases in `tests/test_language_quality.py`
- profile-data assertions for promoted support tokens such as `nhi`, `kr`, `rha`, `bheja`, `koi`,
  `dusra`, and `mangwana`

Run the focused checks:

```bash
python3 -m pytest tests/test_language_profile_data.py tests/test_language_quality.py -q
python3 -m open_vernacular_ai_kit.cli eval --dataset language_sentences --language hi
```

Run a quick CLI smoke:

```bash
gck codemix --language hi --translit-mode sentence "coupon code bilkul work nhi kr rha"
```

Expected output:

```text
coupon code बिल्कुल work नहीं कर रहा
```

## Product Guidance

Use Hindi beta when:

- support or ecommerce inputs are Hindi-English code-mix
- English operational terms such as `coupon`, `pickup`, `support`, `refund`, and `number` should stay
  readable in Latin script
- deterministic preprocessing matters more than broad translation coverage

Keep collecting domain golden cases before treating Hindi as production-ready. Gujarati remains the
primary production-ready language profile in this release.
