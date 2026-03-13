# Sarvam Teacher Mining

Use Sarvam as an offline teacher to discover better Hindi/Gujarati normalization candidates without
putting an LLM in the default runtime path.

## Why This Exists

The default OVAK normalization path should stay:

- deterministic
- offline-first
- easy to debug

This workflow is for mining candidate improvements that can later be reviewed and distilled into:

- language profile entries
- context-token rules
- sentence-level eval cases
- dialect assets

## Input Format

Create a JSONL file with one record per line:

```json
{"text":"tamne aaje office ma aavu chhe","language_hint":"gu","source":"support_chat"}
{"text":"meri maa ka naam kya hai","language_hint":"hi","source":"teacher_seed"}
```

Accepted fields:

- `text` or `input`
- `language_hint` (`gu`, `hi`, `mixed`, `unknown`)
- `source`
- `meta`

Bundled starter dataset:

- `eval/datasets/sarvam_teacher_seed.jsonl`

## Run Mining

Install the optional Sarvam dependency first:

```bash
pip install -e ".[sarvam]"
```

Then run:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_seed.jsonl \
  --output eval/out/sarvam_candidates/seed.jsonl \
  --model sarvam-m
```

Use `SARVAM_API_KEY` in your shell, or pass `--api-key`.

## Output Schema

Each output record contains:

- `input`
- `language_hint`
- `source`
- `model`
- `ovak_baseline`
- `sarvam_native`
- `sarvam_canonical`
- `english_tokens_keep`
- `candidate_tokens`
- `notes`
- `raw_response`

Example:

```json
{
  "input": "tamne aaje office ma aavu chhe",
  "language_hint": "gu",
  "source": "support_chat",
  "model": "sarvam-m",
  "ovak_baseline": "તમને આજે office માં આવું છે",
  "sarvam_native": "તમને આજે office માં આવવું છે",
  "sarvam_canonical": "તમને આજે office માં આવવું છે",
  "english_tokens_keep": ["office"],
  "candidate_tokens": [
    {
      "roman": "ma",
      "native": "માં",
      "type": "context_token",
      "confidence": 0.98,
      "notes": "locative postposition in Gujarati context"
    }
  ],
  "notes": "Keep obvious English tokens in Latin script."
}
```

## Important Constraint

Do not promote these records directly into shipped logic.

Use them as reviewed candidates only. The next step should be:

1. review mined records manually
2. accept or reject candidates
3. promote approved items into profile data or eval datasets
4. rerun tests and evals
