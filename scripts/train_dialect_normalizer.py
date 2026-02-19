from __future__ import annotations

import argparse
from pathlib import Path

from gujarati_codemix_kit.dialect_datasets import load_dialect_normalization_jsonl


def _require_ml() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing deps for training. Install: `pip install -e '.[dialect-ml,eval]'`"
        ) from e


def _ensure_local_or_allowed(model_id_or_path: str, *, allow_remote: bool) -> None:
    try:
        p = Path(model_id_or_path).expanduser()
        if p.exists():
            return
    except Exception:
        pass
    if not allow_remote:
        raise SystemExit(
            "Remote model downloads are disabled. Provide a local path for --base-model "
            "or pass --allow-remote-models."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Gujarati dialect->standard seq2seq normalizer.")
    ap.add_argument("--train", required=True, help="Path to dialect-normalization JSONL (train split).")
    ap.add_argument("--valid", default="", help="Optional path to dialect-normalization JSONL (valid split).")
    ap.add_argument(
        "--base-model",
        default="google/byt5-small",
        help="HF model id or local path (e.g., google/byt5-small).",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write fine-tuned model.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-source-length", type=int, default=128)
    ap.add_argument("--max-target-length", type=int, default=128)
    ap.add_argument(
        "--allow-remote-models",
        action="store_true",
        help="Allow downloading HF model ids (otherwise require local paths).",
    )
    args = ap.parse_args()

    _require_ml()
    _ensure_local_or_allowed(args.base_model, allow_remote=bool(args.allow_remote_models))

    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    train_rows = load_dialect_normalization_jsonl(args.train)
    if not train_rows:
        raise SystemExit("Empty training set.")

    valid_rows = load_dialect_normalization_jsonl(args.valid) if args.valid else []

    train_ds = Dataset.from_list(
        [{"input": r.input, "target": r.expected, "dialect": r.dialect.value} for r in train_rows]
    )
    eval_ds = (
        Dataset.from_list(
            [{"input": r.input, "target": r.expected, "dialect": r.dialect.value} for r in valid_rows]
        )
        if valid_rows
        else None
    )

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    def preprocess(batch):
        src = batch["input"]
        tgt = batch["target"]
        model_inputs = tok(
            src,
            truncation=True,
            max_length=int(args.max_source_length),
        )
        with tok.as_target_tokenizer():
            labels = tok(
                tgt,
                truncation=True,
                max_length=int(args.max_target_length),
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=False)
    if eval_ds is not None:
        eval_ds = eval_ds.map(preprocess, batched=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        seed=int(args.seed),
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=200 if eval_ds is not None else None,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        report_to=[],
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))


if __name__ == "__main__":
    main()

