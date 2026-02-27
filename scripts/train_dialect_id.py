from __future__ import annotations

import argparse
from pathlib import Path

from open_vernacular_ai_kit.dialect_datasets import load_dialect_id_jsonl


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
    ap = argparse.ArgumentParser(description="Train a Gujarati dialect-ID classifier (seq-cls).")
    ap.add_argument("--train", required=True, help="Path to dialect-id JSONL (train split).")
    ap.add_argument("--valid", default="", help="Optional path to dialect-id JSONL (valid split).")
    ap.add_argument(
        "--base-model",
        default="google/muril-large-cased",
        help="HF model id or local path (e.g., google/muril-large-cased).",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write fine-tuned model.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=256)
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
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_rows = load_dialect_id_jsonl(args.train)
    if not train_rows:
        raise SystemExit("Empty training set.")

    valid_rows = load_dialect_id_jsonl(args.valid) if args.valid else []

    labels = sorted({r.dialect.value for r in train_rows} | {r.dialect.value for r in valid_rows})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_ds = Dataset.from_list([{"text": r.text, "label": label2id[r.dialect.value]} for r in train_rows])
    eval_ds = (
        Dataset.from_list([{"text": r.text, "label": label2id[r.dialect.value]} for r in valid_rows])
        if valid_rows
        else None
    )

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def preprocess(batch):
        return tok(batch["text"], truncation=True, max_length=int(args.max_length))

    train_ds = train_ds.map(preprocess, batched=True)
    if eval_ds is not None:
        eval_ds = eval_ds.map(preprocess, batched=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
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
        report_to=[],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Convenience: write label mapping for downstream consumption.
    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

