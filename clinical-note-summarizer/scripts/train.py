#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate
import numpy as np
import shutil


DEFAULT_MODEL = "google/flan-t5-base"
DEFAULT_TOKENIZED_PATH = (
    Path("clinical-note-summarizer/data/processed/flan-t5-base/tokenized_dataset")
)
DEFAULT_SPLITS_PATH = (
    Path("clinical-note-summarizer/data/processed/flan-t5-base/splits")
)
DEFAULT_EXPORT_DIR = Path("models/flan-t5-bhc-summarizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune FLAN-T5 on clinical summarization using Hugging Face Trainer.",
    )

    # Data paths
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=str(DEFAULT_SPLITS_PATH),
        help="Path to DatasetDict with 'train' and 'validation' splits (save_to_disk format)",
    )
    parser.add_argument(
        "--tokenized_dir",
        type=str,
        default=str(DEFAULT_TOKENIZED_PATH),
        help="Fallback: path to tokenized dataset (no splits) if splits_dir is missing",
    )

    # Model/tokenizer
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name or local path to a seq2seq checkpoint (e.g., google/flan-t5-base)",
    )

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="clinical-note-summarizer/runs/flan-t5-base")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)

    # Strategies
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 if supported")

    # Export
    parser.add_argument("--export_dir", type=str, default=str(DEFAULT_EXPORT_DIR))

    return parser.parse_args()


def load_splits_or_fallback(splits_dir: Path, tokenized_dir: Path) -> DatasetDict:
    if splits_dir.exists():
        return load_from_disk(str(splits_dir))
    # Fallback: load single dataset and create a split
    ds = load_from_disk(str(tokenized_dir))
    split = ds.train_test_split(test_size=0.1, seed=42, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_compute_metrics(tokenizer) -> callable:
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in labels before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = float(np.mean(prediction_lens))

        return {k: float(v) for k, v in result.items()}

    return compute_metrics


def copy_best_to_export_dir(output_dir: Path, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        for item in export_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for item in output_dir.iterdir():
            dest = export_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    splits_dir = Path(args.splits_dir)
    tokenized_dir = Path(args.tokenized_dir)

    datasets: DatasetDict = load_splits_or_fallback(splits_dir, tokenized_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    copy_best_to_export_dir(Path(args.output_dir), Path(args.export_dir))
    print(f"Exported model artifacts to: {args.export_dir}")


if __name__ == "__main__":
    main()
