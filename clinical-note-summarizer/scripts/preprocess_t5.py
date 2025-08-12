#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer


DEFAULT_MODEL = "google/flan-t5-base"
DEFAULT_MAX_INPUT_LEN = 2048
DEFAULT_MAX_TARGET_LEN = 256


@dataclass
class PreprocessConfig:
    input_csv_path: Path
    input_text_column: str
    target_text_column: str
    model_name_or_path: str
    max_input_length: int
    max_target_length: int
    output_dir: Path


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Preprocess clinical note data for FLAN-T5 summarization",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(
            Path(
                "data/primary/mts-dialog/clinical_visit_note_summarization_corpus-main/data/mts-dialog/MTS_Dataset_TrainingSet.csv"
            )
        ),
        help="Path to input CSV",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="dialogue",
        help="Column containing source text to summarize (will be prefixed with 'summarize: ')",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="note",
        help="Column containing target summaries",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Tokenizer model name or path (e.g., google/flan-t5-base)",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=DEFAULT_MAX_INPUT_LEN,
        help="Max input sequence length",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=DEFAULT_MAX_TARGET_LEN,
        help="Max target sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("clinical-note-summarizer/data/processed/flan-t5-base")),
        help="Directory to write tokenized dataset",
    )

    args = parser.parse_args()

    return PreprocessConfig(
        input_csv_path=Path(args.input_csv),
        input_text_column=args.input_column,
        target_text_column=args.target_column,
        model_name_or_path=args.model,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        output_dir=Path(args.output_dir),
    )


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_hf_dataset(df: pd.DataFrame, input_col: str, target_col: str) -> Dataset:
    if input_col not in df.columns:
        raise KeyError(f"Missing input column '{input_col}' in CSV; available: {list(df.columns)}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column '{target_col}' in CSV; available: {list(df.columns)}")

    processed_df = pd.DataFrame(
        {
            "input_text": df[input_col].fillna("").astype(str).map(lambda x: f"summarize: {x}"),
            "target_text": df[target_col].fillna("").astype(str),
        }
    )

    features = Features({"input_text": Value("string"), "target_text": Value("string")})
    dataset = Dataset.from_pandas(processed_df, features=features, preserve_index=False)
    return dataset


def tokenize_examples(
    tokenizer: AutoTokenizer,
    example: Dict[str, str],
    max_input_len: int,
    max_target_len: int,
) -> Dict[str, list]:
    model_inputs = tokenizer(
        example["input_text"],
        max_length=max_input_len,
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target_text"],
            max_length=max_target_len,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_and_save_dataset(cfg: PreprocessConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Save a manifest for reproducibility
    manifest = {
        "input_csv": str(cfg.input_csv_path),
        "input_column": cfg.input_text_column,
        "target_column": cfg.target_text_column,
        "model": cfg.model_name_or_path,
        "max_input_length": cfg.max_input_length,
        "max_target_length": cfg.max_target_length,
    }
    (cfg.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    df = load_dataframe(cfg.input_csv_path)
    dataset = build_hf_dataset(df, cfg.input_text_column, cfg.target_text_column)

    # Force slow (sentencepiece) tokenizer to avoid rust-based fast tokenizers on Windows
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=False)

    tokenized = dataset.map(
        lambda ex: tokenize_examples(
            tokenizer=tokenizer,
            example=ex,
            max_input_len=cfg.max_input_length,
            max_target_len=cfg.max_target_length,
        ),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Replace pad token ids with -100 in labels for loss masking
    def mask_label_padding(batch: Dict[str, list]) -> Dict[str, list]:
        labels = batch["labels"]
        pad_token_id = tokenizer.pad_token_id
        masked = [
            [(tok if tok != pad_token_id else -100) for tok in seq] for seq in labels
        ]
        batch["labels"] = masked
        return batch

    tokenized = tokenized.map(mask_label_padding, batched=True, desc="Masking label padding")

    out_path = cfg.output_dir / "tokenized_dataset"
    tokenized.save_to_disk(str(out_path))
    print(f"Saved tokenized dataset to: {out_path}")


if __name__ == "__main__":
    cfg = parse_args()
    prepare_and_save_dataset(cfg)
