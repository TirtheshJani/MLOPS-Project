#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load tokenized dataset from disk and split into train/validation sets.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path("clinical-note-summarizer/data/processed/flan-t5-base/tokenized_dataset")),
        help="Path to the tokenized dataset saved via datasets.save_to_disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("clinical-note-summarizer/data/processed/flan-t5-base/splits")),
        help="Where to save the split DatasetDict",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Validation split fraction (e.g., 0.1 for 90/10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds: Dataset = load_from_disk(str(input_dir))
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    dsd = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    dsd.save_to_disk(str(output_dir))
    print(f"Saved DatasetDict with splits to: {output_dir}")
    print({k: v.num_rows for k, v in dsd.items()})


if __name__ == "__main__":
    main()
