from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


DATA_CSV = Path(
    "data/primary/mts-dialog/clinical_visit_note_summarization_corpus-main/"
    "data/mts-dialog/MTS_Dataset_TrainingSet.csv"
)


def tokenize_words(text: str) -> list[str]:
    return [t for t in text.replace("\n", " ").split() if t]


def main() -> None:
    if not DATA_CSV.exists():
        raise SystemExit(f"Dataset not found: {DATA_CSV}. Run scripts/download_mts_dialog.py first.")

    num_rows = 0
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    # For this dataset, we'll use `dialogue` as input and `section_text` as the target summary prototype
    # for sectioned notes (users may later craft a proper pairing strategy).

    examples: list[tuple[str, str]] = []

    with DATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue = (row.get("dialogue") or "").strip()
            section_text = (row.get("section_text") or "").strip()
            if not dialogue or not section_text:
                continue
            num_rows += 1

            input_len = len(tokenize_words(dialogue))
            target_len = len(tokenize_words(section_text))
            input_lengths.append(input_len)
            target_lengths.append(target_len)

            if len(examples) < 10:
                examples.append((dialogue[:500], section_text[:300]))

    avg_dialogue_len = int(mean(input_lengths)) if input_lengths else 0
    avg_summary_len = int(mean(target_lengths)) if target_lengths else 0

    print(f"Samples with both dialogue and section_text: {num_rows}")
    print(f"Average dialogue length (words): {avg_dialogue_len}")
    print(f"Average section_text (summary) length (words): {avg_summary_len}")

    print("\nExample pairs (truncated):")
    for i, (inp, tgt) in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print("Input (dialogue):", inp)
        print("Target (section_text):", tgt)

    # Write a brief summary to docs
    summary_path = Path("docs/eda_summary_mts_dialog.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("## MTS-Dialog EDA Summary (Training Set)\n\n")
        f.write(f"- Samples with both dialogue and section_text: {num_rows}\n")
        f.write(f"- Average dialogue length (words): {avg_dialogue_len}\n")
        f.write(f"- Average summary length (words): {avg_summary_len}\n")
        f.write("\nNotes: Counts computed from MTS_Dataset_TrainingSet.csv; summaries use the `section_text` field.\n")


if __name__ == "__main__":
    main()


