import pandas as pd
import os

DATASETS = {
    "Train": "Data/Processed/train_processed.csv",
    "Validation": "Data/Processed/validation_processed.csv",
    "Test": "Data/Processed/test_processed.csv"
}

print("=" * 70)
print("CICIoT2023 PROCESSED DATASET VERIFICATION")
print("=" * 70)

for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} DATASET")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        continue

    print(f"File: {file_path}")

    total_rows = 0
    label_counts = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            usecols=["label"],
            chunksize=100_000
        ),
        start=1
    ):
        counts = chunk["label"].value_counts()

        for label, count in counts.items():
            label_counts[label] = (
                label_counts.get(label, 0) + int(count)
            )

        total_rows += len(chunk)

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows: {total_rows:,}"
        )

    print("\nResults:")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique labels: {len(label_counts)}")

    print("\nLabels:")
    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / total_rows) * 100

        print(
            f"{label:<40} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    if len(label_counts) == 34:
        print("\n✓ All 34 labels are present.")
    else:
        print(
            f"\n⚠ Expected 34 labels, "
            f"found {len(label_counts)}."
        )

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)