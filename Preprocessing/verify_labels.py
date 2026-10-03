import pandas as pd

TRAIN_PATH = "Data/Processed/train_processed.csv"
CHUNK_SIZE = 100_000

print("=" * 70)
print("CICIoT2023 LABEL VERIFICATION")
print("=" * 70)

label_counts = {}
total_rows = 0

print("\nReading processed training dataset...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TRAIN_PATH,
        usecols=["label"],
        chunksize=CHUNK_SIZE
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


print("\n" + "=" * 70)
print("LABEL RESULTS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")
print(f"Unique labels: {len(label_counts)}")

print("\nLabel distribution:")
print("-" * 70)

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


print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

if total_rows == 5_491_971:
    print("✓ Training row count is correct.")
else:
    print(
        f"✗ Unexpected row count: {total_rows:,}"
    )

if len(label_counts) == 34:
    print("✓ All 34 labels are present.")
else:
    print(
        f"✗ Expected 34 labels, found {len(label_counts)}."
    )

if any(pd.isna(label) for label in label_counts):
    print("✗ Missing labels detected.")
else:
    print("✓ No missing labels detected.")

print("\n" + "=" * 70)
print("LABEL VERIFICATION COMPLETE")
print("=" * 70)