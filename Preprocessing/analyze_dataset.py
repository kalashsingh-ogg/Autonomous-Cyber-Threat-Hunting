import pandas as pd

# Path to the training dataset
train_path = "Data/Raw/archive/CICIoT23/train/train.csv"

# Number of rows to process at one time
CHUNK_SIZE = 100_000

print("=" * 60)
print("CICIoT2023 TRAINING DATASET ANALYSIS")
print("=" * 60)

total_rows = 0
label_counts = {}
missing_values = None
duplicate_rows = 0

print("\nReading the dataset in chunks...\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(train_path, chunksize=CHUNK_SIZE),
    start=1
):

    total_rows += len(chunk)

    # -------------------------------
    # Count labels
    # -------------------------------
    counts = chunk["label"].value_counts()

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count

    # -------------------------------
    # Count missing values
    # -------------------------------
    chunk_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = chunk_missing
    else:
        missing_values += chunk_missing

    # -------------------------------
    # Count duplicate rows
    # -------------------------------
    duplicate_rows += chunk.duplicated().sum()

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {total_rows:,}"
    )


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nTotal rows: {total_rows:,}")

print("\n" + "-" * 60)
print("LABEL DISTRIBUTION")
print("-" * 60)

for label, count in sorted(
    label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    percentage = (count / total_rows) * 100

    print(
        f"{label:<40} "
        f"{count:>12,} "
        f"({percentage:>6.2f}%)"
    )

print("\n" + "-" * 60)
print("NUMBER OF UNIQUE LABELS")
print("-" * 60)

print(len(label_counts))

print("\n" + "-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing_found = False

for column, count in missing_values.items():

    if count > 0:
        print(f"{column:<30} {count:,}")
        missing_found = True

if not missing_found:
    print("No missing values found.")

print("\n" + "-" * 60)
print("DUPLICATE ROWS")
print("-" * 60)

print(f"{duplicate_rows:,}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)