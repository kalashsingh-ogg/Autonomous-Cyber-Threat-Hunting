import pandas as pd

TRAIN_PATH = "Data/Processed/train_processed.csv"
CHUNK_SIZE = 100_000

print("=" * 70)
print("CICIoT2023 DUPLICATE CHECK")
print("=" * 70)

print("\nReading processed training data in chunks...")

total_rows = 0
duplicate_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE),
    start=1
):
    duplicates = chunk.duplicated().sum()

    duplicate_rows += int(duplicates)
    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,} | "
        f"Duplicates in chunk: {duplicates:,}"
    )

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nTotal rows checked: {total_rows:,}")
print(f"Duplicate rows found within chunks: {duplicate_rows:,}")

if duplicate_rows == 0:
    print("\nNo duplicate rows found within individual chunks.")
else:
    print(
        "\nDuplicate rows were found within individual chunks."
    )

print("\nNOTE:")
print(
    "This chunk-based check does not detect a duplicate "
    "that occurs in two different chunks."
)

print("\n" + "=" * 70)
print("DUPLICATE CHECK COMPLETE")
print("=" * 70)