import pandas as pd
import numpy as np

TRAIN_PATH = "Data/Raw/archive/CICIoT23/train/train.csv"
CHUNK_SIZE = 100_000

print("=" * 70)
print("CICIoT2023 FEATURE ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# Read column names first
# ---------------------------------------------------------

columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

feature_columns = [column for column in columns if column != "label"]

print(f"\nTotal columns: {len(columns)}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Label column: label")

# ---------------------------------------------------------
# Store minimum and maximum values
# ---------------------------------------------------------

minimums = pd.Series(
    np.inf,
    index=feature_columns,
    dtype="float64"
)

maximums = pd.Series(
    -np.inf,
    index=feature_columns,
    dtype="float64"
)

# ---------------------------------------------------------
# Process the large file in chunks
# ---------------------------------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    features = chunk[feature_columns]

    # Update minimum values
    minimums = pd.concat(
        [minimums, features.min()]
    ).groupby(level=0).min()

    # Update maximum values
    maximums = pd.concat(
        [maximums, features.max()]
    ).groupby(level=0).max()

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {chunk_number * CHUNK_SIZE:,}",
        end="\r"
    )

print("\n")

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("=" * 70)
print("FEATURE RANGES")
print("=" * 70)

for column in feature_columns:
    print(
        f"{column:<25} "
        f"Min: {minimums[column]:>15.6f}   "
        f"Max: {maximums[column]:>15.6f}"
    )

# ---------------------------------------------------------
# Check for infinite values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INFINITE VALUE CHECK")
print("=" * 70)

infinite_counts = {
    column: 0
    for column in feature_columns
}

for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE):

    features = chunk[feature_columns]

    infinite_values = np.isinf(features).sum()

    for column in feature_columns:
        infinite_counts[column] += int(infinite_values[column])

total_infinite = sum(infinite_counts.values())

print(f"Total infinite values: {total_infinite:,}")

if total_infinite == 0:
    print("No infinite values found.")

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 70)