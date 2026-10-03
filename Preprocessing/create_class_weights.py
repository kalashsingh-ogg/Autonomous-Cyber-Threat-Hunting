import os
import joblib
import pandas as pd
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = "Data/Raw/archive/CICIoT23/train/train.csv"
OUTPUT_DIR = "Data/Processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "class_weights.pkl"
)

CHUNK_SIZE = 100_000

# Prevent extremely rare classes from receiving huge weights
MIN_WEIGHT = 1.0
MAX_WEIGHT = 8.0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COUNT LABELS
# ============================================================

print("=" * 70)
print("CREATING CLASS WEIGHTS")
print("=" * 70)

label_counts = Counter()
total_rows = 0

print("\nCounting labels...\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    counts = chunk["label"].value_counts()

    for label, count in counts.items():
        label_counts[label] += int(count)

    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )


# ============================================================
# CALCULATE WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING WEIGHTS")
print("=" * 70)

max_count = max(label_counts.values())

class_weights = {}

for label, count in label_counts.items():

    # Square-root weighting reduces the extreme effect
    # of very rare classes.
    weight = (max_count / count) ** 0.5

    # Keep weights within a stable range
    weight = max(
        MIN_WEIGHT,
        min(MAX_WEIGHT, weight)
    )

    class_weights[label] = weight


# ============================================================
# DISPLAY WEIGHTS
# ============================================================

print("\nClass weights:\n")

for label, count in sorted(
    label_counts.items(),
    key=lambda item: item[1]
):

    print(
        f"{label:<40} "
        f"Count: {count:>10,}   "
        f"Weight: {class_weights[label]:>6.3f}"
    )


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    class_weights,
    OUTPUT_FILE
)

print("\n" + "=" * 70)
print("CLASS WEIGHTS SAVED")
print("=" * 70)

print(f"\nLocation:")
print(OUTPUT_FILE)

print("\nTotal classes:", len(class_weights))