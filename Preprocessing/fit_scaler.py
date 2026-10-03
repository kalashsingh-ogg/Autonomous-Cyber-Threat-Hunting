import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "Data/Raw/archive/CICIoT23/train/train.csv"

OUTPUT_DIR = "Data/Processed"

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "scaler.pkl"
)

LABEL_MAP_PATH = os.path.join(
    OUTPUT_DIR,
    "label_mapping.csv"
)

# Number of rows loaded into memory at once
CHUNK_SIZE = 100_000


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GET COLUMN NAMES
# ============================================================

print("=" * 70)
print("PREPARING SCALER")
print("=" * 70)

columns = pd.read_csv(
    TRAIN_PATH,
    nrows=0
).columns.tolist()

# Everything except label is a feature
feature_columns = [
    column for column in columns
    if column != "label"
]

print(f"\nTotal columns: {len(columns)}")
print(f"Feature columns: {len(feature_columns)}")


# ============================================================
# CREATE SCALER
# ============================================================

scaler = StandardScaler()


# ============================================================
# FIRST PASS
# FIT SCALER ON TRAINING DATA
# ============================================================

print("\nFitting scaler using training data...")
print("This may take some time.\n")

total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TRAIN_PATH,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    features = chunk[feature_columns]

    # Update the scaler with this chunk
    scaler.partial_fit(features)

    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\n" + "=" * 70)
print("SCALER SAVED")
print("=" * 70)

print(f"\nLocation:")
print(SCALER_PATH)

print("\nScaler fitting complete.")