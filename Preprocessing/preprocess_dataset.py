import os
import joblib
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

BASE_DIR = "Data/Raw/archive/CICIoT23"
OUTPUT_DIR = "Data/Processed"

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "scaler.pkl"
)


# ============================================================
# DATASET FILES
# ============================================================

DATASETS = {
    "train": os.path.join(
        BASE_DIR,
        "train",
        "train.csv"
    ),

    "validation": os.path.join(
        BASE_DIR,
        "validation",
        "validation.csv"
    ),

    "test": os.path.join(
        BASE_DIR,
        "test",
        "test.csv"
    )
}


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_FILES = {
    "train": os.path.join(
        OUTPUT_DIR,
        "train_processed.csv"
    ),

    "validation": os.path.join(
        OUTPUT_DIR,
        "validation_processed.csv"
    ),

    "test": os.path.join(
        OUTPUT_DIR,
        "test_processed.csv"
    )
}


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD SCALER
# ============================================================

print("=" * 70)
print("CICIoT2023 DATA PREPROCESSING")
print("=" * 70)

print("\nLoading trained scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

print("\nReading column information...")

columns = pd.read_csv(
    DATASETS["train"],
    nrows=0
).columns.tolist()

feature_columns = [
    column
    for column in columns
    if column != "label"
]

print(f"Total columns: {len(columns)}")
print(f"Feature columns: {len(feature_columns)}")
print("Label column: label")


# ============================================================
# PROCESS EACH DATASET
# ============================================================

for dataset_name, input_path in DATASETS.items():

    output_path = OUTPUT_FILES[dataset_name]

    print("\n" + "=" * 70)
    print(f"PROCESSING: {dataset_name.upper()}")
    print("=" * 70)

    print(f"\nInput:")
    print(input_path)

    print(f"\nOutput:")
    print(output_path)

    # Remove an old processed file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

        print("\nExisting processed file removed.")

    total_rows = 0
    first_chunk = True

    # --------------------------------------------------------
    # Read CSV in chunks
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        # ----------------------------------------------------
        # Separate features and labels
        # ----------------------------------------------------

        features = chunk[feature_columns]
        labels = chunk["label"]

        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        scaled_features = scaler.transform(
            features
        )

        # Convert scaled NumPy array back to DataFrame
        scaled_features = pd.DataFrame(
            scaled_features,
            columns=feature_columns
        )

        # ----------------------------------------------------
        # Put label back
        # ----------------------------------------------------

        scaled_features["label"] = labels.values

        # ----------------------------------------------------
        # Write to output CSV
        # ----------------------------------------------------

        scaled_features.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_rows += len(chunk)

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows: {total_rows:,}"
        )

    print(
        f"\n{dataset_name.upper()} COMPLETE"
    )

    print(
        f"Total rows processed: {total_rows:,}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nProcessed files:")

for dataset_name, output_path in OUTPUT_FILES.items():

    print(
        f"{dataset_name:<12} -> {output_path}"
    )

print("\nOriginal CICIoT2023 files were not modified.")