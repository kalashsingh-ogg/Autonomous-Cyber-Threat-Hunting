import os
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier

# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = "./Data/Processed/train_processed.csv"
ENCODER_FILE = "./Data/Processed/label_encoder.pkl"
WEIGHTS_FILE = "./Data/Processed/class_weights.pkl"
MODEL_FILE = "./Data/Processed/threat_classifier_weighted.pkl"

CHUNK_SIZE = 100_000


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("=" * 70)
print("WEIGHTED THREAT CLASSIFIER TRAINING")
print("=" * 70)

print("\nLoading label encoder...")

label_encoder = joblib.load(
    ENCODER_FILE
)

print("Label encoder loaded.")

# Your encoder is a dictionary:
# label -> integer
label_to_id = label_encoder["label_to_id"]
id_to_label = label_encoder["id_to_label"]

classes = list(id_to_label.keys())

print("\nNumber of classes:", len(classes))
print("Classes:", classes)


# ============================================================
# LOAD CLASS WEIGHTS
# ============================================================

print("\nLoading class weights...")

class_weights = joblib.load(
    WEIGHTS_FILE
)

print("Class weights loaded.")
print("Number of weighted classes:", len(class_weights))


# ============================================================
# CONVERT LABEL WEIGHTS TO ENCODED-LABEL WEIGHTS
# ============================================================

encoded_class_weights = {}

for label, encoded_value in label_to_id.items():

    if label not in class_weights:
        raise ValueError(
            f"Missing class weight for label: {label}"
        )

    encoded_class_weights[encoded_value] = class_weights[label]


print("\nEncoded class weights ready.")


# ============================================================
# CREATE MODEL
# ============================================================

model = SGDClassifier(
    loss="log_loss",
    random_state=42,
    max_iter=1,
    learning_rate="optimal"
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING WEIGHTED TRAINING")
print("=" * 70)

print("\nTraining file:", TRAIN_FILE)
print("Chunk size:", CHUNK_SIZE)

first_chunk = True
total_rows = 0
chunk_number = 0

for chunk in pd.read_csv(
    TRAIN_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X = chunk.drop(
        columns=["label"]
    )

    labels = chunk["label"]

    # --------------------------------------------------------
    # Convert labels using dictionary
    # --------------------------------------------------------

    unknown_labels = set(labels.unique()) - set(
    label_to_id.keys()
)

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    y = labels.map(
    label_to_id
).astype("int64")

    # --------------------------------------------------------
    # Create one weight per training row
    # --------------------------------------------------------

    sample_weights = y.map(
        encoded_class_weights
    ).astype("float64")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    if first_chunk:

        model.partial_fit(
            X,
            y,
            classes=classes,
            sample_weight=sample_weights
        )

        first_chunk = False

    else:

        model.partial_fit(
            X,
            y,
            sample_weight=sample_weights
        )

    total_rows += len(chunk)

    print(
        f"Chunk {chunk_number} completed | "
        f"Rows processed: {total_rows:,}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "./Data/Processed",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("WEIGHTED TRAINING COMPLETED")
print("=" * 70)

print(
    "Total rows trained:",
    f"{total_rows:,}"
)

print(
    "Model saved to:",
    MODEL_FILE
)