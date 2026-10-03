import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "./Data/Processed/threat_classifier_weighted.pkl"
ENCODER_FILE = "./Data/Processed/label_encoder.pkl"

VALIDATION_FILE = "./Data/Processed/validation_processed.csv"
TEST_FILE = "./Data/Processed/test_processed.csv"

CHUNK_SIZE = 100_000


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")

label_encoder = joblib.load(ENCODER_FILE)

# Your encoder is a dictionary containing mappings.
# Detect the string -> integer mapping automatically.

if isinstance(label_encoder, dict):

    label_to_id = None

    for value in label_encoder.values():

        if isinstance(value, dict):

            if all(isinstance(k, str) for k in value.keys()) and \
               all(isinstance(v, (int, np.integer)) for v in value.values()):

                label_to_id = value
                break

    if label_to_id is None:
        raise ValueError(
            "Could not find string -> integer label mapping "
            "inside label_encoder.pkl"
        )

else:
    raise TypeError(
        f"Unsupported encoder type: {type(label_encoder)}"
    )


# Create reverse mapping

id_to_label = {
    int(value): key
    for key, value in label_to_id.items()
}

classes = sorted(id_to_label.keys())

print("Number of classes:", len(classes))
print("Classes:", classes)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(file_path, dataset_name):

    print("\n" + "=" * 60)
    print(f"EVALUATING {dataset_name}")
    print("=" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    y_true = []
    y_pred = []

    total_rows = 0
    chunk_number = 0

    print("Dataset:", file_path)
    print("Chunk size:", f"{CHUNK_SIZE:,}")
    print("\nStarting evaluation...\n")

    # --------------------------------------------------------
    # Read dataset in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        # Separate features and labels

        X = chunk.drop(columns=["label"])

        labels = chunk["label"]

        # Convert string labels -> integer IDs

        try:

            y = labels.map(label_to_id)

        except Exception as e:

            raise ValueError(
                f"Error encoding labels in chunk {chunk_number}: {e}"
            )

        # Check for unknown labels

        if y.isna().any():

            unknown_labels = labels[y.isna()].unique()

            raise ValueError(
                f"Unknown labels found in chunk "
                f"{chunk_number}: {unknown_labels}"
            )

        y = y.astype(int)

        # ----------------------------------------------------
        # Make predictions
        # ----------------------------------------------------

        predictions = model.predict(X)

        # Store results

        y_true.extend(y.tolist())
        y_pred.extend(predictions.tolist())

        total_rows += len(chunk)

        print(
            f"Chunk {chunk_number} completed | "
            f"Rows evaluated: {total_rows:,}"
        )

    # Convert to NumPy arrays

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print(f"{dataset_name} RESULTS")
    print("=" * 60)

    print(f"Total rows evaluated : {total_rows:,}")

    print(f"\nAccuracy             : {accuracy:.4f}")
    print(f"Macro Precision      : {precision_macro:.4f}")
    print(f"Macro Recall         : {recall_macro:.4f}")
    print(f"Macro F1             : {f1_macro:.4f}")

    print(f"\nWeighted Precision   : {precision_weighted:.4f}")
    print(f"Weighted Recall      : {recall_weighted:.4f}")
    print(f"Weighted F1         : {f1_weighted:.4f}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print(f"{dataset_name} CLASSIFICATION REPORT")
    print("=" * 60)

    target_names = [
        id_to_label[class_id]
        for class_id in classes
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("=" * 60)
    print(f"{dataset_name} CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    )

    print(cm_df.to_string())

    return {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted
    }


# ============================================================
# VALIDATION SET
# ============================================================

validation_results = evaluate_dataset(
    VALIDATION_FILE,
    "VALIDATION"
)


# ============================================================
# TEST SET
# ============================================================

test_results = evaluate_dataset(
    TEST_FILE,
    "TEST"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 70)

print("\nValidation:")
print(
    f"Accuracy: {validation_results['accuracy']:.4f} | "
    f"Macro F1: {validation_results['macro_f1']:.4f} | "
    f"Weighted F1: {validation_results['weighted_f1']:.4f}"
)

print("\nTest:")
print(
    f"Accuracy: {test_results['accuracy']:.4f} | "
    f"Macro F1: {test_results['macro_f1']:.4f} | "
    f"Weighted F1: {test_results['weighted_f1']:.4f}"
)

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)