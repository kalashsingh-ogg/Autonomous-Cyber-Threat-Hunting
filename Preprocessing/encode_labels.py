import pandas as pd
import pickle
import os

TRAIN_PATH = "Data/Processed/train_processed.csv"
OUTPUT_PATH = "Data/Processed/label_encoder.pkl"

print("=" * 70)
print("CICIoT2023 LABEL ENCODER")
print("=" * 70)

print("\nReading labels from training dataset...")

labels = set()
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TRAIN_PATH,
        usecols=["label"],
        chunksize=100_000
    ),
    start=1
):
    labels.update(chunk["label"].dropna().unique())
    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )

labels = sorted(labels)

print("\n" + "=" * 70)
print("LABEL MAPPING")
print("=" * 70)

label_to_id = {
    label: index
    for index, label in enumerate(labels)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}

for label, index in label_to_id.items():
    print(f"{index:2d} -> {label}")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Total unique labels: {len(labels)}")

if len(labels) == 34:
    print("✓ All 34 labels found.")
else:
    print("⚠ Expected 34 labels.")

encoder = {
    "label_to_id": label_to_id,
    "id_to_label": id_to_label
}

os.makedirs("Data/Processed", exist_ok=True)

with open(OUTPUT_PATH, "wb") as file:
    pickle.dump(encoder, file)

print(f"\n✓ Label encoder saved to:")
print(OUTPUT_PATH)

print("\nLabel encoding preparation complete.")