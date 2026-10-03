import pickle

ENCODER_PATH = "Data/Processed/label_encoder.pkl"

print("=" * 70)
print("VERIFYING LABEL ENCODER")
print("=" * 70)

with open(ENCODER_PATH, "rb") as file:
    encoder = pickle.load(file)

label_to_id = encoder["label_to_id"]
id_to_label = encoder["id_to_label"]

print(f"\nTotal labels: {len(label_to_id)}")

print("\nTesting label -> ID:")
test_label = "DDoS-SYN_Flood"
test_id = label_to_id[test_label]

print(f"{test_label} -> {test_id}")

print("\nTesting ID -> label:")
print(f"{test_id} -> {id_to_label[test_id]}")

print("\nChecking all mappings...")

for label, index in label_to_id.items():
    if id_to_label[index] != label:
        print("ERROR:", label, index)
        raise ValueError("Encoder mapping is inconsistent.")

print("\n✓ All mappings are consistent.")
print("✓ Label encoder verification successful.")

print("\n" + "=" * 70)