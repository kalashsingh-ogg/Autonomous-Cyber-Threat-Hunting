import pandas as pd

# Path to the CICIoT2023 training dataset
train_path = "Data/Raw/archive/CICIoT23/train/train.csv"

print("Reading a small sample of the dataset...\n")

# Read ONLY the first 5 rows
# This prevents the 1.6 GB file from being loaded into memory.
train_sample = pd.read_csv(train_path, nrows=5)

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for number, column in enumerate(train_sample.columns, start=1):
    print(f"{number}. {column}")

print("\n" + "=" * 60)
print("NUMBER OF COLUMNS")
print("=" * 60)

print(len(train_sample.columns))

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(train_sample.to_string())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(train_sample.dtypes)