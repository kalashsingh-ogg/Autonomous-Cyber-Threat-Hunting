import pandas as pd

from Detection.threat_detector import ThreatDetector


TEST_FILE = "Data/Processed/test_processed.csv"


def main():
    print("Loading threat detector...")

    detector = ThreatDetector()

    print("Loading test samples...")

    df = pd.read_csv(TEST_FILE, nrows=5)

    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    print(f"Number of features: {len(feature_columns)}")

    for index, row in df.iterrows():

        features = row[feature_columns]

        result = detector.predict(features)

        print("\n--------------------------------")
        print(f"Sample: {index}")
        print(f"Actual label: {row['label']}")
        print(f"Predicted label: {result['label']}")
        print(f"Class ID: {result['class_id']}")
        print(f"Confidence: {result['probability']:.4f}")
        print("--------------------------------")


if __name__ == "__main__":
    main()