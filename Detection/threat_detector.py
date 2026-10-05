import joblib
import pandas as pd
import numpy as np


class ThreatDetector:

    def __init__(
        self,
        model_path="./Data/Processed/threat_classifier_tree.pkl",
        encoder_path="./Data/Processed/label_encoder.pkl",
    ):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

        self.label_to_id = self.encoder["label_to_id"]
        self.id_to_label = self.encoder["id_to_label"]

        self.expected_features = self.model.n_features_in_

        # Exact feature names and order used during training
        self.feature_names = list(
            self.model.feature_names_in_
        )

    def predict(self, features):
        """
        Predict the threat class for one network-flow observation.
        """

        # If a pandas Series is supplied
        if isinstance(features, pd.Series):
            features = features.to_frame().T

        # If a DataFrame is supplied
        elif isinstance(features, pd.DataFrame):
            features = features.copy()

        # Otherwise convert list/NumPy input to DataFrame
        else:
            features = np.asarray(
                features,
                dtype=np.float64
            )

            if features.ndim == 1:
                features = features.reshape(1, -1)

            features = pd.DataFrame(
                features,
                columns=self.feature_names
            )

        # Never allow the target label into the model
        if "label" in features.columns:
            features = features.drop(
                columns=["label"]
            )

        # Check number of features
        if features.shape[1] != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, "
                f"but received {features.shape[1]}"
            )

        # Make sure feature order exactly matches training
        features = features[self.feature_names]

        prediction = int(
            self.model.predict(features)[0]
        )

        probabilities = self.model.predict_proba(features)[0]

        confidence = float(
            np.max(probabilities)
        )

        label = self.id_to_label.get(
            prediction,
            "Unknown"
        )

        return {
            "class_id": prediction,
            "label": label,
            "probability": confidence,
        }