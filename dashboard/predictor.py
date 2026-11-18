
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class XHIDSAutoencoder(nn.Module):
    """Architecture matching the trained X-HIDS checkpoint."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(71, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 71)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class XHIDSPredictor:
    """Loads trained X-HIDS models and predicts network-flow threats."""

    def __init__(self, base_dir=None):

        self.base_dir = Path(
            base_dir or Path(__file__).resolve().parent
        )

        self.model_dir = self.base_dir / "models"
        self.preprocessing_dir = self.base_dir / "preprocessing"

        # Load feature configuration
        with open(
            self.preprocessing_dir / "final_feature_configuration.json",
            "r"
        ) as f:
            feature_config = json.load(f)

        self.features = feature_config["features"]

        if len(self.features) != 71:
            raise ValueError("Expected exactly 71 input features.")

        # Load training-only medians
        median_bundle = joblib.load(
            self.preprocessing_dir / "training_feature_medians.joblib"
        )

        if isinstance(median_bundle, dict):
            median_features = median_bundle["features"]
            median_values = median_bundle["medians"]
        else:
            raise ValueError("Unexpected training median artifact format.")

        if median_features != self.features:
            raise ValueError("Training median feature order mismatch.")

        self.medians = np.asarray(
            median_values, dtype=np.float64
        )

        if len(self.medians) != 71:
            raise ValueError("Expected 71 training median values.")

        # Load scaler
        self.scaler = joblib.load(
            self.preprocessing_dir / "standard_scaler.joblib"
        )

        if self.scaler.n_features_in_ != 71:
            raise ValueError("Scaler feature count mismatch.")

        # Load Random Forest
        self.rf = joblib.load(
            self.model_dir / "random_forest_baseline.joblib"
        )

        if self.rf.n_features_in_ != 71:
            raise ValueError("Random Forest feature count mismatch.")

        # Load XGBoost bundle
        self.xgb_bundle = joblib.load(
            self.model_dir / "xgboost_baseline.joblib"
        )

        self.xgb = self.xgb_bundle["model"]
        self.label_encoder = self.xgb_bundle["label_encoder"]

        if self.xgb_bundle["features"] != self.features:
            raise ValueError("XGBoost feature order mismatch.")

        # Load Isolation Forest bundle
        self.if_bundle = joblib.load(
            self.model_dir / "isolation_forest_baseline.joblib"
        )

        self.isolation_forest = self.if_bundle["model"]
        self.if_threshold = float(self.if_bundle["threshold"])

        if self.if_bundle["features"] != self.features:
            raise ValueError("Isolation Forest feature order mismatch.")

        # Load Autoencoder checkpoint
        checkpoint = torch.load(
            self.model_dir / "autoencoder_baseline.pth",
            map_location="cpu",
            weights_only=False
        )

        if checkpoint["input_features"] != self.features:
            raise ValueError("Autoencoder feature order mismatch.")

        self.autoencoder = XHIDSAutoencoder()

        self.autoencoder.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.autoencoder.eval()

        self.ae_threshold = float(checkpoint["threshold"])

        # Load hybrid configuration
        with open(
            self.model_dir / "hybrid_model_config.json",
            "r"
        ) as f:
            self.hybrid_config = json.load(f)

        self.weights = self.hybrid_config["hybrid_risk_weights"]
        self.alert_threshold = float(
            self.hybrid_config["alert_threshold"]
        )

        # Original data-cleaning policy
        with open(
            self.preprocessing_dir / "data_cleaning_policy.json",
            "r"
        ) as f:
            self.cleaning_policy = json.load(f)

        self.negative_to_missing = set(
            self.cleaning_policy["negative_to_missing"]
        )

        self.preserve_negative_sentinel = set(
            self.cleaning_policy["preserve_negative_sentinel"]
        )

        # Check the expected weight configuration
        expected_weights = {
            "xgboost": 0.35,
            "random_forest": 0.15,
            "isolation_forest": 0.25,
            "autoencoder": 0.25
        }

        for name, expected in expected_weights.items():
            if not np.isclose(self.weights[name], expected):
                raise ValueError(
                    f"Unexpected hybrid weight for {name}"
                )

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Hybrid weights must sum to 1.")

        print("X-HIDS prediction engine loaded successfully.")

    def preprocess(self, uploaded_df):
        """Apply the original X-HIDS feature and cleaning policy."""

        if not isinstance(uploaded_df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if uploaded_df.empty:
            raise ValueError("Uploaded CSV contains no records.")

        if uploaded_df.columns.duplicated().any():
            raise ValueError("CSV contains duplicate column names.")

        missing_features = [
            feature for feature in self.features
            if feature not in uploaded_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        # Select only the original 71 features in training order
        X = uploaded_df[self.features].copy()

        # Convert non-numeric values to missing values
        for column in self.features:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )

        # Replace positive and negative infinity
        X = X.replace([np.inf, -np.inf], np.nan)

        # Apply the original negative-value cleaning policy
        for column in self.negative_to_missing:
            if column in X.columns:
                X.loc[X[column] < 0, column] = np.nan

        # Training-only median imputation
        X = X.fillna(
            pd.Series(self.medians, index=self.features)
        )

        X = X.astype(np.float64)

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "Non-finite values remain after preprocessing."
            )

        return X

    def predict(self, uploaded_df):
        """Run all four models and calculate the hybrid risk score."""

        X = self.preprocess(uploaded_df)

        # Unscaled input for supervised models
        rf_predictions = self.rf.predict(X)

        rf_probabilities = self.rf.predict_proba(X)
        rf_classes = list(self.rf.classes_)

        xgb_predictions_encoded = self.xgb.predict(X)
        xgb_predictions = self.label_encoder.inverse_transform(
            np.asarray(xgb_predictions_encoded, dtype=int)
        )

        xgb_probabilities = self.xgb.predict_proba(X)
        xgb_classes = list(self.label_encoder.classes_)

        # Probability of an attack (anything other than BENIGN)
        if "BENIGN" not in rf_classes:
            raise ValueError("BENIGN class missing from Random Forest.")

        if "BENIGN" not in xgb_classes:
            raise ValueError("BENIGN class missing from XGBoost.")

        rf_benign_index = rf_classes.index("BENIGN")
        xgb_benign_index = xgb_classes.index("BENIGN")

        rf_attack_probability = (
            1.0 - rf_probabilities[:, rf_benign_index]
        )

        xgb_attack_probability = (
            1.0 - xgb_probabilities[:, xgb_benign_index]
        )

        # Scaled input for anomaly models
        X_scaled = self.scaler.transform(X)

        # Isolation Forest: consistently use score_samples()
        if_scores = self.isolation_forest.score_samples(X_scaled)

        if_anomaly = if_scores < self.if_threshold

        # Autoencoder reconstruction errors
        input_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():
            reconstructed = self.autoencoder(input_tensor)

            ae_errors = torch.mean(
                (input_tensor - reconstructed) ** 2,
                dim=1
            ).numpy()

        ae_anomaly = ae_errors > self.ae_threshold

        # Original anomaly severity calculations
        if_severity = np.clip(
            (self.if_threshold - if_scores) / 0.15,
            0,
            1
        )

        ae_severity = np.clip(
            (ae_errors - self.ae_threshold)
            / (self.ae_threshold * 0.5),
            0,
            1
        )

        # Original hybrid formula
        hybrid_risk = 100.0 * (
            self.weights["xgboost"] * xgb_attack_probability
            + self.weights["random_forest"] * rf_attack_probability
            + self.weights["isolation_forest"] * if_severity
            + self.weights["autoencoder"] * ae_severity
        )

        hybrid_risk = np.clip(hybrid_risk, 0, 100)

        hybrid_prediction = (
            hybrid_risk >= self.alert_threshold
        )

        # Severity categories
        risk_category = np.select(
            [
                hybrid_risk < 25,
                hybrid_risk < 50,
                hybrid_risk < 75
            ],
            [
                "Low",
                "Medium",
                "High"
            ],
            default="Critical"
        )

        # Recommended actions
        recommendations = []

        for category in risk_category:
            if category == "Critical":
                recommendations.append(
                    "Urgent investigation; consider isolating the affected host."
                )
            elif category == "High":
                recommendations.append(
                    "Prioritize investigation and review related network flows."
                )
            elif category == "Medium":
                recommendations.append(
                    "Monitor traffic and investigate suspicious patterns."
                )
            else:
                recommendations.append(
                    "No immediate action; continue routine monitoring."
                )

        # Combined output
        results = pd.DataFrame({
            "RF_Prediction": rf_predictions,
            "RF_Attack_Probability": rf_attack_probability,
            "XGB_Prediction": xgb_predictions,
            "XGB_Attack_Probability": xgb_attack_probability,
            "IF_Score": if_scores,
            "IF_Anomaly": if_anomaly,
            "AE_Reconstruction_Error": ae_errors,
            "AE_Anomaly": ae_anomaly,
            "IF_Severity": if_severity,
            "AE_Severity": ae_severity,
            "Hybrid_Risk_Score": hybrid_risk,
            "Hybrid_Prediction": hybrid_prediction,
            "Risk_Category": risk_category,
            "Recommended_Action": recommendations
        })

        return results
