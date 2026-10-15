"""
Feature Engineering — transforms raw ContextSnapshot data into 19-feature ML input.
12 raw features → encoding → 7 derived features → 19 total.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Optional
import pickle


class FeatureEngine:
    """Converts raw event data into numerical features for XGBoost."""

    # Raw feature columns used for encoding
    CATEGORICAL_COLS = ["current_app", "prev_app_1", "prev_app_2", "prev_app_3",
                        "network_type", "app_category"]
    NUMERICAL_COLS = ["hour", "day_of_week", "battery", "ram_free_mb"]
    BOOLEAN_COLS = ["is_charging", "ai_task_active"]

    # All 19 feature names in order
    FEATURE_NAMES = [
        # 12 raw (encoded)
        "current_app", "prev_app_1", "prev_app_2", "prev_app_3",
        "hour", "day_of_week", "battery", "is_charging",
        "ram_free_mb", "network_type", "ai_task_active", "app_category",
        # 7 derived
        "ram_utilization", "is_peak_hour", "is_weekend",
        "session_depth", "time_since_last_switch", "battery_bucket",
        "prev_app_category",
    ]

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngine":
        """Fit label encoders and scaler on training data."""
        # Fit label encoders for categorical columns
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            # Add "unknown" for unseen categories at inference time
            unique_vals = list(df[col].fillna("").astype(str).unique()) + ["__unknown__"]
            le.fit(unique_vals)
            self.label_encoders[col] = le

        # Derive prev_app_category and fit encoder for it
        le_cat = LabelEncoder()
        all_cats = list(df["app_category"].fillna("").astype(str).unique()) + ["__unknown__", ""]
        le_cat.fit(all_cats)
        self.label_encoders["prev_app_category"] = le_cat

        # Fit scaler on the full feature matrix
        X = self.transform(df, fit_mode=True)
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        """Transform raw event DataFrame into numerical feature matrix."""
        n = len(df)
        X = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float32)

        # ── Encode categoricals (columns 0-3, 9, 11) ──
        for i, col in enumerate(self.CATEGORICAL_COLS):
            col_idx = self.FEATURE_NAMES.index(col)
            vals = df[col].fillna("").astype(str)
            if col in self.label_encoders and not fit_mode:
                le = self.label_encoders[col]
                encoded = []
                for v in vals:
                    if v in le.classes_:
                        encoded.append(le.transform([v])[0])
                    else:
                        encoded.append(le.transform(["__unknown__"])[0])
                X[:, col_idx] = encoded
            else:
                X[:, col_idx] = pd.factorize(vals)[0]

        # ── Numerical columns ──
        X[:, 4] = df["hour"].values
        X[:, 5] = df["day_of_week"].values
        X[:, 6] = df["battery"].values
        X[:, 8] = df["ram_free_mb"].values

        # ── Boolean columns ──
        X[:, 7] = df["is_charging"].astype(int).values
        X[:, 10] = df["ai_task_active"].astype(int).values

        # ── 7 Derived features ──
        # ram_utilization
        X[:, 12] = 1.0 - (df["ram_free_mb"].values / df["ram_total_mb"].values.clip(min=1))

        # is_peak_hour
        X[:, 13] = df["hour"].isin([8, 9, 12, 13, 18, 19, 20, 21]).astype(float).values

        # is_weekend
        X[:, 14] = (df["day_of_week"] >= 5).astype(float).values

        # session_depth
        if "session_depth" in df.columns:
            X[:, 15] = df["session_depth"].values
        else:
            X[:, 15] = 0

        # time_since_last_switch
        if "time_since_last_switch" in df.columns:
            X[:, 16] = np.log1p(df["time_since_last_switch"].values)  # log-transform
        else:
            X[:, 16] = 0

        # battery_bucket (0=low <20%, 1=med 20-50%, 2=high >50%)
        battery = df["battery"].values
        X[:, 17] = np.where(battery < 0.2, 0, np.where(battery < 0.5, 1, 2))

        # prev_app_category — map prev_app_1's category
        if "prev_app_category" not in df.columns:
            # Derive from prev_app_1 using app_profiles
            from src.data.app_profiles import get_profile
            prev_cats = df["prev_app_1"].fillna("").apply(lambda a: get_profile(a).category if a else "")
        else:
            prev_cats = df["prev_app_category"].fillna("").astype(str)

        if "prev_app_category" in self.label_encoders and not fit_mode:
            le = self.label_encoders["prev_app_category"]
            encoded = []
            for v in prev_cats:
                if v in le.classes_:
                    encoded.append(le.transform([v])[0])
                else:
                    encoded.append(le.transform(["__unknown__"])[0])
            X[:, 18] = encoded
        else:
            X[:, 18] = pd.factorize(prev_cats)[0]

        # Apply scaler if fitted (not during fit_mode)
        if self.scaler is not None and not fit_mode:
            X = self.scaler.transform(X)

        return X

    def get_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target variable (next_app) as encoded labels."""
        if "target_encoder" not in self.label_encoders:
            le = LabelEncoder()
            le.fit(df["next_app"].astype(str).unique())
            self.label_encoders["target_encoder"] = le

        le = self.label_encoders["target_encoder"]
        return le.transform(df["next_app"].astype(str))

    def decode_target(self, encoded: np.ndarray) -> List[str]:
        """Decode encoded labels back to app names."""
        le = self.label_encoders["target_encoder"]
        return list(le.inverse_transform(encoded))

    @property
    def n_classes(self) -> int:
        if "target_encoder" in self.label_encoders:
            return len(self.label_encoders["target_encoder"].classes_)
        return 0

    @property
    def target_classes(self) -> List[str]:
        if "target_encoder" in self.label_encoders:
            return list(self.label_encoders["target_encoder"].classes_)
        return []

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FeatureEngine":
        with open(path, "rb") as f:
            return pickle.load(f)
