"""Stage 2 price prediction service using trained ML model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from backend.schemas.stage_1 import Stage1ExtractedFeatures
from backend.schemas.stage_2 import PredictionResponse


def _get_model_path() -> Path:
    """Return the absolute path to the trained model artifact."""
    return Path(__file__).resolve().parent.parent / 'artifacts' / 'ames_houses_price_predictor.joblib'


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the trained model from joblib artifact."""
    model_path = _get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_price(features: Stage1ExtractedFeatures) -> PredictionResponse:
    """Predict house sale price from extracted features.

    Args:
        features: Stage 1 extracted features (12-feature vector).

    Returns:
        PredictionResponse with predicted_price.

    Raises:
        ValueError: If required features are missing.
        FileNotFoundError: If model artifact not found.
    """
    # Define the expected feature order to match model training.
    feature_order = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "FullBath",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "Fireplaces",
        "BsmtFinSF1",
        "LotFrontage",
        "1stFlrSF",
        "OpenPorchSF",
    ]

    # Extract feature values in the expected order; raise if any required field is missing.
    feature_data = features.model_dump(by_alias=True)
    feature_values = []
    for field_name in feature_order:
        value = feature_data.get(field_name)
        if value is None:
            raise ValueError(f"Required feature '{field_name}' is missing or None. All 12 features required for prediction.")
        feature_values.append(value)

    # Load the cached model.
    model = _load_model()

    # Predict using a single-row dataframe because the pipeline was trained on named columns.
    feature_frame = pd.DataFrame([dict(zip(feature_order, feature_values))])
    predicted_price = float(model.predict(feature_frame)[0])

    if predicted_price <= 0:
        raise ValueError(f"Model returned invalid price: {predicted_price}. Expected positive value.")

    return PredictionResponse(predicted_price=predicted_price)
