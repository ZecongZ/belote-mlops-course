# -*- coding: utf-8 -*-
"""
E2E pipeline to evaluate whether our production model is suffering from model drift
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from google.cloud import storage

from mlops.model.model_features import (
    CustomFeaturesBuilder,
    run_preprocessing_pipeline,
    run_processing_pipeline,
)


ENV = os.getenv("MLFLOW_ENV", "development")
GS_BUCKET = os.getenv("GCS_BUCKET", "mlopsbelote")

SYNTHETIC_DATA_CONTRACT_PATH = f"gs://{GS_BUCKET}/{ENV}/synthetic_data_contract.csv"
FEATURE_STORE_PATH = f"gs://{GS_BUCKET}/{ENV}/feature_store_drift.csv"

PRODUCTION_MODEL_NAME = (
    "development/0/0b10db7f4cea44b986d769436f43b9d9/artifacts/game_winner/model.pkl"
)

COVARIATES = [
    "last_bidder",
    "starter",
    "p1_face_value",
    "p2_face_value",
    "has_x_cards_in_suit_clubs",
    "has_x_cards_in_suit_diamonds",
    "has_x_cards_in_suit_hearts",
    "has_x_cards_in_suit_spades",
    "has_x_sevens",
    "has_x_eights",
    "has_x_nines",
    "has_x_tens",
    "has_x_jacks",
    "has_x_queens",
    "has_x_kings",
    "has_x_aces",
    "has_BR_at_clubs",
    "has_BR_at_diamonds",
    "has_BR_at_hearts",
    "has_BR_at_spades",
    "has_tierce_at_clubs",
    "has_tierce_at_diamonds",
    "has_tierce_at_hearts",
    "has_tierce_at_spades",
    "total_BR_points",
    "total_tierce_points",
    "total_AnD_points",
    "clubs",
    "diamonds",
    "hearts",
    "sans_atouts",
    "spades",
    "tout_atouts",
]

BASE_THRESHOLD = 0.5
TARGET = "p1_has_won"

BASE_ROC_SCORE = 0.78
ACCEPTABLE_DRIFT = 0.05


def _get_production_model(bucket_name: str, model_name: str):
    """
    Retrieve production model from google cloud storage
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob_model = bucket.blob(model_name)
    with blob_model.open("rb") as handle:
        production_model = pickle.load(handle)
    return production_model


def generate_feature_store(
    synthetic_data_path: str = SYNTHETIC_DATA_CONTRACT_PATH,
    feature_store_path: str = FEATURE_STORE_PATH,
    base_df: pd.DataFrame = pd.DataFrame(),
) -> None:
    """
    Run the feature engineering pipeline
    """
    new_synthetic_data = pd.read_csv(synthetic_data_path, nrows=1000)
    intermediate = run_preprocessing_pipeline(new_synthetic_data, base_df)
    feature_store_drift = run_processing_pipeline(
        CustomFeaturesBuilder(new_synthetic_data, intermediate)
    )
    feature_store_drift.to_csv(feature_store_path, index=False)


def evaluate_roc(
    bucket_name: str = GS_BUCKET,
    model_name: str = PRODUCTION_MODEL_NAME,
    feature_store_path: str = FEATURE_STORE_PATH,
):
    """
    Run the model development pipeline
    """
    model = _get_production_model(bucket_name, model_name)
    feature_store_drift = pd.read_csv(feature_store_path).dropna()
    predictions = model.predict_proba(feature_store_drift[COVARIATES])[:, 1]
    predicted = np.where(predictions > BASE_THRESHOLD, 1, 0)
    current_roc: float = round(roc_auc_score(feature_store_drift[TARGET], predicted), 3)
    minimum_roc: float = round((BASE_ROC_SCORE - ACCEPTABLE_DRIFT), 3)
    failure = (
        f"Current ROC of {current_roc} is < than minimum required of {minimum_roc}.\n"
        f"The drifting of the model is below the acceptable threshold, re-training is necessary."
    )
    assert current_roc > minimum_roc, failure


if __name__ == "__main__":
    generate_feature_store(
        synthetic_data_path=SYNTHETIC_DATA_CONTRACT_PATH,
        feature_store_path=FEATURE_STORE_PATH,
    )
    evaluate_roc(
        bucket_name=GS_BUCKET,
        model_name=PRODUCTION_MODEL_NAME,
        feature_store_path=FEATURE_STORE_PATH,
    )
