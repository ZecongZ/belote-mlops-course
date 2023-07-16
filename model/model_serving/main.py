# -*- coding: utf-8 -*-
"""
gcloud functions deploy belote-model-info \
--gen2 \
--runtime=python38 \
--memory=1GiB \
--region=europe-west1 \
--source=. \
--entry-point=get_model_info \
--trigger-http \
--allow-unauthenticated

gcloud functions delete belote-model-info --gen2 --region europe-west1

gcloud functions deploy belote-model-serving \
--gen2 \
--runtime=python38 \
--memory=1GiB \
--region=europe-west1 \
--source=. \
--entry-point=serve_model \
--trigger-http \
--allow-unauthenticated

gcloud functions delete belote-model-serving --gen2 --region europe-west1
"""
import pickle
from typing import Dict
import pandas as pd
import functions_framework
from google.cloud import storage
from flask import jsonify
import requests


FEATURE_CONVERTER_ENDPOINT = "http://34.77.247.189:1337/cards_to_features"
BUCKET_NAME = "mlopsbelote"
PRODUCTION_MODEL_NAME = (
    "development/0/0b10db7f4cea44b986d769436f43b9d9/artifacts/game_winner/model.pkl"
)
MODEL_NAME = "game_winner"
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


def _get_production_model():
    """
    Retrieve production model from google cloud storage
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob_model = bucket.blob(PRODUCTION_MODEL_NAME)
    with blob_model.open("rb") as handle:
        production_model = pickle.load(handle)
    return production_model


@functions_framework.http
def get_model_info(request):  # pylint: disable=unused-argument
    """
    In: -
    Out:
    RandomForestClassifier(max_depth=5, min_samples_leaf=5, min_samples_split=5, n_estimators=10000)
    """
    production_model = _get_production_model()
    return str(production_model)


def _clean_feature(hand: str, return_as_scalar: bool = True) -> Dict[str, int]:
    """
    In:
    {
        "raw_hand": "8H.KC.QH.9D.QC.TC.7H.QD.AH.JD.TH.TS"
    }
    Out:
        has_3_cards_in_suit_clubs
        has_2_cards_in_suit_diamonds
        has_5_cards_in_suit_hearts
        has_1_cards_in_suit_spades
        [...]
    """
    response = requests.post(
        FEATURE_CONVERTER_ENDPOINT,
        json={"raw_hand": hand, "return_as_scalar": return_as_scalar},
    )
    features: Dict = response.json()
    return features


def _build_extra_features(features_dict: Dict) -> Dict:
    features_dict["total_BR_points"] = (
        features_dict["has_BR_at_clubs"]
        + features_dict["has_BR_at_diamonds"]
        + features_dict["has_BR_at_hearts"]
        + features_dict["has_BR_at_spades"]
    )
    features_dict["total_tierce_points"] = (
        features_dict["has_tierce_at_clubs"]
        + features_dict["has_tierce_at_diamonds"]
        + features_dict["has_tierce_at_hearts"]
        + features_dict["has_tierce_at_spades"]
    )
    features_dict["total_AnD_points"] = (
        features_dict["total_BR_points"] + features_dict["total_tierce_points"]
    )
    return features_dict


@functions_framework.http
def serve_model(request):
    """
    In:
    {
        'raw_features': 'AS.QS.TS.9S.TD.AD.KD.TC.9C.JC.KH.8S',
        "last_bidder": 1,
        "starter": 1,
        "clubs": 0,
        "diamonds": 0,
        "hearts": 0,
        "spades": 1,
        "sans_atouts": 0,
        "tout_atouts": 0,
        "p1_face_value": 79,
        "p2_face_value": 24
    }
    Out:
    {
        "prediction": {
            "win_probability": [
                0.61
            ]
        }
    }
    """
    data: Dict = request.json
    raw_features: str = data.get("raw_features")
    other_info: Dict = {k: v for k, v in data.items() if k != "raw_features"}
    processed_features = _clean_feature(raw_features, return_as_scalar=True)
    processed_features = _build_extra_features(processed_features)
    processed_features = {**other_info, **processed_features}
    processed_features = {k: [v] for k, v in processed_features.items()}
    features_df = pd.DataFrame()
    features_df = pd.concat([features_df, pd.DataFrame(processed_features)])
    production_model = _get_production_model()
    predictions = production_model.predict_proba(features_df[COVARIATES])[:, 1]
    return jsonify({"prediction": {"win_probability": predictions.tolist()}})
