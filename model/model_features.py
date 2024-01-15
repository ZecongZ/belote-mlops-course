# -*- coding: utf-8 -*-
"""
We need to convert the raw features to a proper format,
so that it is ingestable by the ML Model
"""
import os
from typing import Dict
import multiprocess as mp
import pandas as pd
import requests


ENV = os.getenv("MLFLOW_ENV", "development")
BASE_PATH = os.getenv("MLOPS_BELOTE_BASE_PATH", "/belote")
GS_BUCKET = os.getenv("GCS_BUCKET", "mlopsbelote")

PATH_TO_SYNTHETIC_DATA = f"gs://{GS_BUCKET}/{ENV}/synthetic_data_contract.csv"
PATH_TO_FEATURE_STORE = f"gs://{GS_BUCKET}/{ENV}/feature_store.csv"


FEATURE_CONVERTER_ENDPOINT = "http://34.77.247.189:1337/cards_to_features"


def read_data() -> pd.DataFrame:
    """Base IO"""
    data = pd.read_csv(PATH_TO_SYNTHETIC_DATA)
    return data


def clean_feature(hand: str, return_as_scalar: bool = True) -> Dict[str, int]:
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


class CustomFeaturesBuilder:
    """
    Interface to centralize all new features creation
    """

    def __init__(self, synthetic_df: pd.DataFrame, features_df: pd.DataFrame):
        self.synthetic_df = synthetic_df
        self.features_df = features_df
        self.merged_df = pd.DataFrame()

    def balanced_resampling(self) -> "CustomFeaturesBuilder":
        """
        Ensure the common case of contract failure don't overshadow everything
        """
        synthetic_game_data_unbalanced = self.synthetic_df[
            ((self.synthetic_df.reward >= 160) & (self.synthetic_df.reward <= 170))
            | ((self.synthetic_df.reward >= -170) & (self.synthetic_df.reward <= -160))
        ]
        synthetic_game_data_balanced = self.synthetic_df[
            ~((self.synthetic_df.reward >= 160) & (self.synthetic_df.reward <= 170))
            & ~((self.synthetic_df.reward >= -170) & (self.synthetic_df.reward <= -160))
        ]

        synthetic_game_data_rebalanced = synthetic_game_data_unbalanced.sample(
            frac=0.99
        )
        self.synthetic_df = pd.concat(
            [synthetic_game_data_balanced, synthetic_game_data_rebalanced]
        )
        return self

    def feature_total_br_points(self) -> "CustomFeaturesBuilder":
        """
        Sum of all the Belote Rebelote declaration points
        """
        self.features_df["total_BR_points"] = (
            self.features_df["has_BR_at_clubs"]
            + self.features_df["has_BR_at_diamonds"]
            + self.features_df["has_BR_at_hearts"]
            + self.features_df["has_BR_at_spades"]
        )
        return self

    def feature_total_tierce_points(self) -> "CustomFeaturesBuilder":
        """
        Sum of all the tierce announced points
        """
        self.features_df["total_tierce_points"] = (
            self.features_df["has_tierce_at_clubs"]
            + self.features_df["has_tierce_at_diamonds"]
            + self.features_df["has_tierce_at_hearts"]
            + self.features_df["has_tierce_at_spades"]
        )
        return self

    def feature_total_and_points(self) -> "CustomFeaturesBuilder":
        """
        Sum of the announced & declared points
        """
        self.features_df["total_AnD_points"] = (
            self.features_df["total_BR_points"]
            + self.features_df["total_tierce_points"]
        )
        return self

    def merge_synthetic_and_features(self) -> "CustomFeaturesBuilder":
        """
        Merge base set of features with newly build ones
        """
        self.merged_df = pd.concat(
            [
                self.synthetic_df[
                    [
                        "contract",
                        "reward",
                        "last_bidder",
                        "starter",
                        "p1_face_value",
                        "p2_face_value",
                    ]
                ].reset_index(drop=True),
                self.features_df.reset_index(drop=True),
            ],
            axis=1,
        )
        return self

    def encode_contract(self) -> "CustomFeaturesBuilder":
        """
        One hot encoding of the contract
        """
        self.merged_df = pd.concat(
            [self.merged_df, pd.get_dummies(self.merged_df.contract)], axis=1
        )
        return self

    def categorize_reward(self) -> "CustomFeaturesBuilder":
        """
        Turn the rewards into a binary stating whether P1 has won
        """
        self.merged_df["p1_has_won"] = self.merged_df["reward"].apply(
            lambda reward: 1 if reward > 0 else 0
        )
        return self


def run_preprocessing_pipeline(
    synthetic_df: pd.DataFrame, features_df: pd.DataFrame
) -> pd.DataFrame:
    """Leverage multiprocessing to speed up feature cleaning"""
    with mp.Pool(5) as mpool:
        output = mpool.map(clean_feature, synthetic_df["raw_features"].values)
    features_df = pd.concat([features_df, pd.DataFrame(output)])
    return features_df



# @timing
# def multiprocessing_requests() -> pd.DataFrame:
#     features_df = base_features_df.copy()
#     with mp.Pool(5) as p: # set up a pool of 5 workers
#         output = p.map(clean_feature, synthetic_game_data["raw_features"].values[0:SAMPLE]) # launch the cleaning
#     features_df = pd.concat( # merge the placeholder with the clean features
#         [features_df, pd.DataFrame(output)]
#     )
#     return features_df
#
# features_df = multiprocessing_requests()



def run_processing_pipeline(
    custom_feature_builder: CustomFeaturesBuilder,
) -> pd.DataFrame:
    """Run end to end custom feature building"""
    return (
        custom_feature_builder.balanced_resampling()
        .feature_total_br_points()
        .feature_total_tierce_points()
        .feature_total_and_points()
        .merge_synthetic_and_features()
        .encode_contract()
        .categorize_reward()
        .merged_df
    )


if __name__ == "__main__":
    synthetic_game_data = read_data()
    intermediate = run_preprocessing_pipeline(synthetic_game_data, pd.DataFrame())
    feature_store = run_processing_pipeline(
        CustomFeaturesBuilder(synthetic_game_data, intermediate)
    )
    feature_store.to_csv(PATH_TO_FEATURE_STORE, index=False)
