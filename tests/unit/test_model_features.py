# -*- coding: utf-8 -*-
"""
Unit Testing suite for model/model_features
Run the tests:
Terminal    => pytest --cov=./mlops mlops/tests/ -v
HTML Report => pytest --cov-report html:/belote/mlops/htmlcov --cov=./mlops mlops/tests/ -v
"""
from typing import Dict, List
import pytest
import pandas as pd

from mlops.preparation.cleaning import (
    process_stringified_features,
    explode_feature,
)
from mlops.model.model_features import CustomFeaturesBuilder


@pytest.fixture
def raw_feature() -> str:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return (
        "{'feature_belote_rebelote_points': array([0, 0, 0, 0]), "
        "'feature_count_of_cards': array([1, 1, 2, 2, 1, 2, 1, 2]), "
        "'feature_count_of_suit': array([3, 2, 4, 3]), "
        "'feature_tierce_plus_points': array([0, 0, 0, 0])}"
    )


@pytest.fixture
def stringified_feature() -> Dict[
    str, List[int]
]:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return {
        "feature_belote_rebelote_points": [0, 0, 0, 0],
        "feature_count_of_cards": [1, 1, 2, 2, 1, 2, 1, 2],
        "feature_count_of_suit": [3, 2, 4, 3],
        "feature_tierce_plus_points": [0, 0, 0, 0],
    }


@pytest.fixture
def exploded_feature() -> Dict[
    str, int
]:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return {
        "has_x_cards_in_suit_clubs": 3,
        "has_x_cards_in_suit_diamonds": 2,
        "has_x_cards_in_suit_hearts": 4,
        "has_x_cards_in_suit_spades": 3,
        "has_x_sevens": 1,
        "has_x_eights": 1,
        "has_x_nines": 2,
        "has_x_tens": 2,
        "has_x_jacks": 1,
        "has_x_queens": 2,
        "has_x_kings": 1,
        "has_x_aces": 2,
        "has_BR_at_clubs": 0,
        "has_BR_at_diamonds": 0,
        "has_BR_at_hearts": 0,
        "has_BR_at_spades": 0,
        "has_tierce_at_clubs": 0,
        "has_tierce_at_diamonds": 0,
        "has_tierce_at_hearts": 0,
        "has_tierce_at_spades": 0,
    }


@pytest.fixture
def custom_feature_builder():  # pylint: disable=redefined-outer-name, missing-function-docstring
    return CustomFeaturesBuilder


@pytest.fixture
def synthetic_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return pd.DataFrame(
        {
            "reward": [1, 2, 3],
            "last_bidder": [1, 2, 1],
            "starter": [2, 1, 2],
            "contract": ["hearts", "spades", "diamonds"],
            "p1_face_value": [10, 20, 30],
            "p2_face_value": [20, 30, 40],
        }
    )


@pytest.fixture
def features_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return pd.DataFrame(
        {
            "has_BR_at_clubs": [1, 2, 3],
            "has_BR_at_diamonds": [1, 2, 3],
            "has_BR_at_hearts": [1, 2, 3],
            "has_BR_at_spades": [1, 2, 3],
            "has_tierce_at_clubs": [2, 3, 4],
            "has_tierce_at_diamonds": [2, 3, 4],
            "has_tierce_at_hearts": [2, 3, 4],
            "has_tierce_at_spades": [2, 3, 4],
            "total_BR_points": [4, 8, 12],
            "total_tierce_points": [8, 12, 16],
            "total_AnD_points": [12, 20, 28],
        }
    )


def test_stringified_feature_works_correctly(  # pylint: disable=redefined-outer-name
    raw_feature, stringified_feature
):
    """NA"""
    processed_output = process_stringified_features(raw_feature)
    assert processed_output == stringified_feature


def test_explode_feature_with_raw_feature_works_correctly(  # pylint: disable=redefined-outer-name
    mocker, raw_feature, stringified_feature, exploded_feature
):
    """NA"""
    mocker_stringified = mocker.patch(
        "mlops.preparation.cleaning.process_stringified_features"
    )
    mocker_stringified.return_value = stringified_feature
    processed_output = explode_feature(raw_feature, return_as_scalar=True)
    mocker_stringified.assert_called_once_with(raw_feature)
    assert processed_output == exploded_feature


def test_explode_feature_with_stringified_feature_works_correctly(  # pylint: disable=redefined-outer-name
    stringified_feature, exploded_feature
):
    """NA"""
    processed_output = explode_feature(stringified_feature, return_as_scalar=True)
    assert processed_output == exploded_feature


def balanced_resampling_works_correctly():
    """NA"""


def test_custom_feature_total_br_points_works_correctly(  # pylint: disable=redefined-outer-name
    mocker, custom_feature_builder, features_df
):
    """NA"""
    intermediate_df = features_df.drop(
        ["total_BR_points", "total_tierce_points", "total_AnD_points"], axis=1
    )
    features_df = features_df.drop(["total_tierce_points", "total_AnD_points"], axis=1)
    processed_output = (
        custom_feature_builder(mocker.MagicMock(), intermediate_df)
        .feature_total_br_points()
        .features_df
    )
    pd.testing.assert_frame_equal(processed_output, features_df)


def test_custom_feature_total_tierce_points_works_correctly():
    """NA"""


def test_custom_feature_total_and_points_works_correctly():
    """NA"""


def test_custom_feature_merge_synthetic_and_features_works_correctly():
    """NA"""


def test_custom_feature_encode_contract_works_correctly():
    """NA"""


def test_custom_feature_categorize_reward_works_correctly():
    """NA"""
