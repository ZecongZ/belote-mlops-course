# -*- coding: utf-8 -*-
"""
Train an ML Random Forest model to predict the likeliness of winning
"""
import os
import mlflow
import pandas as pd


ENV = os.getenv("MLFLOW_ENV", "development")
BASE_PATH = os.getenv("MLOPS_BELOTE_BASE_PATH", "/belote")
GS_BUCKET = os.getenv("GCS_BUCKET", "mlopsbelote")

MODEL_NAME = "game_winner"

PATH_TO_FEATURE_STORE = f"gs://{GS_BUCKET}/{ENV}/feature_store.csv"
PATH_TO_CONDA = f"{BASE_PATH}/mlops/model/setup.yaml"
PATH_TO_OPTIMAL_MODEL = f"models:/{MODEL_NAME}/Staging"

FEATURE_STORE = pd.read_csv(PATH_TO_FEATURE_STORE)
POTENTIAL_TARGETS = ["reward", "p1_has_won"]
TARGET = "p1_has_won"
SEGMENTS = ["reward", "contract"]
COVARIATES = list(
    filter(
        lambda covariate: covariate not in (POTENTIAL_TARGETS + SEGMENTS),
        FEATURE_STORE.columns,
    )
)


# Load model from mlflow registry
try:
    optimal_model = mlflow.sklearn.load_model(model_uri=PATH_TO_OPTIMAL_MODEL)
except mlflow.exceptions.MlflowException as exc:
    raise ValueError(
        "Optimal Model hasn't been registered yet through the development pipeline"
    ) from exc


training_df = FEATURE_STORE[COVARIATES + [TARGET]]

with mlflow.start_run(run_name=f"{MODEL_NAME} - Production Run") as run:

    # Set explicit authoring tag for your experiment
    mlflow.set_tag("mlflow.user", os.getenv("USER", "Sean Ariel"))

    # Log hyper parameters in registry
    for parameter, value in optimal_model.get_params().items():
        mlflow.log_param(parameter, value)

    # Train the model with the latest training data
    optimal_model.fit(
        training_df[COVARIATES], training_df[TARGET].values.reshape(-1, 1)
    )

    # Persist model in registry
    mlflow.sklearn.log_model(
        optimal_model,
        MODEL_NAME,
        conda_env=PATH_TO_CONDA,
        registered_model_name=MODEL_NAME,
        signature=mlflow.models.signature.infer_signature(
            training_df[COVARIATES], optimal_model.predict(training_df[COVARIATES])
        ),
    )
