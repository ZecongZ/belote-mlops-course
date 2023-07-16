# -*- coding: utf-8 -*-
"""
Develops an ML Random Forest model to predict the likeliness of winning

mlflow server \
    --backend-store-uri sqlite://///Users/seanariel/Desktop/la-maniee/data/mlflow/mlruns.db \
    --default-artifact-root gs://mlopsbelote/development \
    --host 0.0.0.0:5001

> python mlops/model/model_development.py
> mlflow run mlops/model
"""
import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt


ENV = os.getenv("MLFLOW_ENV", "development")
BASE_PATH = os.getenv("MLOPS_BELOTE_BASE_PATH", "/belote")
GS_BUCKET = os.getenv("GCS_BUCKET", "mlopsbelote")

MODEL_NAME = "game_winner"

PATH_TO_FEATURE_STORE = f"gs://{GS_BUCKET}/{ENV}/feature_store.csv"
PATH_TO_DEV_TRAINING_DATA = f"gs://{GS_BUCKET}/{ENV}/dev_training.csv"
PATH_TO_DEV_TESTING_DATA = f"gs://{GS_BUCKET}/{ENV}/dev_testing.csv"
PATH_TO_PRECISION_RECALL = f"gs://{GS_BUCKET}/{ENV}/precision_recall.csv"
PATH_TO_VISUAL = f"{BASE_PATH}/data/mlops/precision_recall_visual.png"
PATH_TO_CONDA = f"{BASE_PATH}/mlops/model/setup.yaml"


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

BUSINESS_THRESHOLD = 0.65
RANDOM_STATE = 42

with mlflow.start_run(run_name=f"{MODEL_NAME} - Development Run") as run:

    # Set explicit authoring tag for your experiment
    mlflow.set_tag("mlflow.user", os.getenv("USER", "Sean Ariel"))

    # Training - Testing Split
    (
        covariates_training,
        covariates_testing,
        target_training,
        target_testing,
    ) = train_test_split(
        FEATURE_STORE[COVARIATES],
        FEATURE_STORE[TARGET],
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    # Hyperparameter search
    base_model = RandomForestClassifier()
    hyper_parameters_grid = {
        "n_estimators": [500, 5000],
        "max_depth": [5, 50],
        "min_samples_split": [5, 50],
        "min_samples_leaf": [5, 50],
    }
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=hyper_parameters_grid,
        n_iter=10,
        scoring=None,
        n_jobs=-1,
        cv=None,
        verbose=2,
        refit=False,
    )
    random_search.fit(
        covariates_training[COVARIATES], target_training.values.reshape(-1, 1)
    )
    optimal_hyper_parameters = random_search.best_params_

    # Persist the best hyperparameters
    mlflow.log_dict(
        optimal_hyper_parameters,
        "hyper_parameters.json",
    )

    # Training of the model
    base_model = RandomForestClassifier()
    optimal_model = base_model.set_params(**optimal_hyper_parameters)
    optimal_model.fit(
        covariates_training[COVARIATES].values, target_training.values.ravel()
    )

    # Persist training data
    covariates_training.to_csv(PATH_TO_DEV_TRAINING_DATA, index=False)

    # Persist model in registry
    try:
        mlflow.sklearn.log_model(
            optimal_model,
            MODEL_NAME,
            conda_env=PATH_TO_CONDA,
            registered_model_name=MODEL_NAME,
            signature=mlflow.models.signature.infer_signature(
                covariates_training[COVARIATES],
                optimal_model.predict(covariates_training[COVARIATES]),
            ),
        )
    except mlflow.exceptions.MlflowException:
        # We haven't set up the registry yet, skip this
        pass

    # Precision - Recall
    predictions = optimal_model.predict_proba(covariates_testing[COVARIATES])[:, 1]
    covariates_testing["predictions"] = predictions
    covariates_testing["target"] = target_testing
    covariates_testing["predicted"] = covariates_testing["predictions"].apply(
        lambda x: 1 if x > BUSINESS_THRESHOLD else 0
    )

    # Persist testing data
    covariates_testing.to_csv(PATH_TO_DEV_TESTING_DATA, index=False)

    # Generate scoring metrics
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(
        target_testing, predictions
    )
    metrics_df = pd.DataFrame(
        {
            "precision": precision_arr[1:],
            "recall": recall_arr[1:],
            "threshold": threshold_arr,
        }
    )
    metrics_df["threshold"] = metrics_df["threshold"].apply(lambda x: round(x, 2))
    metrics_df.drop_duplicates(subset=["threshold"], keep="first", inplace=True)
    metrics_df.to_csv(PATH_TO_PRECISION_RECALL)

    # Log metric & stats plotting
    mlflow.log_metric("ROC_AUC", roc_auc_score(target_testing, predictions))

    fig, ax = plt.subplots()
    ax.plot(metrics_df["recall"].values, metrics_df["precision"].values, color="purple")
    ax.set_title("Precision-Recall Curve")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    fig.savefig(PATH_TO_VISUAL)
