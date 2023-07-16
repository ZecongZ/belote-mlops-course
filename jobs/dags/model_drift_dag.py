# -*- coding: utf-8 -*-
"""
Drift pipeline scheduling through Airflow DAG
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from mlops.model.model_drift import generate_feature_store, evaluate_roc

with DAG(
    "model_drift",
    default_args={
        "depends_on_past": False,
        "email": ["seanariel.bridge@gmail.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "trigger_rule": "all_success",
    },
    description="A simple tutorial DAG",
    schedule="@weekly",
    start_date=datetime.today(),
    catchup=False,
    tags=["production"],
) as dag:

    feature_store_task = PythonOperator(
        task_id="feature_store_task",
        python_callable=generate_feature_store,
        op_kwargs={},
    )

    evaluate_roc_task = PythonOperator(
        task_id="evaluate_roc_task",
        python_callable=evaluate_roc,
        op_kwargs={},
    )

    feature_store_task >> evaluate_roc_task  # pylint: disable=pointless-statement
