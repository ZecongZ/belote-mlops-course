# -*- coding: utf-8 -*-
"""
gcloud dataproc clusters create belote \
    --properties 'yarn:yarn.resourcemanager.webapp.methods-allowed=ALL' \
    --enable-component-gateway \
    --region europe-west1 \
    --zone europe-west1-b \
    --master-machine-type n1-standard-4 \
    --master-boot-disk-size 500 \
    --num-workers 2 \
    --worker-machine-type n1-standard-4 \
    --worker-boot-disk-size 500 \
    --image-version 2.0-debian10 \
    --project bionic-ventures

gcloud dataproc jobs submit pyspark --region=europe-west1 --cluster=belote model_spark.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

import pandas as pd


ENV = os.getenv("MLFLOW_ENV", "development")
BASE_PATH = os.getenv("MLOPS_BELOTE_BASE_PATH", "/belote")
GS_BUCKET = os.getenv("GCS_BUCKET", "mlopsbelote")

PATH_TO_FEATURE_STORE = f"gs://{GS_BUCKET}/{ENV}/feature_store.csv"
PATH_SPARK_RF = "gs://mlopsbelote/spark_rf"

SPARK = SparkSession.builder.getOrCreate()

FEATURE_STORE = pd.read_csv(PATH_TO_FEATURE_STORE)
POTENTIAL_TARGETS = ["reward", "p1_has_won"]
TARGET = "p1_has_won"
SEGMENTS = ["reward", "contract"]
COVARIATES = list(
    filter(
        lambda covariate: covariate not in SEGMENTS,
        FEATURE_STORE.columns,
    )
)
SCHEMA = StructType(
    [StructField("contract", StringType(), True)]
    + [StructField(column, DoubleType(), True) for column in COVARIATES]
)

feature_df = (
    SPARK.read.format("csv")
    .option("header", True)
    .schema(SCHEMA)
    .load(PATH_TO_FEATURE_STORE)
)

assembler = VectorAssembler(inputCols=COVARIATES, outputCol="features")
ml_df = assembler.transform(feature_df)
rf_classifier = RandomForestClassifier(
    featuresCol="features", labelCol=TARGET, numTrees=10000
)
rf_model = rf_classifier.fit(ml_df)
rf_model.save(PATH_SPARK_RF)
