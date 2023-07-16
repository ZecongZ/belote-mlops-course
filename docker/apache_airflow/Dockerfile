FROM python:3.8

RUN mkdir -p /belote

WORKDIR /belote

COPY ../../credentials/belote_credentials.json /credentials/

COPY ../../requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/belote/mlops
ENV AIRFLOW_HOME=/belote/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW_VERSION=2.5.0
ENV PYTHON_VERSION="3.8"
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
ENV GOOGLE_APPLICATION_CREDENTIALS="/credentials/belote_credentials.json"
ENV AIRFLOW_VAR_GOOGLE_APPLICATION_CREDENTIALS="/credentials/belote_credentials.json"

RUN pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

RUN apt-get update && apt-get install nano

CMD airflow standalone
