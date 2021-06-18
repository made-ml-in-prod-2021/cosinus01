ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.0.0-python3.6"
FROM ${AIRFLOW_BASE_IMAGE}

COPY requirements.txt /home/airflow/.local/lib/python3.6/site-packages/requirements.txt
COPY mlproject-1.0-py2.py3-none-any.whl /home/airflow/.local/lib/python3.6/site-packages/mlproject-1.0-py2.py3-none-any.whl

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==1.2.0
RUN pip install -r /home/airflow/.local/lib/python3.6/site-packages/requirements.txt

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.6/site-packages
