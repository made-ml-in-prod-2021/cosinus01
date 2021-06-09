import os

import airflow
from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# import pandas as pd
from mlproject import data

def _get_data(output_dir, n_samples=10000):

    newdata = data.DataCreate(n_samples=n_samples)
    newdata = newdata.create()
   
    newdata.data['threat'] = newdata.data['threat'].astype(int)
    os.makedirs(output_dir, exist_ok=True)

    newdata.data.to_csv(os.path.join(output_dir, "data.csv"), index=None)
    # data_target.to_csv(os.path.join(output_dir, "target.csv"), index=None) # В моей структуре лучше хранить всё в одном файле


with DAG(
    dag_id="1_download",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval=None,
) as dag:

    download = PythonOperator(
        task_id="download_1", python_callable=_get_data, op_kwargs={'output_dir': "/data/raw/{{ ds }}/"}
    )

    

    download
