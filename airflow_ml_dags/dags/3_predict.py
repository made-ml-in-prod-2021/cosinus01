import os

import airflow
from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pandas as pd
from mlproject import data
from mlproject import model

input_dir_model = Variable.get("modelpath", default_var="/data/models/{{ ds }}/")

def _inference(input_dir_data, input_dir_model, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    model_ = model.Model()
    model_.load(os.path.join(input_dir_model, "model.model"))
    data_pred = pd.read_csv((os.path.join(input_dir_data, "data.csv")))
    data_pred = data.Data(data_pred, columns=data_pred.columns)

    data_pred.std() 
    data_pred.new_features()
    
    preds = model_.predict_score(data_pred)
    preds = pd.DataFrame(preds, columns=["preds"])
    preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=None)
    
with DAG(
    dag_id="3_inference",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval="@daily",
) as dag:

    
    inference = PythonOperator(
        task_id="inference", python_callable=_inference, op_kwargs={'output_dir': "/data/predictions/{{ ds }}/",
                                                                  'input_dir_data': "/data/raw/{{ ds }}/",
                                                                  'input_dir_model': input_dir_model}
    )

    
    inference
