import os

import airflow
from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import pandas as pd
from mlproject import data
from mlproject import model
from mlproject import metrics

def _prepare_data(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    train_data = pd.read_csv(os.path.join(input_dir, "data.csv"))    
    train_data = data.Data(train_data)
    
    # небольшой препроцессинг
    train_data.std() 
    train_data.new_features()
    
    train_data.data.to_csv(os.path.join(output_dir, "train_data.csv"), index=None)

def _split_data(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))    
    train_data = data.Data(train_data, columns=train_data.columns)
    
    data_train, data_val = train_data.train_test_split()
    
    data_train.data.to_csv(os.path.join(output_dir, "data_train.csv"), index=None)
    data_val.data.to_csv(os.path.join(output_dir, "data_val.csv"), index=None)

def _training(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    train_data = pd.read_csv(os.path.join(input_dir, "data_train.csv"))    
    train_data = data.Data(train_data, columns=train_data.columns)
    
    model_ = model.Model()
    model_.train(train_data)
    model_.save(os.path.join(output_dir, "model.model"))

def _validate(input_dir_data, input_dir_model, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    model_ = model.Model()
    model_.load(os.path.join(input_dir_model, "model.model"))
    data_val = pd.read_csv((os.path.join(input_dir_data, "data_val.csv")))
    data_val = data.Data(data_val, columns=data_val.columns)
    
    preds = model_.predict_score(data_val)
    metrics.plot_roc_curve(data_val.data["threat"], preds, name=os.path.join(output_dir, "roc.png"))
    
with DAG(
    dag_id="2_training",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval="@weekly",
) as dag:

    prepare = PythonOperator(
        task_id="prepare", python_callable=_prepare_data, op_kwargs={'output_dir': "/data/processed/{{ ds }}/",
                                                                     'input_dir': "/data/raw/{{ ds }}/"}
    )

    split = PythonOperator(
        task_id="split", python_callable=_split_data, op_kwargs={'output_dir': "/data/splitted/{{ ds }}/",
                                                                 'input_dir': "/data/processed/{{ ds }}/"}
    )

    train = PythonOperator(
        task_id="train", python_callable=_training, op_kwargs={'output_dir': "/data/models/{{ ds }}/",
                                                               'input_dir': "/data/splitted/{{ ds }}/"}
    )

    validate = PythonOperator(
        task_id="validate", python_callable=_validate, op_kwargs={'output_dir': "/data/metrics/{{ ds }}/",
                                                                  'input_dir_data': "/data/splitted/{{ ds }}/",
                                                                  'input_dir_model': "/data/models/{{ ds }}/"}
    )

    
    prepare >> split >> train >> validate
