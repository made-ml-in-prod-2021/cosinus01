import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import yaml
import main

def test_main():
    with open("../configs/data_links.yml", 'r') as stream:
        datalinks = yaml.safe_load(stream)
    with open("../configs/train_config_0.yml", 'r') as stream:
        trainConfig = yaml.safe_load(stream)
    main.train_test_pipeline_threat(trainConfig, datalinks, download_path=None, n_samples=1000,
                   proxy=None, model_path="../models/test.model", image_path="../images/auc_test.png")