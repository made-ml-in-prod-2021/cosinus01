import pytest
import numpy as np
import os, sys
sys.path.append('../')
import dataManager
import modelManager

def test_train():
    data = data = dataManager.DataCreate(50)
    data = data.create()
    model = modelManager.Model()
    model.train(data)

def test_predict():
    data = data = dataManager.DataCreate(50)
    data = data.create()
    model = modelManager.Model()
    model.train(data)
    preds = model.predict_score(data)