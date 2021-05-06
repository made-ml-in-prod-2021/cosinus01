import pytest
import numpy as np
import os, sys
sys.path.append('../')
import dataManager

def test_data_create():
    samples = 20
    data = dataManager.DataCreate(samples)
    data = data.create()
    assert len(data) == 20, "WRONG LENGTH"
    assert list(data.data.columns) == ["X", "Y", "Z", "dX", "dY" ,"dZ", "threat"], "WRONG COLUMNS"

def test_data_class():
    data = dataManager.Data([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
    assert list(data.get_label()) == [1, 0], "WRONG LABEL"
    data.augment()
    assert sum(np.sum(data.data)) == 2, "WRONG SUM"
    data.add_transformed("2 * X", "2X")
    assert all(data.data["2X"] == 2 * data.data["X"]), "WRONG TRANSFORM" 