import pytest
import numpy as np
import os, sys
sys.path.append('../')
import metrics

def test_roc_auc():
    metrics.plot_roc_curve([0, 1], [0, 1])