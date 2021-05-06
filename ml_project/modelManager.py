import pickle
from dataManager import Data
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import logging

logger = logging.getLogger()

class Model:
    def __init__(self, model="LR", params={}):
        self.params = params
        if model == "LR" or model == "logisticRegression":
            self.model_type = "LR"
            self.model = LogisticRegression(**params)
        elif model == "lgbm":
            self.model_type = "LGBM"
    def train(self, data: Data, target="threat"):
        """train model"""
        X_train = data.data.drop(columns=[target])
        y_train = data.data[target]
        if self.model_type == "LR":
            self.model.fit(X=X_train, y=y_train)
        elif self.model_type == "LGBM":
            dtrain = lgb.Dataset(X_train, label=y_train)
            num_trees = 100
            self.model = lgb.train(self.params.copy(), dtrain, num_boost_round=num_trees)
    def predict_score(self, data: Data, target="threat", proba=True):
        """predict"""
        X_val = data.data.drop(columns=[target])
        if self.model_type == "LR":
            preds = self.model.predict_proba(X_val)[:, 1]
        elif self.model_type == "LGBM":
            preds = self.model.predict(X_val)
        return preds
    def save(self, name):
        """save model"""
        pickle.dump(self.model, open(name, 'wb'))