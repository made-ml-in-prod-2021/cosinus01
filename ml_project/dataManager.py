"""data manager"""
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()

class DataCreate:
    """data with no good labels. just for test"""
    def __init__(self, n_samples):
        self.samples = n_samples
    def create(self, r_range=8000, v_range=8):
        data = np.random.randn(self.samples, 7)
        data[:, 0:3] = data[:, 0:3] * r_range
        data[:, 3:6] = data[:, 3:6] * v_range
        data[:, 6] = np.random.randint(2, size=self.samples)
        data = Data(data, columns=["X", "Y", "Z", "dX", "dY" ,"dZ", "threat"])
        return data


class Data:
    """main dataclass"""
    def __init__(self, data=None, datapath=None, columns=["X", "Y", "Z", "dX", "dY" ,"dZ", "threat"]):
        if data is not None:
            self.data = pd.DataFrame(data, columns=columns)
        elif datapath:
            self.data = pd.read_csv(datapath)
        else:
            self.data = pd.DataFrame([], columns=columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def get_label(self):
        return self.data["threat"]

    def std(self, columns=["X", "Y", "Z", "dX", "dY" ,"dZ"]):
        std = StandardScaler()
        data = std.fit_transform(self.data[columns])
        self.data[columns] = data

    def add_transformed(self, rule, name):
        """add new column from X, Y, Z, dX, dY, dZ with rule"""
        try:
            if rule:
                rule = eval("lambda X, Y, Z, dX, dY, dZ: " + rule) # Не очень хорошая затея! но пока так
                self.data[name] = rule(self.data["X"], self.data["Y"], self.data["Z"], 
                                       self.data["dX"], self.data["dY"], self.data["dZ"])
        except:
            logger.exception("Wrong rule!")
    def augment(self):
        """simple augmentation"""
        if len(self.data.columns) > 7:
            logger.exception("augment shoulde be before adding new columns!")
            return
        data = self.data.copy()
        data[["X", "Y", "Z", "dX", "dY" ,"dZ"]] = - data[["X", "Y", "Z", "dX", "dY" ,"dZ"]]
        self.data = self.data.append(data)
        self.data.index = range(len(self.data))

    def train_test_split(self, test_size=0.1):
        X_train, X_test = train_test_split(self.data, test_size=test_size)
        return Data(X_train, columns=self.data.columns), Data(X_test, columns=self.data.columns)
