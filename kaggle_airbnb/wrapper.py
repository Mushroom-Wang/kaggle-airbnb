import pickle
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class AlgorithmWrapper(ABC):
    def __init__(self, algo_cls, algo_kwargs: Dict):
        self.algo = algo_cls(**algo_kwargs)

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "AlgorithmWrapper":
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, x, y):
        acc = (self.predict(x) == y).sum() / len(y)
        return acc

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            wrapper = pickle.load(f)
        return wrapper


class SklearnWrapper(AlgorithmWrapper):
    def fit(self, x, y):
        return self.algo.fit(x, y)

    def predict(self, x) -> np.ndarray:
        return self.algo.predict(x)


class LogisticRegressionWrapper(SklearnWrapper):
    def __init__(self, algo_kwargs: Dict):
        super().__init__(LogisticRegression, algo_kwargs)


class AdaBoostWrapper(SklearnWrapper):
    def __init__(self, algo_kwargs: Dict):
        super().__init__(AdaBoostClassifier, algo_kwargs)


class RandomForestWrapper(SklearnWrapper):
    def __init__(self, algo_kwargs: Dict):
        super().__init__(RandomForestClassifier, algo_kwargs)


class CatBoostWrapper(SklearnWrapper):
    def __init__(self, algo_kwargs: Dict):
        super().__init__(CatBoostClassifier, algo_kwargs)


class XGBoostWrapper(AlgorithmWrapper):
    def __init__(self, algo_kwargs: Dict):
        self.algo_kwargs = algo_kwargs

    def fit(self, x, y) -> "XGBoostWrapper":
        train_set = xgb.DMatrix(x, label=y)
        self.algo = xgb.train(self.algo_kwargs,
                              train_set)
        return self

    def predict(self, x) -> np.ndarray:
        x = xgb.DMatrix(x)
        return (self.algo.predict(x) > 0.5).astype(int)
