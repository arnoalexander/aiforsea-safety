import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier


class ClassifierUtility:
    
    @classmethod
    def get_sample_weight(cls, y):
        return np.where(y == 1, 2.0, 1.0)


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            self.model = LGBMClassifier()
        else:
            self.model = base_estimator

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        probabilities = self.model.predict_proba(X)
        return np.array([probability[-1] for probability in probabilities])

    def score(self, X, y, sample_weight=None):
        return roc_auc_score(y, self.predict(X), sample_weight=sample_weight)
