import statistics

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class Evaluation:

    @classmethod
    def evaluate(cls, model, X, y, sample_weight=None, n_splits=10, shuffle=True):
        fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        scores = []
        for index, [train_index, test_index] in enumerate(fold.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if sample_weight is not None:
                sample_weight_train, sample_weight_test = sample_weight[train_index], sample_weight[test_index]
                model.fit(X_train, y_train, sample_weight_train)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = roc_auc_score(y_test, y_pred)
            print("Fold " + str(index+1) + ": " + str(score))
            scores.append(score)
        print("MEAN: " + str(statistics.mean(scores)))
        print("STDDEV: " + str(statistics.stdev(scores)))
        return scores

    @classmethod
    def score(cls, y_true, y_score):
        return roc_auc_score(y_true, y_score)
