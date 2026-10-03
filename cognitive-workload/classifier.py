import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_score

import config


class WorkloadClassifier:
    def __init__(self, kernel='rbf', cv=5):
        self.clf = SVC(kernel=kernel)
        self.cv = cv

    @staticmethod
    def load_features(path=config.FEATURES_FILE):
        data = np.load(path)
        return data['ratio_low'], data['ratio_high'], data['groups_low'], data['groups_high']

    @staticmethod
    def build_dataset(ratio_low, ratio_high, groups_low, groups_high):
        X = np.concatenate([ratio_low, ratio_high]).reshape(-1, 1)
        y = np.array([0]*len(ratio_low) + [1]*len(ratio_high))
        groups = np.concatenate([groups_low, groups_high])
        return X, y, groups

    def evaluate(self, X, y, groups):
        gkf = GroupKFold(n_splits=5)
        scores = cross_val_score(self.clf, X, y, cv=gkf, groups=groups)
        print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
        return scores
