from classifier import WorkloadClassifier

if __name__ == "__main__":
    clf = WorkloadClassifier()
    ratio_low, ratio_high, groups_low, groups_high = clf.load_features()
    X, y, groups = clf.build_dataset(ratio_low, ratio_high, groups_low, groups_high)
    clf.evaluate(X, y, groups)
