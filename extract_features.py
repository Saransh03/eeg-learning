from data_manager import DataManager
from feature_extractor import FeatureExtractor

if __name__ == "__main__":
    dm = DataManager()
    bids_paths = dm.prepare()

    fe = FeatureExtractor()
    ratio_low, ratio_high, groups_low, groups_high = fe.process_all(bids_paths)
    fe.save_features(ratio_low, ratio_high, groups_low, groups_high)
