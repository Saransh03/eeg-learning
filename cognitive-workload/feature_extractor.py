import numpy as np
import mne
from mne_bids import read_raw_bids
from mne.time_frequency import psd_array_welch

import config


class FeatureExtractor:
    def __init__(self,
                 frontal_channels=config.FRONTAL_CHANNELS,
                 parietal_channels=config.PARIETAL_CHANNELS,
                 theta_band=config.THETA_BAND,
                 alpha_band=config.ALPHA_BAND):
        self.frontal_channels = frontal_channels
        self.parietal_channels = parietal_channels
        self.theta_band = theta_band
        self.alpha_band = alpha_band

    @staticmethod
    def _band_power(data, sfreq, band):
        psds, _ = psd_array_welch(data, sfreq, fmin=band[0], fmax=band[1])
        return psds.mean(-1)

    def _load_and_preprocess(self, bids_path):
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        raw.filter(1., 40.)
        raw.set_eeg_reference('average')
        return raw

    def _epoch(self, raw):
        events, event_id = mne.events_from_annotations(raw)

        low_keys = [k for k in event_id if 'memory' in k and 'in 5 digit sequence' in k]
        high_keys = [k for k in event_id if 'memory' in k and 'in 13 digit sequence' in k]

        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0,
                            baseline=(None, 0), preload=True)
        return epochs, low_keys, high_keys

    def _channel_indices(self, raw):
        frontal = [ch for ch in self.frontal_channels if ch in raw.ch_names]
        parietal = [ch for ch in self.parietal_channels if ch in raw.ch_names]
        f_idx = [raw.ch_names.index(c) for c in frontal]
        p_idx = [raw.ch_names.index(c) for c in parietal]
        return f_idx, p_idx

    def extract_ratio(self, epochs, raw, f_idx, p_idx):
        sfreq = raw.info['sfreq']
        theta = self._band_power(epochs.get_data(), sfreq, self.theta_band)[:, f_idx].mean(1)
        alpha = self._band_power(epochs.get_data(), sfreq, self.alpha_band)[:, p_idx].mean(1)
        return theta / alpha

    def process_subject(self, bids_path):
        raw = self._load_and_preprocess(bids_path)
        epochs, low_keys, high_keys = self._epoch(raw)
        f_idx, p_idx = self._channel_indices(raw)

        low = epochs[low_keys]
        high = epochs[high_keys]

        ratio_low = self.extract_ratio(low, raw, f_idx, p_idx)
        ratio_high = self.extract_ratio(high, raw, f_idx, p_idx)
        return ratio_low, ratio_high

    def process_all(self, bids_paths):
        all_ratio_low, all_ratio_high, subject_groups_low, subject_groups_high = [], [], [], []
        for i, bp in enumerate(bids_paths):
            ratio_low, ratio_high = self.process_subject(bp)
            all_ratio_low.append(ratio_low)
            all_ratio_high.append(ratio_high)
            subject_groups_low.append(np.full(len(ratio_low), i))
            subject_groups_high.append(np.full(len(ratio_high), i))
        return (np.concatenate(all_ratio_low), np.concatenate(all_ratio_high),
                np.concatenate(subject_groups_low), np.concatenate(subject_groups_high))

    def save_features(self, ratio_low, ratio_high, groups_low, groups_high, path=config.FEATURES_FILE):
        np.savez(path, ratio_low=ratio_low, ratio_high=ratio_high,
                groups_low=groups_low, groups_high=groups_high)
