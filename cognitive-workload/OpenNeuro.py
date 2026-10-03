import os
import pandas as pd
import openneuro
import mne
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

DATASET = "ds003838"   # swap if this one stays broken
ROOT = "./eeg_data"

def ensure_downloaded(dataset, root, include):
    missing = []
    for f in include:
        path = os.path.join(root, f)
        if f.endswith("/eeg"):
            # folder-level include: verify at least one .set exists inside
            set_files = [x for x in os.listdir(path) if x.endswith(".set")] if os.path.exists(path) else []
            if not set_files:
                missing.append(f)
        else:
            if not os.path.exists(path):
                missing.append(f)

    if missing:
        print(f"Downloading missing: {missing}")
        openneuro.download(dataset=dataset, target_dir=root, include=missing)
    else:
        print("All required files already present, skipping download.")

# --- Step 1: metadata only, confirm structure ---
os.makedirs(ROOT, exist_ok=True)
ensure_downloaded(DATASET, ROOT, ["dataset_description.json", "participants.tsv"])

participants = pd.read_csv(f"{ROOT}/participants.tsv", sep="\t")
valid = participants[participants['EEG_excluded'] == 'no']
subject_ids = valid['participant_id'].str.replace('sub-', '').tolist()[:5]
print(subject_ids)

# --- Step 2: download actual EEG for those confirmed subjects ---
includes = [f"sub-{s}/eeg" for s in subject_ids]
ensure_downloaded(DATASET, ROOT, includes)   # <-- use includes here, not metadata list again

# --- Step 3: find what mne-bids actually sees ---
paths = find_matching_paths(ROOT, datatypes="eeg", suffixes="eeg",
                             tasks="memory", extensions=".set")
print(paths)

# --- Step 4: load, preprocess, epoch per subject ---
all_ratio_low, all_ratio_high, all_labels = [], [], []

for bp in paths:
    #Filter (1–40 Hz)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    raw.filter(1., 40.)
    raw.set_eeg_reference('average')

    events, event_id = mne.events_from_annotations(raw)
    print(event_id)  # inspect actual condition labels before hardcoding

    # Cuts continuous signal into event-locked windows so you're comparing the same relative time window across trials, not arbitrary segments
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0,
                         baseline=(None, 0), preload=True)

    # replace these keys with what event_id actually printed
    low = epochs['low_load']
    high = epochs['high_load']

    frontal = [ch for ch in ['Fz','F3','F4'] if ch in raw.ch_names]
    parietal = [ch for ch in ['Pz','P3','P4'] if ch in raw.ch_names]
    f_idx = [raw.ch_names.index(c) for c in frontal]
    p_idx = [raw.ch_names.index(c) for c in parietal]

    def band_power(data, sfreq, band):
        psds, _ = psd_array_welch(data, sfreq, fmin=band[0], fmax=band[1])
        return psds.mean(-1)

    theta_l = band_power(low.get_data(), raw.info['sfreq'], (4,8))[:, f_idx].mean(1)
    alpha_l = band_power(low.get_data(), raw.info['sfreq'], (8,13))[:, p_idx].mean(1)
    theta_h = band_power(high.get_data(), raw.info['sfreq'], (4,8))[:, f_idx].mean(1)
    alpha_h = band_power(high.get_data(), raw.info['sfreq'], (8,13))[:, p_idx].mean(1)

    all_ratio_low.append(theta_l / alpha_l)
    all_ratio_high.append(theta_h / alpha_h)

# --- Step 5: classify ---
ratio_low = np.concatenate(all_ratio_low)
ratio_high = np.concatenate(all_ratio_high)
X = np.concatenate([ratio_low, ratio_high]).reshape(-1, 1)
y = np.array([0]*len(ratio_low) + [1]*len(ratio_high))

clf = SVC(kernel='rbf')
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")