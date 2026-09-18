import os
import pandas as pd
import openneuro
from mne_bids import find_matching_paths

import config


class DataManager:
    def __init__(self, dataset=config.DATASET, root=config.ROOT):
        self.dataset = dataset
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def ensure_downloaded(self, include):
        missing = []
        for f in include:
            path = os.path.join(self.root, f)
            if f.endswith("/eeg"):
                set_files = (
                    [x for x in os.listdir(path) if x.endswith(".set")]
                    if os.path.exists(path) else []
                )
                if not set_files:
                    missing.append(f)
            else:
                if not os.path.exists(path):
                    missing.append(f)

        if missing:
            print(f"Downloading missing: {missing}")
            openneuro.download(dataset=self.dataset, target_dir=self.root, include=missing)
        else:
            print("All required files already present, skipping download.")

    def get_valid_subject_ids(self, n_subjects=config.N_SUBJECTS):
        self.ensure_downloaded(["dataset_description.json", "participants.tsv"])
        participants = pd.read_csv(f"{self.root}/participants.tsv", sep="\t")
        valid = participants[participants['EEG_excluded'] == 'no']
        subject_ids = valid['participant_id'].str.replace('sub-', '').tolist()[:n_subjects]
        print("Using subjects:", subject_ids)
        return subject_ids

    def download_subjects_eeg(self, subject_ids):
        includes = [f"sub-{s}/eeg" for s in subject_ids]
        self.ensure_downloaded(includes)

    def get_bids_paths(self, task="memory"):
        paths = find_matching_paths(
            self.root, datatypes="eeg", suffixes="eeg",
            tasks=task, extensions=".set"
        )
        print(paths)
        return paths

    def prepare(self, n_subjects=config.N_SUBJECTS, task="memory"):
        """Runs the full download pipeline and returns BIDSPaths ready for feature extraction."""
        subject_ids = self.get_valid_subject_ids(n_subjects)
        self.download_subjects_eeg(subject_ids)
        return self.get_bids_paths(task)
