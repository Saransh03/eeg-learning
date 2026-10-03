# EEG Learning Projects

A collection of small, from-scratch EEG/BCI projects built to learn the standard signal-processing and classification workflow used in EEG and brain-computer interface research. Each project lives in its own folder with its own notebook, README, and learning notes.

## Projects

| Folder | Description | Result |
|---|---|---|
| [`motor-imagery-classification/`](./motor-imagery-classification) | Classifying imagined left-hand vs. right-hand movement using CSP + SVM/LDA | 0.63 ± 0.05 (single subject) |
| [`cognitive-workload-classification/`](./cognitive-workload-classification) | Classifying low vs. high working-memory load using frontal theta / parietal alpha power ratio | 0.72 ± 0.01 |

## Tools used across projects
Python, MNE-Python, MNE-BIDS, scikit-learn

## Setup
Each project folder has its own `pip install` line in its README — dependencies are mostly shared (MNE ecosystem + scikit-learn) but check each folder for specifics.
