# EEG Motor Imagery Classification

A from-scratch replication of a classic motor imagery BCI pipeline: classifying imagined left-hand vs. right-hand movement from EEG signals.

Built as a learning project to understand the standard EEG → feature extraction → classification workflow used in brain-computer interface research.

## Dataset
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 64-channel EEG, 160 Hz sampling rate. Auto-downloaded via MNE (no manual download needed).

## Pipeline
1. **Preprocessing** — bandpass filter (7–30 Hz, the mu/beta rhythm range relevant to motor activity)
2. **Epoching** — segment continuous EEG into trial windows around each movement-imagery cue
3. **Feature extraction** — [Common Spatial Patterns (CSP)](https://doi.org/10.1109/MSP.2008.4408441), which learns spatial filters that best separate the two classes
4. **Classification** — Linear Discriminant Analysis (LDA) and Support Vector Machine (SVM), evaluated with 5-fold cross-validation

## Results

| Setup | Classifier | Accuracy (5-fold CV) |
|---|---|---|
| Single subject | SVM (linear) | 0.63 ± 0.05 |
| Single subject | LDA | *(add once run)* |
| 5 subjects pooled, single CSP | LDA | 0.45 ± 0.05 |
| 5 subjects pooled, single CSP | SVM (linear) | 0.46 ± 0.04 |

Chance level = 0.5 (2-class). Single-subject accuracy is in line with published baselines on this dataset. Pooling subjects without per-subject CSP calibration drops accuracy to near-chance — a known effect in BCI research called **inter-subject variability** (see `learning.md` for why).

See `learning.md` for detailed notes on the concepts used here (CSP, cross-validation, inter-subject variability in BCI, etc.) — written while building this as a learning log.

## Limitations
- Small subset of data used (1–5 subjects out of 109) for tractability on modest hardware.
- Single CSP fit across pooled subjects — no per-subject calibration or transfer-learning approach yet.
- No independent held-out test set beyond cross-validation folds.

## Setup
```bash
pip install mne mne-bids numpy scikit-learn matplotlib openneuro-py pandas
```

## Usage
Open `motor_imagery_classification.ipynb` and run cells top to bottom. Data downloads automatically on first run.

## References
- Blankertz, B. et al. (2008). [Optimizing Spatial Filters for Robust EEG Single-Trial Analysis](https://doi.org/10.1109/MSP.2008.4408441). *IEEE Signal Processing Magazine*.
- [MNE-Python documentation](https://mne.tools/stable/index.html)
