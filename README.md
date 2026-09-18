# EEG Cognitive Workload Classification

Classifying low vs. high working-memory load from EEG using frontal theta power, parietal alpha power, and their ratio — a standard, literature-backed cognitive workload marker.

## Dataset
[OpenNeuro ds003838](https://openneuro.org/datasets/ds003838) — Digit Span Task. 64-channel EEG, pupillometry, and cardiovascular data from 86 participants performing a working-memory task at varying load. BIDS-formatted. This project uses a subset of subjects for tractable local analysis.

## Why this task
Frontal midline theta rises with executive/working-memory engagement; parietal alpha drops with reduced idling/attention. The theta/alpha ratio is a more robust workload marker than either band alone, and is grounded in known prefrontal (executive) vs. parietal (attention/sensory) functional roles.

## Pipeline
1. **Load** — `mne-bids` reads raw EEG + BIDS metadata (channels, events) directly into MNE.
2. **Filter** — 1–40 Hz bandpass to remove drift and high-frequency noise.
3. **ICA** — removes eye-blink and muscle artifacts, critical since frontal channels (used for theta) are eye-blink-prone.
4. **Epoching** — segments continuous signal into event-locked trials (low load vs. high load), baseline-corrected.
5. **Feature extraction** — Welch PSD to estimate theta (4–8 Hz) and alpha (8–13 Hz) power per epoch; compute frontal-theta/parietal-alpha ratio.
6. **Classification** — SVM classifier, 5-fold cross-validation, to distinguish load conditions from the 3-feature set (theta, alpha, ratio).

## Results
0.72 +- 0.01

## Limitations
- Subset of full 86-subject dataset used (list N).
- Small, hand-picked feature set (3 features) — no exhaustive feature search or channel optimization.
- Within/across-subject validation approach: [specify which you used].
- No independent held-out test set beyond cross-validation folds.

## Tools
Python, MNE-Python, MNE-BIDS, scikit-learn

## Setup
```bash
pip install mne mne-bids scikit-learn openneuro-py pandas
```
