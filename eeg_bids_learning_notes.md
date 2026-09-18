# EEG / BIDS / MNE — Learning Notes

## 1. BIDS (Brain Imaging Data Structure)
Standardized way to organize neuroimaging data so any tool/lab can read it without custom scripts.

```
sub-01/eeg/sub-01_task-rest_eeg.edf
sub-01/eeg/sub-01_task-rest_channels.tsv
sub-01/eeg/sub-01_task-rest_events.tsv
sub-01/eeg/sub-01_task-rest_eeg.json
```
- `channels.tsv` — channel names, types (EEG/EOG), units, sampling freq
- `events.tsv` — onset (sec), duration, trial_type
- `eeg.json` — sampling frequency, reference electrode, filters applied
- `participants.tsv` — age, sex, group per subject

Spec maintained by the BIDS community (bids.neuroimaging.io).

## 2. MNE-Python
Main open-source library for EEG/MEG analysis. Core objects:
- `Raw` — continuous signal (channels × time)
- `Epochs` — signal cut into event-locked chunks
- `Evoked` — average over epochs (ERP)

## 3. mne-bids
Bridges the two: reads BIDS folders directly into MNE objects (`read_raw_bids`), so you never manually parse the `.tsv`/`.json` sidecar files — sampling rate, channel names, and events are pulled in automatically.

## 4. Dataset used
**OpenNeuro ds003838** — Digit Span Task. 64-channel EEG, pupillometry, cardiovascular data, 86 participants, BIDS-formatted. Working memory load task — real cognitive science angle, large enough N for credible cross-subject validation.

## 5. Why each preprocessing step exists

| Step | Why |
|---|---|
| **Filter (1–40 Hz)** | Removes slow drift and high-frequency muscle noise — standard first pass on raw EEG |
| **ICA (Independent Component Analysis)** | Separates and removes eye-blink/muscle artifacts. Critical here specifically because frontal channels — which we use for theta power — are the most blink-contaminated |
| **Epoching** | Cuts continuous signal into event-locked windows so you're comparing the *same relative time window* across trials, not arbitrary segments |
| **Baseline correction** | Subtracts pre-stimulus signal level, so any power difference reflects the task/condition, not pre-existing drift |
| **Welch's method (PSD)** | Estimates power spectral density from noisy, finite EEG by averaging overlapping windows — reduces noise vs. a single FFT |

## 6. Why theta and alpha specifically
- **Frontal midline theta (4–8 Hz)** increases with cognitive/working-memory load — reflects prefrontal executive engagement.
- **Parietal/occipital alpha (8–13 Hz)** decreases with load — reflects reduced sensory idling/attention.
- **Theta/alpha ratio** is a more robust workload marker than either band alone, because it combines two independent, anatomically-grounded signals rather than relying on one noisy measure.
- Channel selection isn't arbitrary: frontal region ↔ executive function, parietal region ↔ attention/sensory idling. This spatial mapping is something an interviewer may ask about directly.

## 7. Why cross-validation
Tests whether the classifier generalizes to unseen data rather than memorizing the training set. Cross-subject validation (vs. within-subject) is harder but more credible — it shows the pattern holds across different people's brains, not just one dataset quirk.

## 8. Honest limitations to state (shows maturity, not weakness)
- Using a subset of the full 86 subjects, not the whole dataset
- Small, hand-picked feature set (3 features: theta, alpha, ratio) — no exhaustive feature search or channel optimization
- Cross-validation folds only — no fully independent held-out test set
