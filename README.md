# Voice Gender Recognition — Machine Learning Experiment

A Python classification experiment using acoustic measurements from a voice dataset and a Random Forest model.

> **Status:** Training workflow documented; microphone inference is experimental and has not been validated end to end.

## Project overview

The training script reads a tabular acoustic-feature dataset, separates training and test data, fits preprocessing on the training partition, trains a Random Forest classifier, and reports held-out evaluation metrics.

The repository also contains a microphone recording script. Its current MFCC feature extraction does not match the feature columns used during model training, so its predictions must not be treated as a valid end-to-end result.

## Repository contents

| File | Purpose |
|---|---|
| train_gender_model.py | Dataset loading, train/test split, preprocessing, model training, and evaluation |
| test_model.py | Experimental microphone recording and inference script |
| voice.csv.xls | CSV-formatted dataset stored with an .xls filename extension |

## Technology

Python 3.10+, pandas, NumPy, scikit-learn, joblib, sounddevice, and librosa (experimental microphone script).

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install pandas numpy scikit-learn joblib sounddevice librosa
```

## Train and evaluate

From the repository root:

```bash
python train_gender_model.py
```

The script reads the dataset beside the Python file, performs a stratified train/test split, fits the scaler on the training data only, trains the Random Forest model, prints evaluation metrics, and writes gender_model.joblib and scaler.joblib.

Do not publish an accuracy figure unless you have run the training script and recorded the result for the exact dataset revision.

## Inference limitation

The training dataset contains a fixed set of tabular acoustic measurements. The current microphone script computes 20 MFCC means instead. Those feature vectors are not equivalent in definition or ordering, even if their dimensions happen to be similar.

A reliable inference workflow requires one consistent feature-extraction implementation for both training and microphone audio, followed by validation on held-out recordings. Until that work is completed, test_model.py is an experimental placeholder rather than a validated predictor.

## Responsible interpretation

The model predicts labels present in the dataset from acoustic measurements. It cannot determine a person's gender identity. Dataset provenance, consent, representativeness, and redistribution rights should be checked before reuse or publication.
