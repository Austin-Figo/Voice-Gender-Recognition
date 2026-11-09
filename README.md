# Voice Gender Recognition

A Python machine-learning project that trains a Random Forest classifier on acoustic voice measurements to classify the dataset labels.

## Repository contents

- `train_gender_model.py` — loads the tabular dataset, splits training and test data, fits preprocessing on the training partition, trains a Random Forest classifier, and saves model artifacts.
- `test_model.py` — records microphone audio, computes 20 MFCC means with Librosa, and attempts inference.
- `voice.csv.xls` — dataset file. Despite its extension, the training script reads it as CSV text.

## Environment

Use Python 3.10 or newer. Install the packages used by the scripts:

```bash
python -m pip install pandas numpy scikit-learn joblib sounddevice librosa
```

## Train

```bash
python train_gender_model.py
```

The script writes `gender_model.joblib` and `scaler.joblib` to the working directory. The scaler is fitted only on the training split to avoid preprocessing leakage.

## Important inference limitation

The training script uses the dataset's tabular acoustic columns (for example, mean frequency, spectral measures, and pitch-related statistics). The current microphone script computes MFCC features instead. These are not the same feature representation or column order, so the current `test_model.py` pipeline is not validated as a compatible inference path and must not be presented as a reliable end-to-end predictor until a matching feature extractor and evaluation are implemented.

## Evaluation

The training script reports accuracy and a classification report on a held-out stratified split. Do not quote a model accuracy unless you have run the script against the exact committed dataset and recorded the output. Verify dataset provenance and redistribution rights before publishing or reusing it.

## Scope

This is an educational classification experiment. The model predicts dataset labels from voice characteristics; it does not determine a person's gender identity.
