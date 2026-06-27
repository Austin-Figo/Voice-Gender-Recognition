import sounddevice as sd
import numpy as np
import librosa
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# Load trained model and scaler
model = load("gender_model.joblib")
scaler = load("scaler.joblib")

# === Function: Record audio from microphone ===
def record_audio(duration=3, sr=16000):
    print("🎙️ Recording for {} seconds...".format(duration))
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten(), sr

# === Function: Extract features (similar to training) ===
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# === Main Prediction Function ===
def predict_gender():
    audio, sr = record_audio()
    features = extract_features(audio, sr)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    label = "Male" if prediction == 1 else "Female"
    print(f"\n🔊 Predicted Gender: {label}")

if __name__ == "__main__":
    predict_gender()
