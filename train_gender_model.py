import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import os

# === Step 1: Load Dataset ===
file_path = "voice.csv"  

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ The file '{file_path}' was not found. Please check the path.")

df = pd.read_csv(file_path)

# === Step 2: Encode Labels (male/female) ===
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # male=1, female=0

# === Step 3: Split into Features & Labels ===
X = df.drop('label', axis=1)
y = df['label']

# === Step 4: Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Step 6: Train Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 7: Evaluate on Test Set ===
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# === Step 8: Save Model & Scaler ===
dump(model, 'gender_model.joblib')
dump(scaler, 'scaler.joblib')
print("\n💾 Model and scaler saved successfully.")


