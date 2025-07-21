import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("adult 3.csv")
print("✅ Dataset loaded successfully")

# Clean data
data.replace(' ?', np.nan, inplace=True)
data.dropna(inplace=True)
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
data.drop(columns=['education'], inplace=True)
data = data[(data['age'] >= 17) & (data['age'] <= 75)]

# Encode categorical features using separate LabelEncoders
categorical_cols = ['workclass', 'gender', 'marital-status', 'occupation',
                    'relationship', 'race', 'native-country', 'income']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoder

# Save all label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Prepare features and target
X = data.drop(columns=['income'])
y = data['income']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
with open("salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🎉 Model, scaler, and encoders saved successfully.")
