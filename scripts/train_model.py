import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("dataset/posture_landmarks.csv")

print("Columns in dataset:")
print(data.columns)

# Detect label column automatically
possible_labels = ["label", "class", "posture"]

label_column = None
for col in possible_labels:
    if col in data.columns:
        label_column = col
        break

if label_column is None:
    raise Exception("No label column found in dataset!")

print("Using label column:", label_column)

# Separate features and labels
X = data.drop(label_column, axis=1)
y = data[label_column]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/posture_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("\nModel saved to: model/posture_model.pkl")
print("Encoder saved to: model/label_encoder.pkl")
