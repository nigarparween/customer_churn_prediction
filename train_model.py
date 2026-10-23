
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load and clean dataset
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna()

print("Clean dataset shape:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Save the valid categories for API encoding
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

categories = {
    col: sorted(X[col].unique().tolist())
    for col in categorical_columns
}

# 3. Convert text features into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Remember the exact columns used during training
feature_columns = X.columns.tolist()

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
print("Model training completed successfully!")

# 6. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Save model and preprocessing information
artifact = {
    "model": model,
    "feature_columns": feature_columns,
    "categories": categories
}

model_path = MODEL_DIR / "churn_model.pkl"
joblib.dump(artifact, model_path)

print("\nModel saved to:", model_path)
