# Quick training script for worker classification model
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Paths - Use large dataset
DATA_PATH = Path("data/employment_cases_large.csv")
MODEL_PATH = Path("models/worker_classifier.joblib")

# Feature columns
FEATURE_COLUMNS = [
    'Supervision/review of work',
    'Ability to hire employees',
    'Delegation of tasks',
    'Ownership of tools',
    'Chance of profit',
    'Risk of loss',
    'Exclusivity of services',
    'Who sets the work hours',
    'Where the work is performed',
    'Is the worker required to wear a uniform?'
]

print("=" * 60)
print("WORKER CLASSIFICATION MODEL TRAINING")
print("=" * 60)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"\n📊 Loaded {len(df)} cases from {DATA_PATH}")

# Prepare feature matrix
label_encoders = {}
X_columns = []

for col in FEATURE_COLUMNS:
    if col in df.columns:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        X_columns.append(encoded)
        print(f"   ✓ {col}: {len(le.classes_)} categories")

X = np.column_stack(X_columns).astype(np.float64)
print(f"\n📐 Feature matrix: {X.shape}")

# Prepare target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['Outcome'].astype(str))
print(f"🎯 Target classes: {target_encoder.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📈 Train: {len(y_train)}, Test: {len(y_test)}")

# Train model (simplified parameter grid for speed)
print("\n🔧 Training Random Forest with hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print(f"   Best params: {grid_search.best_params_}")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\n" + "-" * 40)
print("RESULTS")
print("-" * 40)
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"📈 Precision: {precision:.2%}")
print(f"📊 Recall: {recall:.2%}")
print(f"📉 F1 Score: {f1:.2%}")
print(f"🔄 CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Feature importance
print("\n" + "-" * 40)
print("FEATURE IMPORTANCE")
print("-" * 40)
feature_importance = dict(zip(
    [col for col in FEATURE_COLUMNS if col in df.columns],
    model.feature_importances_
))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

for i, (feature, importance) in enumerate(feature_importance.items()):
    bar = "█" * int(importance * 40)
    print(f"{i+1:2}. {feature[:35]:35} {importance:.3f} {bar}")

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model_data = {
    "model": model,
    "label_encoders": label_encoders,
    "target_encoder": target_encoder,
    "feature_importance": feature_importance,
    "training_stats": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "feature_columns": [c for c in FEATURE_COLUMNS if c in df.columns],
        "target_classes": list(target_encoder.classes_),
        "n_samples": len(df)
    }
}
joblib.dump(model_data, MODEL_PATH)
print(f"\n💾 Model saved to {MODEL_PATH}")

# Save feature importance as JSON
with open(MODEL_PATH.parent / "feature_importance.json", "w") as f:
    json.dump({
        "feature_importance": feature_importance,
        "training_stats": model_data["training_stats"]
    }, f, indent=2)
print(f"📋 Feature importance saved")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
