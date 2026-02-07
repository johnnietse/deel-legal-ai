# Worker Classification Model

This module implements the Machine Learning classifier for determining worker status (Employee vs Independent Contractor).

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 10 key legal factors derived from the *Sagaz Industries* test.
- **Accuracy**: 100% on synthetic test set.
- **Output**: Binary classification + Confidence score + Feature Importance.

## 📂 Files

| File | Purpose |
|------|---------|
| `train_classifier.py` | Main training script. Handles data loading, preprocessing, training, and saving. |
| `model_inference.py` | Utility for making predictions with the saved `.joblib` model. |

## 🚀 Usage

### Training
To retrain the model with new data in `data/employment_cases_large.csv`:
```bash
python -m ml_classifier.train_classifier
```

# Demo
To run an interactive prediction demo:
```bash
python -m ml_classifier.train_classifier --demo
```
