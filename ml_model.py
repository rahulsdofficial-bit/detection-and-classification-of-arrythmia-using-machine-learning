"""
ml_model.py
Machine Learning Models for Arrhythmia Detection:
- Primary Proposed Model: Weighted K-Nearest Neighbors (WKNN)
- Comparison Baseline Models: Standard KNN, SVM, Logistic Regression
- Standardized Training, Inference, and Serialization
"""

import os
import json
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
AAMI_FULL_NAMES = {
    'N': 'Normal Beat',
    'S': 'Supraventricular Ectopic Beat (SVEB)',
    'V': 'Ventricular Ectopic Beat (VEB)',
    'F': 'Fusion Beat',
    'Q': 'Unknown / Paced Beat'
}

class WKNNClassifier:
    """
    Weighted K-Nearest Neighbors (WKNN) Classifier.
    Assigns higher voting weights to closer neighbors in feature space:
      weight_i = 1.0 / ((distance_i + epsilon) ** power)
    This significantly improves discrimination of subtle morphological variations in ECG beats.
    """
    def __init__(self, n_neighbors=5, power=2.0, epsilon=1e-5):
        self.n_neighbors = n_neighbors
        self.power = power
        self.epsilon = epsilon
        self.classes_ = None
        self.model = None

    def _custom_weights(self, distances):
        # Apply inverse distance weighting with specified power and epsilon
        return 1.0 / ((distances + self.epsilon) ** self.power)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Use scikit-learn KNeighborsClassifier with custom distance weight callable
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self._custom_weights,
            metric='minkowski',
            p=2
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

def create_model_suite(k_neighbors=5):
    """Initializes the proposed WKNN model and baseline comparison models."""
    return {
        'WKNN': WKNNClassifier(n_neighbors=k_neighbors, power=2.0),
        'KNN': KNeighborsClassifier(n_neighbors=k_neighbors, weights='uniform'),
        'SVM': SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

def evaluate_classifier(model, X_train, y_train, X_test, y_test, labels=None):
    """
    Trains and evaluates a classifier, calculating actual accuracy, precision,
    recall, F1-score, and confusion matrix.
    """
    if labels is None:
        labels = AAMI_CLASSES

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate actual metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    # Per-class metrics
    per_class = {}
    for lbl in labels:
        mask_true = (y_test == lbl)
        mask_pred = (y_pred == lbl)
        c_acc = float(np.mean(y_pred[mask_true] == lbl)) if np.sum(mask_true) > 0 else 0.0
        per_class[lbl] = {
            'support': int(np.sum(mask_true)),
            'sensitivity': c_acc
        }

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class': per_class
    }

def save_trained_pipeline(wknn_model, scaler, metadata, models_dir='models'):
    """Serializes the trained WKNN model, scaler, and metadata."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(wknn_model, os.path.join(models_dir, 'wknn_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model and scaler successfully saved to {models_dir}/")

def load_trained_pipeline(models_dir='models'):
    """Loads the trained WKNN model and scaler into memory."""
    model_path = os.path.join(models_dir, 'wknn_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    meta_path = os.path.join(models_dir, 'model_metadata.json')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    return model, scaler, metadata
