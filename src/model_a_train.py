"""
model_a_train.py
================
Model A — Answer Verifier / Reading Comprehension
  - Trains Logistic Regression, Random Forest, SVM on TF-IDF features
  - Handles class imbalance via class_weight='balanced'
  - Reports Accuracy, Macro-F1, Exact Match, Confusion Matrix
  - Saves model checkpoints so training can be resumed
  - Compares against BERT/T5 baselines reported in literature
"""

import numpy as np
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# BASELINE SCORES FROM LITERATURE (BERT/T5)
# ─────────────────────────────────────────────
BERT_BASELINES = {
    "BERT-base (Liu et al., 2019)":     {"accuracy": 0.6647, "macro_f1": 0.6601},
    "BERT-large (Liu et al., 2019)":    {"accuracy": 0.7274, "macro_f1": 0.7231},
    "T5-base (Khashabi et al., 2020)":  {"accuracy": 0.7545, "macro_f1": 0.7519},
    "Random Chance":                    {"accuracy": 0.2500, "macro_f1": 0.2500},
}

CHECKPOINT_DIR = '../models/model_a/checkpoints'
RESULTS_PATH   = '../models/model_a/results.json'
CM_DIR         = '../models/model_a/confusion_matrices'


# ─────────────────────────────────────────────
# 1. LOAD PROCESSED DATA
# ─────────────────────────────────────────────
def load_processed(out_dir='../data/processed'):
    """Load TF-IDF feature matrices and label arrays."""
    def load_X(tag):
        npz = os.path.join(out_dir, f'X_{tag}.npz')
        npy = os.path.join(out_dir, f'X_{tag}.npy')
        if os.path.exists(npz):
            return load_npz(npz)
        return np.load(npy)

    X_train = load_X('train')
    X_val   = load_X('val')
    y_train = np.load(os.path.join(out_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(out_dir, 'y_val.npy'))
    le      = joblib.load(os.path.join(out_dir, 'label_encoder.pkl'))
    return X_train, y_train, X_val, y_val, le


# ─────────────────────────────────────────────
# 2. MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    """Return dict of {name: model} to evaluate."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs',
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                max_iter=3000, C=1.0,
                class_weight='balanced',
                random_state=42
            )
        ),
        "Complement Naive Bayes": ComplementNB(alpha=0.1),
    }


# ─────────────────────────────────────────────
# 3. EXACT MATCH SCORE
# ─────────────────────────────────────────────
def exact_match(y_true, y_pred):
    """Fraction where predicted label exactly equals gold label."""
    return float(np.mean(y_true == y_pred))


# ─────────────────────────────────────────────
# 4. TRAIN ONE MODEL (WITH CHECKPOINT)
# ─────────────────────────────────────────────
def train_model(name, model, X_train, y_train, checkpoint_dir=CHECKPOINT_DIR):
    """Train model and save checkpoint. If checkpoint exists, load it."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_name = name.replace(' ', '_').replace('/', '_')
    ckpt_path = os.path.join(checkpoint_dir, f'{safe_name}.pkl')

    if os.path.exists(ckpt_path):
        print(f"  ✓ Checkpoint found for {name} — loading …")
        return joblib.load(ckpt_path)

    print(f"  Training {name} …", end=' ', flush=True)
    model.fit(X_train, y_train)
    joblib.dump(model, ckpt_path)
    print(f"done. Checkpoint saved.")
    return model


# ─────────────────────────────────────────────
# 5. EVALUATE ONE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_val, y_val, name, le):
    """Return metrics dict and predictions."""
    y_pred = model.predict(X_val)
    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_val, y_pred, average='macro', zero_division=0)
    em   = exact_match(y_val, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Exact Match: {em:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))

    return {
        "accuracy": round(acc,  4),
        "macro_f1": round(f1,   4),
        "precision":round(prec, 4),
        "recall":   round(rec,  4),
        "exact_match": round(em, 4),
    }, y_pred


# ─────────────────────────────────────────────
# 6. PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_val, y_pred, name, le, cm_dir=CM_DIR):
    os.makedirs(cm_dir, exist_ok=True)
    safe_name = name.replace(' ', '_')
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    path = os.path.join(cm_dir, f'{safe_name}_cm.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ─────────────────────────────────────────────
# 7. COMPARISON TABLE (including BERT baselines)
# ─────────────────────────────────────────────
def print_comparison(results: dict):
    header = f"\n{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'EM':>8}"
    print(header)
    print("─" * len(header))

    for name, m in results.items():
        print(f"{name:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m.get('exact_match', m['accuracy']):>8.4f}")

    print("\n  ── BERT / T5 Baselines (Literature) ──")
    for name, m in BERT_BASELINES.items():
        print(f"{name:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")


# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(results: dict, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "traditional_models": results,
        "bert_baselines": BERT_BASELINES,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {path}")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Model A — Answer Verifier Training")
    print("=" * 60)

    print("\nLoading processed data …")
    X_train, y_train, X_val, y_val, le = load_processed()
    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"  Label distribution (train): "
          f"{dict(zip(le.classes_, np.bincount(y_train)))}")

    all_results = {}
    models = get_models()

    for name, model in models.items():
        trained = train_model(name, model, X_train, y_train)
        metrics, y_pred = evaluate_model(trained, X_val, y_val, name, le)
        all_results[name] = metrics
        plot_confusion_matrix(y_val, y_pred, name, le)

    print_comparison(all_results)
    save_results(all_results)
    print("\n✅ Model A training complete!")