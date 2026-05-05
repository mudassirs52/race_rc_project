import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. LOAD PROCESSED DATA ─────────────────────────────────
def load_processed():
    X_train = np.load('../data/processed/X_train.npy')
    y_train = np.load('../data/processed/y_train.npy')
    X_val = np.load('../data/processed/X_val.npy')
    y_val = np.load('../data/processed/y_val.npy')
    return X_train, y_train, X_val, y_val

# ─── 2. TRAIN LOGISTIC REGRESSION ───────────────────────────
def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    lr.fit(X_train, y_train)
    return lr

# ─── 3. TRAIN SVM ───────────────────────────────────────────
def train_svm(X_train, y_train):
    print("Training SVM...")
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm

# ─── 4. EVALUATE MODEL ──────────────────────────────────────
def evaluate_model(model, X_val, y_val, model_name):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['A','B','C','D']))
    
    return acc, f1, y_pred

# ─── 5. PLOT CONFUSION MATRIX ───────────────────────────────
def plot_confusion_matrix(y_val, y_pred, model_name):
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['A','B','C','D'],
                yticklabels=['A','B','C','D'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'../models/model_a/traditional/{model_name}_confusion_matrix.png')
    plt.show()
    print(f"Saved confusion matrix!")

# ─── 6. SAVE MODEL ──────────────────────────────────────────
def save_model(model, model_name):
    os.makedirs('../models/model_a/traditional', exist_ok=True)
    path = f'../models/model_a/traditional/{model_name}.pkl'
    joblib.dump(model, path)
    print(f"Saved {model_name} to {path}")

# ─── 7. MAIN ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading processed data...")
    X_train, y_train, X_val, y_val = load_processed()
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Sirf 5000 samples use karo
    X_train = X_train[:5000]
    y_train = y_train[:5000]

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_acc, lr_f1, lr_pred = evaluate_model(lr_model, X_val, y_val, "Logistic Regression")
    plot_confusion_matrix(y_val, lr_pred, "Logistic_Regression")
    save_model(lr_model, "logistic_regression")

    # SVM
    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_f1, svm_pred = evaluate_model(svm_model, X_val, y_val, "SVM")
    plot_confusion_matrix(y_val, svm_pred, "SVM")
    save_model(svm_model, "svm")

    # Comparison
    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 47)
    print(f"{'Logistic Regression':<25} {lr_acc:>10.4f} {lr_f1:>10.4f}")
    print(f"{'SVM':<25} {svm_acc:>10.4f} {svm_f1:>10.4f}")