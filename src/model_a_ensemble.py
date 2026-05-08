"""
model_a_ensemble.py
====================
Model A — Ensemble Methods (5 marks)

Implements:
  1. Voting Classifier (Hard + Soft)
  2. Stacking Classifier (LR meta-learner)
  3. Comparison vs individual model baselines
"""

import numpy as np
import joblib, os, json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

DATA_DIR = '../data/processed'
CKPT_DIR = '../models/model_a/checkpoints'
OUT_DIR  = '../models/model_a/ensemble'


def load_data():
    def load_X(tag):
        p = os.path.join(DATA_DIR, f'X_{tag}.npz')
        return load_npz(p) if os.path.exists(p) \
               else np.load(os.path.join(DATA_DIR, f'X_{tag}.npy'))
    X_tr = load_X('train'); X_va = load_X('val')
    y_tr = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    ans_va = np.load(os.path.join(DATA_DIR, 'ans_val.npy'))
    le   = joblib.load(os.path.join(DATA_DIR, 'label_encoder.pkl'))
    return X_tr, y_tr, X_va, ans_va, le


def get_base_models():
    """Load checkpoints if available, else define fresh."""
    models = {}
    defs = {
        "Logistic_Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs',
            class_weight='balanced',
            random_state=42, n_jobs=-1),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=-1),
        "Linear_SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=3000, C=1.0, class_weight='balanced',
                      random_state=42)),
        "Complement_NB": ComplementNB(alpha=0.1),
    }
    for name, model in defs.items():
        ckpt = os.path.join(CKPT_DIR, f'{name}.pkl')
        if os.path.exists(ckpt):
            models[name] = joblib.load(ckpt)
            print(f"  Loaded checkpoint: {name}")
        else:
            models[name] = model
    return models


def train_or_load(name, model, X_tr, y_tr):
    ckpt = os.path.join(CKPT_DIR, f'{name}.pkl')
    if os.path.exists(ckpt):
        print(f"  Loading {name} from checkpoint…")
        return joblib.load(ckpt)
    print(f"  Training {name}…", end=' ', flush=True)
    model.fit(X_tr, y_tr)
    joblib.dump(model, ckpt)
    print("done.")
    return model


def evaluate(model, X_va, ans_va, le, name):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_va)[:, 1]
    else:
        probs = getattr(model, 'decision_function', model.predict)(X_va)
        
    probs = probs.reshape(-1, 4)
    y_pred_ans = np.argmax(probs, axis=1)

    acc = accuracy_score(ans_va, y_pred_ans)
    f1  = f1_score(ans_va, y_pred_ans, average='macro', zero_division=0)
    print(f"\n── {name} ──")
    print(f"  Accuracy : {acc:.4f}  |  Macro F1 : {f1:.4f}")
    print(classification_report(ans_va, y_pred_ans, target_names=le.classes_, zero_division=0))

    # Confusion matrix
    os.makedirs(os.path.join(OUT_DIR,'plots'), exist_ok=True)
    cm = confusion_matrix(ans_va, y_pred_ans)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plots',
                             f'{name.replace(" ","_")}_cm.png'), dpi=150)
    plt.close()
    return {"accuracy": round(acc,4), "macro_f1": round(f1,4)}, y_pred_ans


def plot_ensemble_comparison(all_results):
    names = list(all_results.keys())
    accs  = [m['accuracy'] for m in all_results.values()]
    f1s   = [m['macro_f1'] for m in all_results.values()]
    # Color ensemble models differently
    ens_keys = {"Voting Hard", "Voting Soft", "Stacking"}
    colors_acc = ['#FF7043' if n in ens_keys else '#42A5F5' for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.4), 5))
    b1 = ax.bar(x-0.2, accs, 0.35, color=colors_acc, label='Accuracy', alpha=0.9)
    b2 = ax.bar(x+0.2, f1s,  0.35, color=['#FF8A65' if n in ens_keys else '#64B5F6'
                                            for n in names], label='Macro F1', alpha=0.9)
    for b, v in zip(b1, accs):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylim(0,1.0); ax.set_ylabel('Score')
    ax.set_title('Ensemble vs Individual Models (Validation Set)')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ensemble_comparison.png'), dpi=150)
    plt.show()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("="*60)
    print("  Model A — Ensemble Methods")
    print("="*60)

    X_tr, y_tr, X_va, ans_va, le = load_data()
    base_models = get_base_models()

    # ── Ensure all base models are trained ──
    trained = {}
    for name, m in base_models.items():
        trained[name] = train_or_load(name, m, X_tr, y_tr)

    # ── Individual baselines ──
    all_results = {}
    for name, m in trained.items():
        metrics, _ = evaluate(m, X_va, ans_va, le, name.replace('_',' '))
        all_results[name.replace('_',' ')] = metrics

    # ── 1. Voting — Hard ──
    ckpt_vh = os.path.join(CKPT_DIR, 'Voting_Hard.pkl')
    if os.path.exists(ckpt_vh):
        voting_hard = joblib.load(ckpt_vh)
    else:
        voting_hard = VotingClassifier(
            estimators=[(n, m) for n, m in trained.items()],
            voting='hard', n_jobs=-1)
        print("\n  Training Voting (Hard)…", end=' ', flush=True)
        voting_hard.fit(X_tr, y_tr)
        joblib.dump(voting_hard, ckpt_vh)
        print("done.")
    m, _ = evaluate(voting_hard, X_va, ans_va, le, "Voting Hard")
    all_results["Voting Hard"] = m

    # ── 2. Voting — Soft ──
    ckpt_vs = os.path.join(CKPT_DIR, 'Voting_Soft.pkl')
    if os.path.exists(ckpt_vs):
        voting_soft = joblib.load(ckpt_vs)
    else:
        voting_soft = VotingClassifier(
            estimators=[(n, m) for n, m in trained.items()],
            voting='soft', n_jobs=-1)
        print("\n  Training Voting (Soft)…", end=' ', flush=True)
        voting_soft.fit(X_tr, y_tr)
        joblib.dump(voting_soft, ckpt_vs)
        print("done.")
    m, _ = evaluate(voting_soft, X_va, ans_va, le, "Voting Soft")
    all_results["Voting Soft"] = m

    # ── 3. Stacking ──
    ckpt_st = os.path.join(CKPT_DIR, 'Stacking.pkl')
    if os.path.exists(ckpt_st):
        stacking = joblib.load(ckpt_st)
    else:
        meta = LogisticRegression(max_iter=1000, C=0.5, random_state=42, n_jobs=-1)
        stacking = StackingClassifier(
            estimators=[(n, m) for n, m in trained.items()],
            final_estimator=meta,
            cv=3, n_jobs=-1, passthrough=False)
        print("\n  Training Stacking (this may take a while)…", end=' ', flush=True)
        stacking.fit(X_tr, y_tr)
        joblib.dump(stacking, ckpt_st)
        print("done.")
    m, _ = evaluate(stacking, X_va, y_va, le, "Stacking")
    all_results["Stacking"] = m

    # ── Comparison ──
    plot_ensemble_comparison(all_results)

    print(f"\n{'Model':<28} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*50)
    for name, m in all_results.items():
        tag = " ★" if name in {"Voting Hard","Voting Soft","Stacking"} else ""
        print(f"{name+tag:<28} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

    with open(os.path.join(OUT_DIR,'results.json'), 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "results": all_results}, f, indent=2)
    print("\n✅ Ensemble training complete!")
