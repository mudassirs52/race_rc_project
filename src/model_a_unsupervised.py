"""
model_a_unsupervised.py
========================
Model A — Unsupervised & Semi-Supervised Approaches (20 marks)

Implements:
  1. K-Means Clustering  (unsupervised)
  2. Gaussian Mixture Models / EM  (unsupervised)
  3. Label Spreading  (semi-supervised)
  4. Self-Training  (semi-supervised)

All evaluated with Accuracy, Macro-F1, and compared to supervised baselines.
"""

import numpy as np
import joblib, os, json, time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.sparse import load_npz, issparse

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, adjusted_rand_score, silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR   = '../data/processed'
OUT_DIR    = '../models/model_a/unsupervised'
CKPT_DIR   = '../models/model_a/checkpoints'

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    def load_X(tag):
        p = os.path.join(DATA_DIR, f'X_{tag}.npz')
        return load_npz(p) if os.path.exists(p) \
               else np.load(os.path.join(DATA_DIR, f'X_{tag}.npy'))

    X_tr = load_X('train'); X_va = load_X('val'); X_te = load_X('test')
    y_tr = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_va = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    ans_va = np.load(os.path.join(DATA_DIR, 'ans_val.npy'))
    le   = joblib.load(os.path.join(DATA_DIR, 'label_encoder.pkl'))
    return X_tr, y_tr, X_va, y_va, ans_va, le


# ─────────────────────────────────────────────
# DIMENSIONALITY REDUCTION (sparse → dense)
# ─────────────────────────────────────────────
def reduce_dimensions(X_tr, X_va, n_components=100, ckpt=None):
    """TruncatedSVD (LSA) to convert sparse TF-IDF → dense for clustering."""
    ckpt_path = ckpt or os.path.join(CKPT_DIR, 'svd_reducer.pkl')
    if os.path.exists(ckpt_path):
        print("  Loading SVD reducer from checkpoint…")
        svd = joblib.load(ckpt_path)
        return normalize(svd.transform(X_tr)), normalize(svd.transform(X_va)), svd

    print(f"  Fitting TruncatedSVD (n={n_components})…")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tr_d = normalize(svd.fit_transform(X_tr))
    X_va_d = normalize(svd.transform(X_va))
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump(svd, ckpt_path)
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    return X_tr_d, X_va_d, svd


# ─────────────────────────────────────────────
# HELPER: map cluster IDs → class labels (majority vote)
# ─────────────────────────────────────────────
def map_clusters_to_labels(cluster_ids, y_true, n_clusters):
    """Assign each cluster the majority true label."""
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() == 0:
            mapping[c] = 0
            continue
        labels, counts = np.unique(y_true[mask], return_counts=True)
        mapping[c] = labels[np.argmax(counts)]
    return np.array([mapping[c] for c in cluster_ids])


def evaluate_group(y_pred_options, ans_va, name, le, y_va, val_clusters=None):
    # Group options by 4
    y_pred_options = y_pred_options.reshape(-1, 4)
    # Pick the first option that was predicted as 1
    y_pred_ans = np.argmax(y_pred_options, axis=1)

    acc = accuracy_score(ans_va, y_pred_ans)
    f1  = f1_score(ans_va, y_pred_ans, average='macro', zero_division=0)
    
    ari = 0
    if val_clusters is not None:
        ari = adjusted_rand_score(y_va, val_clusters)

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro F1   : {f1:.4f}")
    if ari: print(f"  ARI        : {ari:.4f}")
    print(classification_report(ans_va, y_pred_ans, target_names=le.classes_, zero_division=0))

    _plot_cm(ans_va, y_pred_ans, le, name)
    return {"accuracy": round(acc,4), "macro_f1": round(f1,4), "ari": round(ari,4)}, y_pred_ans


# ─────────────────────────────────────────────
# 1. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def run_kmeans(X_tr_d, y_tr, X_va_d, y_va, ans_va, le, n_clusters=4):
    print("\n── K-Means Clustering ──")
    ckpt = os.path.join(CKPT_DIR, 'kmeans.pkl')

    if os.path.exists(ckpt):
        km = joblib.load(ckpt)
        print("  Loaded from checkpoint.")
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             batch_size=1024, n_init=10)
        km.fit(X_tr_d)
        joblib.dump(km, ckpt)

    val_clusters   = km.predict(X_va_d)
    y_pred_opt = map_clusters_to_labels(val_clusters, y_va, n_clusters)
    m, y_pred = evaluate_group(y_pred_opt, ans_va, "K-Means", le, y_va, val_clusters)
    return m, y_pred


# ─────────────────────────────────────────────
# 2. GAUSSIAN MIXTURE MODEL (EM)
# ─────────────────────────────────────────────
def run_gmm(X_tr_d, y_tr, X_va_d, y_va, ans_va, le, n_components=4):
    print("\n── Gaussian Mixture Model (EM) ──")
    ckpt = os.path.join(CKPT_DIR, 'gmm.pkl')

    if os.path.exists(ckpt):
        gmm = joblib.load(ckpt)
        print("  Loaded from checkpoint.")
    else:
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                               random_state=42, max_iter=200, n_init=3)
        gmm.fit(X_tr_d)
        joblib.dump(gmm, ckpt)

    val_clusters = gmm.predict(X_va_d)
    y_pred_opt = map_clusters_to_labels(val_clusters, y_va, n_components)
    m, y_pred = evaluate_group(y_pred_opt, ans_va, "GMM-EM", le, y_va, val_clusters)
    return m, y_pred


# ─────────────────────────────────────────────
# 3. LABEL SPREADING (Semi-Supervised)
# ─────────────────────────────────────────────
def run_label_spreading(X_tr_d, y_tr, X_va_d, y_va, ans_va, le,
                         labeled_fraction=0.10):
    print(f"\n── Label Spreading (labeled={labeled_fraction*100:.0f}%) ──")
    ckpt = os.path.join(CKPT_DIR, f'label_spreading_{int(labeled_fraction*100)}.pkl')

    if os.path.exists(ckpt):
        ls = joblib.load(ckpt)
        print("  Loaded from checkpoint.")
    else:
        n = len(y_tr)
        n_labeled = int(n * labeled_fraction)
        y_semi = np.full(n, -1, dtype=int)
        labeled_idx = np.random.RandomState(42).choice(n, n_labeled, replace=False)
        y_semi[labeled_idx] = y_tr[labeled_idx]

        print(f"  Labeled: {n_labeled}/{n} samples. Fitting…", end=' ', flush=True)
        ls = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2,
                             max_iter=100, n_jobs=-1)
        ls.fit(X_tr_d, y_semi)
        print("done.")
        joblib.dump(ls, ckpt)

    y_pred_opt = ls.predict_proba(X_va_d)[:, 1]
    m, y_pred = evaluate_group(y_pred_opt, ans_va, f"Label Spreading ({int(labeled_fraction*100)}pct)", le, y_va)
    m["labeled_pct"] = labeled_fraction
    return m, y_pred


# ─────────────────────────────────────────────
# 4. SELF-TRAINING (Semi-Supervised)
# ─────────────────────────────────────────────
def run_self_training(X_tr, y_tr, X_va, y_va, ans_va, le,
                       labeled_fraction=0.10):
    print(f"\n── Self-Training (labeled={labeled_fraction*100:.0f}%) ──")
    ckpt = os.path.join(CKPT_DIR, f'self_training_{int(labeled_fraction*100)}.pkl')

    if os.path.exists(ckpt):
        clf = joblib.load(ckpt)
        print("  Loaded from checkpoint.")
    else:
        n = X_tr.shape[0]
        n_labeled = int(n * labeled_fraction)
        y_semi = np.full(n, -1, dtype=int)
        labeled_idx = np.random.RandomState(42).choice(n, n_labeled, replace=False)
        y_semi[labeled_idx] = y_tr[labeled_idx]

        base = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                   solver='lbfgs', random_state=42, n_jobs=-1)
        clf = SelfTrainingClassifier(base, threshold=0.85, max_iter=10, verbose=True)
        print(f"  Fitting Self-Training ({n_labeled} labeled)…")
        clf.fit(X_tr, y_semi)
        joblib.dump(clf, ckpt)

    y_pred_opt = clf.predict_proba(X_va)[:, 1]
    m, y_pred = evaluate_group(y_pred_opt, ans_va, f"Self-Training ({int(labeled_fraction*100)}pct)", le, y_va)
    m["labeled_pct"] = labeled_fraction
    return m, y_pred


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOTTER
# ─────────────────────────────────────────────
def _plot_cm(y_true, y_pred, le, name):
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}', fontsize=13)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    safe = name.replace(' ','_').replace('(','').replace(')','').replace('%','pct')
    plt.savefig(os.path.join(OUT_DIR, 'plots', f'{safe}_cm.png'), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# COMPARISON CHART
# ─────────────────────────────────────────────
def plot_unsupervised_comparison(all_results, supervised_acc=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    names = list(all_results.keys())
    accs  = [m['accuracy'] for m in all_results.values()]
    f1s   = [m['macro_f1'] for m in all_results.values()]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.5), 5))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label='Accuracy', color='#7C4DFF', alpha=0.85)
    bars2 = ax.bar(x + 0.2, f1s,  0.35, label='Macro F1', color='#00BCD4', alpha=0.85)
    if supervised_acc:
        ax.axhline(supervised_acc, color='#FF5722', linestyle='--',
                   label=f'Supervised LR ({supervised_acc:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.0)
    ax.set_title('Unsupervised & Semi-Supervised vs Supervised Baseline')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'unsupervised_comparison.png'), dpi=150)
    plt.show()
    print(f"  Chart saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("="*60)
    print("  Model A — Unsupervised & Semi-Supervised")
    print("="*60)

    print("\nLoading data…")
    X_tr, y_tr, X_va, y_va, ans_va, le = load_data()

    print("\nReducing dimensions (TruncatedSVD/LSA)…")
    X_tr_d, X_va_d, svd = reduce_dimensions(X_tr, X_va, n_components=100)

    all_results = {}

    # 1. K-Means
    m, _ = run_kmeans(X_tr_d, y_tr, X_va_d, y_va, ans_va, le)
    all_results["K-Means"] = m

    # 2. GMM/EM
    m, _ = run_gmm(X_tr_d, y_tr, X_va_d, y_va, ans_va, le)
    all_results["GMM-EM"] = m

    # 3. Label Spreading — 10% labeled
    m, _ = run_label_spreading(X_tr_d, y_tr, X_va_d, y_va, ans_va, le, labeled_fraction=0.10)
    all_results["LabelSpreading (10%)"] = m

    # 4. Label Spreading — 30% labeled
    m, _ = run_label_spreading(X_tr_d, y_tr, X_va_d, y_va, ans_va, le, labeled_fraction=0.30)
    all_results["LabelSpreading (30%)"] = m

    # 5. Self-Training — 10% labeled
    m, _ = run_self_training(X_tr, y_tr, X_va, y_va, ans_va, le, labeled_fraction=0.10)
    all_results["SelfTraining (10%)"] = m

    # 6. Self-Training — 30% labeled
    m, _ = run_self_training(X_tr, y_tr, X_va, y_va, ans_va, le, labeled_fraction=0.30)
    all_results["SelfTraining (30%)"] = m

    # Load supervised baseline if available
    sup_acc = None
    sup_ckpt = os.path.join(CKPT_DIR, 'Logistic_Regression.pkl')
    if os.path.exists(sup_ckpt):
        sup_model = joblib.load(sup_ckpt)
        probs = sup_model.predict_proba(X_va)[:, 1]
        probs = probs.reshape(-1, 4)
        sup_pred = np.argmax(probs, axis=1)
        sup_acc   = accuracy_score(ans_va, sup_pred)
        print(f"\nSupervised LR baseline (val): {sup_acc:.4f}")

    plot_unsupervised_comparison(all_results, sup_acc)

    # Summary table
    print(f"\n{'Model':<28} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*50)
    for name, m in all_results.items():
        print(f"{name:<28} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
    if sup_acc:
        print(f"{'[Supervised] LR (100%)':<28} {sup_acc:>10.4f}         —")

    # Save
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "results": all_results}, f, indent=2)
    print("\n✅ Unsupervised/Semi-supervised training complete!")
