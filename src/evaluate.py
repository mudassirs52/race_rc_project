"""
evaluate.py
===========
Unified evaluation script for Model A and Model B.

Model A:  Accuracy, Macro-F1, Exact Match, Confusion Matrix
          + comparison vs BERT/T5 baselines
Model B:  BLEU, ROUGE-L, METEOR (distractor quality)
          Precision@K (hint extraction quality)
"""

import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


# ─────────────────────────────────────────────────────
# MODEL A EVALUATION
# ─────────────────────────────────────────────────────
BERT_BASELINES = {
    "Random Chance":                    {"accuracy": 0.250, "macro_f1": 0.250},
    "BERT-base (Liu et al., 2019)":     {"accuracy": 0.665, "macro_f1": 0.660},
    "BERT-large (Liu et al., 2019)":    {"accuracy": 0.727, "macro_f1": 0.723},
    "T5-base (Khashabi et al., 2020)":  {"accuracy": 0.755, "macro_f1": 0.752},
}


def load_model_a_artifacts(data_dir='../data/processed',
                            model_dir='../models/model_a/checkpoints'):
    """Load feature matrices + all trained Model A checkpoints."""
    def load_X(tag):
        p = os.path.join(data_dir, f'X_{tag}.npz')
        return load_npz(p) if os.path.exists(p) \
               else np.load(os.path.join(data_dir, f'X_{tag}.npy'))

    X_test = load_X('test')
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    le     = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))

    models = {}
    if os.path.isdir(model_dir):
        for fname in os.listdir(model_dir):
            if fname.endswith('.pkl'):
                name = fname.replace('.pkl', '').replace('_', ' ').title()
                models[name] = joblib.load(os.path.join(model_dir, fname))
    return X_test, y_test, le, models


def evaluate_model_a(X_test, y_test, le, models,
                      out_dir='../models/model_a'):
    """Evaluate all Model A checkpoints on test set."""
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        em   = float(np.mean(y_test == y_pred))

        print(f"\n{'─'*60}")
        print(f"  {name}  [TEST SET]")
        print(f"{'─'*60}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Macro F1    : {f1:.4f}")
        print(f"  Precision   : {prec:.4f}")
        print(f"  Recall      : {rec:.4f}")
        print(f"  Exact Match : {em:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                     target_names=le.classes_,
                                     zero_division=0))

        # Confusion matrix
        cm_dir = os.path.join(out_dir, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix — {name} (Test)', fontsize=13)
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout()
        safe = name.replace(' ', '_')
        plt.savefig(os.path.join(cm_dir, f'{safe}_test_cm.png'), dpi=150)
        plt.close()

        all_results[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1,  4),
            "precision":round(prec,4),
            "recall":   round(rec, 4),
            "exact_match": round(em, 4),
        }

    # Comparison table
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON — Traditional ML vs BERT/T5 Baselines")
    print(f"{'='*70}")
    print(f"{'Model':<38} {'Accuracy':>10} {'Macro F1':>10} {'EM':>8}")
    print(f"{'─'*70}")
    for name, m in all_results.items():
        print(f"{'[Ours] '+name:<38} "
              f"{m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['exact_match']:>8.4f}")
    print(f"{'─'*70}")
    for name, m in BERT_BASELINES.items():
        print(f"{'[Baseline] '+name:<38} "
              f"{m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}     —")

    # Bar chart: accuracy comparison
    _plot_comparison(all_results, BERT_BASELINES, out_dir)

    # Save results
    payload = {
        "timestamp": datetime.now().isoformat(),
        "traditional_models": all_results,
        "bert_baselines": BERT_BASELINES,
    }
    path = os.path.join(out_dir, 'test_results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Model A test results saved → {path}")
    return all_results


def _plot_comparison(our_results, baselines, out_dir):
    """Bar chart: our models vs BERT baselines."""
    names, accs = [], []
    for n, m in our_results.items():
        names.append('[Ours]\n' + n)
        accs.append(m['accuracy'])
    for n, m in baselines.items():
        names.append('[Base]\n' + n)
        accs.append(m['accuracy'])

    colors = ['#2196F3']*len(our_results) + ['#FF7043']*len(baselines)
    plt.figure(figsize=(max(10, len(names)*1.8), 6))
    bars = plt.bar(names, accs, color=colors, edgecolor='white', width=0.6)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.axhline(0.25, color='gray', linestyle='--', label='Random Chance (25%)')
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model A — Accuracy Comparison (Traditional ML vs BERT/T5)', fontsize=13)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    print(f"  Comparison chart saved.")


# ─────────────────────────────────────────────────────
# MODEL B EVALUATION
# ─────────────────────────────────────────────────────

def evaluate_model_b_on_dataset(df, cv, method='tfidf', n_samples=200,
                                  out_dir='../models/model_b'):
    """
    Run Model B on a sample of the RACE dataset and report BLEU/ROUGE/METEOR.
    df must have columns: article, question, A, B, C, D, answer
    """
    from src.model_b_distractor_hint import (
        generate_distractors_and_hints,
        evaluate_distractors,
        evaluate_hints_precision,
    )

    os.makedirs(out_dir, exist_ok=True)
    sample = df.sample(min(n_samples, len(df)), random_state=42)

    all_bleu, all_rouge, all_meteor, all_hint_prec = [], [], [], []

    for _, row in sample.iterrows():
        passage  = str(row['article'])
        question = str(row['question'])
        answer   = str(row['answer']).strip().upper()
        correct  = str(row[answer]) if answer in 'ABCD' else ''
        refs = [str(row[l]) for l in 'ABCD' if l != answer]

        result = generate_distractors_and_hints(
            passage, question, correct, cv, method)
        gen = result['distractors']
        scores = evaluate_distractors(gen, refs)
        all_bleu.append(scores['BLEU'])
        all_rouge.append(scores['ROUGE-L'])
        all_meteor.append(scores['METEOR'])
        hint_prec = evaluate_hints_precision(result['hints'], passage)
        all_hint_prec.append(hint_prec)

    final = {
        "method": method,
        "n_samples": len(sample),
        "BLEU":          round(np.mean(all_bleu),      4),
        "ROUGE-L":       round(np.mean(all_rouge),     4),
        "METEOR":        round(np.mean(all_meteor),    4),
        "Hint Prec@3":   round(np.mean(all_hint_prec), 4),
    }

    print(f"\n{'='*55}")
    print(f"  Model B Evaluation — {method.upper()}")
    print(f"{'='*55}")
    for k, v in final.items():
        print(f"  {k:<20}: {v}")

    path = os.path.join(out_dir, f'eval_{method}.json')
    with open(path, 'w') as f:
        json.dump({**final, "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"\n✅ Model B results saved → {path}")
    return final


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Unified Evaluation Script")
    print("=" * 60)

    # ── Model A ──
    print("\n[1/2] Evaluating Model A on test set …")
    try:
        X_test, y_test, le, models = load_model_a_artifacts()
        if models:
            evaluate_model_a(X_test, y_test, le, models)
        else:
            print("  No Model A checkpoints found. Run model_a_train.py first.")
    except FileNotFoundError as e:
        print(f"  Skipping Model A eval: {e}")

    # ── Model B ──
    print("\n[2/2] Evaluating Model B …")
    try:
        import pandas as pd
        from src.model_b_distractor_hint import CorpusVocab
        df_test = pd.read_csv('../data/raw/test.csv')
        # Build vocab from test corpus
        cv = CorpusVocab(max_features=3000).fit(
            (df_test['article'] + ' ' + df_test['question']).tolist())
        for method in ['tfidf', 'ohe', 'frequency']:
            evaluate_model_b_on_dataset(df_test, cv, method=method)
    except FileNotFoundError as e:
        print(f"  Skipping Model B eval: {e}")
