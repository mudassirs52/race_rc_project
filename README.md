# RACE Reading Comprehension — AL2002 Lab Project

## Overview
Two-model NLP pipeline on the RACE dataset using **traditional ML only** (no neural networks).

| Component | Task | Models |
|-----------|------|--------|
| **Model A** | Answer Verifier | Logistic Regression, Random Forest, Linear SVM, Complement NB |
| **Model B** | Distractor & Hint Generator | TF-IDF cosine, OHE cosine, Frequency-based |

Evaluation: **BLEU, ROUGE-L, METEOR** (Model B) · **Accuracy, Macro-F1, EM** (Model A)

---

## Quick Start

### Option 1: Google Colab (Recommended)
1. Upload `colab_train.py` to Colab
2. Run cells top to bottom
3. Download checkpoints from `models/model_a/checkpoints/` to Drive after each model

### Option 2: Local
```bash
pip install -r requirements.txt
python src/data_download.py      # download RACE
python src/preprocessing.py     # build TF-IDF features
python src/model_a_train.py     # train Model A
python src/evaluate.py          # evaluate both models
streamlit run app.py            # launch UI
```

---

## Project Structure
```
race_rc_project/
├── app.py                        # Streamlit UI (4 screens)
├── colab_train.py                # All-in-one Colab script
├── requirements.txt
├── src/
│   ├── preprocessing.py          # TF-IDF + OHE feature pipeline
│   ├── model_a_train.py          # Model A: answer verifier
│   ├── model_b_distractor_hint.py# Model B: distractors + hints
│   ├── evaluate.py               # BLEU/ROUGE/METEOR/F1 evaluation
│   └── data_download.py          # HuggingFace RACE downloader
├── data/raw/                     # train.csv, dev.csv, test.csv
├── data/processed/               # X_train.npz, tfidf_*.pkl, etc.
└── models/model_a/
    ├── checkpoints/              # .pkl model files (resumable)
    ├── confusion_matrices/       # per-model CM plots
    └── results.json              # metrics vs BERT baselines
```

---

## Checkpointing (for Colab compute limits)
Every model is saved to `models/model_a/checkpoints/<ModelName>.pkl` immediately after training.  
If Colab disconnects, re-run `train_model_a()`— it will **skip already-trained models** and load from checkpoint.

Mount Google Drive first:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then copy checkpoints there for persistence.

---

## Baselines (BERT/T5 from literature)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Random Chance | 25.0% | 25.0% |
| BERT-base (Liu 2019) | 66.5% | 66.0% |
| BERT-large (Liu 2019) | 72.7% | 72.3% |
| T5-base (Khashabi 2020) | 75.5% | 75.2% |
| **Our LR (TF-IDF)** | ~42–48% | ~40–46% |

---

## GCR Compliance
- ✅ Traditional ML only (no neural networks)
- ✅ Option-level training
- ✅ Class imbalance handled via `class_weight='balanced'`
- ✅ Confusion matrix, Precision, Recall, F1 reported
- ✅ BLEU, ROUGE, METEOR for Model B
- ✅ Checkpoint support for Colab compute limits
- ✅ Compared against BERT/T5 baselines
