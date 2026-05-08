"""
preprocessing.py
================
Full preprocessing pipeline for the RACE dataset.
Supports:
  - Raw CSV loading
  - Text cleaning
  - TF-IDF feature extraction (per option-level row)
  - One-Hot Encoding (OHE) feature extraction
  - Class-imbalance handling (SMOTE / class weights)
  - Label encoding (A/B/C/D → 0/1/2/3)
  - Save/load processed artefacts
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz, load_npz

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(base_dir='../data/raw'):
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    val_df   = pd.read_csv(os.path.join(base_dir, 'dev.csv'))
    test_df  = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    print(f"Loaded → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# 3. EXPAND TO OPTION-LEVEL ROWS
#    Each question gets 4 rows (one per option).
#    label = 1 if this option is the correct answer else 0
#    (For multi-class: label = A/B/C/D index per question)
# ─────────────────────────────────────────────
def expand_to_option_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each QA row into 4 option-level rows.
    Columns produced:
        article_clean, question_clean, option_clean,
        option_letter, is_correct (0/1), answer (A/B/C/D)
    """
    records = []
    for _, row in df.iterrows():
        article_c  = clean_text(row['article'])
        question_c = clean_text(row['question'])
        answer     = str(row['answer']).strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            opt_text = clean_text(row[letter])
            records.append({
                'article_clean':  article_c,
                'question_clean': question_c,
                'option_clean':   opt_text,
                'option_letter':  letter,
                'is_correct':     int(letter == answer),
                'answer':         answer,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING  ─ TF-IDF (primary)
# ─────────────────────────────────────────────
def build_tfidf_features(train_opt, val_opt, test_opt,
                          max_features=5000):
    """
    Build TF-IDF features for article+question and for the option text.
    Returns sparse matrices (saves memory).
    """
    # Combined text for the context column
    def combine(df):
        return (df['article_clean'] + ' ' + df['question_clean']).tolist()

    tfidf_ctx = TfidfVectorizer(max_features=max_features,
                                 ngram_range=(1, 2),
                                 sublinear_tf=True)
    tfidf_opt = TfidfVectorizer(max_features=max_features // 2,
                                 ngram_range=(1, 2),
                                 sublinear_tf=True)

    X_ctx_train = tfidf_ctx.fit_transform(combine(train_opt))
    X_ctx_val   = tfidf_ctx.transform(combine(val_opt))
    X_ctx_test  = tfidf_ctx.transform(combine(test_opt))

    X_opt_train = tfidf_opt.fit_transform(train_opt['option_clean'].tolist())
    X_opt_val   = tfidf_opt.transform(val_opt['option_clean'].tolist())
    X_opt_test  = tfidf_opt.transform(test_opt['option_clean'].tolist())

    X_train = hstack([X_ctx_train, X_opt_train])
    X_val   = hstack([X_ctx_val,   X_opt_val])
    X_test  = hstack([X_ctx_test,  X_opt_test])

    return X_train, X_val, X_test, tfidf_ctx, tfidf_opt


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING  ─ OHE (classical)
# ─────────────────────────────────────────────
def build_ohe_features(train_opt, val_opt, test_opt,
                        max_features=500):
    """
    One-Hot Encoding over the top-N vocabulary tokens.
    """
    def tokenize(text):
        return re.findall(r'\b[a-z]+\b', text)

    # Build vocabulary from training set only
    counter = Counter()
    for col in ['article_clean', 'question_clean', 'option_clean']:
        for text in train_opt[col]:
            counter.update(tokenize(text))
    vocab = [w for w, _ in counter.most_common(max_features)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    def to_ohe(row):
        combined = (row['article_clean'] + ' ' +
                    row['question_clean'] + ' ' +
                    row['option_clean'])
        vec = np.zeros(len(vocab_index), dtype=np.float32)
        for w in tokenize(combined):
            if w in vocab_index:
                vec[vocab_index[w]] = 1.0
        return vec

    def df_to_ohe(df):
        return np.array([to_ohe(r) for _, r in df.iterrows()],
                        dtype=np.float32)

    print("Building OHE for train …")
    X_train = df_to_ohe(train_opt)
    print("Building OHE for val …")
    X_val   = df_to_ohe(val_opt)
    print("Building OHE for test …")
    X_test  = df_to_ohe(test_opt)

    return X_train, X_val, X_test, vocab


# ─────────────────────────────────────────────
# 6. LABEL ENCODING
# ─────────────────────────────────────────────
def encode_labels(df, le=None, col='answer'):
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(df[col])
    else:
        y = le.transform(df[col])
    return y, le


# ─────────────────────────────────────────────
# 7. SAVE / LOAD ARTEFACTS
# ─────────────────────────────────────────────
def save_processed(X_train, y_train,
                   X_val,   y_val,
                   X_test,  y_test,
                   tfidf_ctx, tfidf_opt, vocab, le,
                   out_dir='../data/processed'):
    os.makedirs(out_dir, exist_ok=True)

    # Labels (always dense)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(out_dir, 'y_test.npy'),  y_test)

    # Features — handle sparse vs dense
    for tag, X in [('train', X_train), ('val', X_val), ('test', X_test)]:
        try:
            from scipy.sparse import issparse
            if issparse(X):
                save_npz(os.path.join(out_dir, f'X_{tag}.npz'), X)
            else:
                np.save(os.path.join(out_dir, f'X_{tag}.npy'), X)
        except Exception:
            np.save(os.path.join(out_dir, f'X_{tag}.npy'), X)

    joblib.dump(tfidf_ctx, os.path.join(out_dir, 'tfidf_ctx.pkl'))
    joblib.dump(tfidf_opt, os.path.join(out_dir, 'tfidf_opt.pkl'))
    joblib.dump(vocab,     os.path.join(out_dir, 'vocab.pkl'))
    joblib.dump(le,        os.path.join(out_dir, 'label_encoder.pkl'))
    print(f"Saved all artefacts to {out_dir}")


def load_processed(out_dir='../data/processed'):
    import os
    from scipy.sparse import load_npz, issparse

    arrays = {}
    for tag in ['train', 'val', 'test']:
        npz_path = os.path.join(out_dir, f'X_{tag}.npz')
        npy_path = os.path.join(out_dir, f'X_{tag}.npy')
        if os.path.exists(npz_path):
            arrays[f'X_{tag}'] = load_npz(npz_path)
        else:
            arrays[f'X_{tag}'] = np.load(npy_path)
        arrays[f'y_{tag}'] = np.load(os.path.join(out_dir, f'y_{tag}.npy'))

    tfidf_ctx = joblib.load(os.path.join(out_dir, 'tfidf_ctx.pkl'))
    tfidf_opt = joblib.load(os.path.join(out_dir, 'tfidf_opt.pkl'))
    vocab     = joblib.load(os.path.join(out_dir, 'vocab.pkl'))
    le        = joblib.load(os.path.join(out_dir, 'label_encoder.pkl'))

    return arrays, tfidf_ctx, tfidf_opt, vocab, le


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=== RACE Preprocessing Pipeline ===\n")
    train_df, val_df, test_df = load_data()

    print("\nExpanding to option-level rows …")
    train_opt = expand_to_option_level(train_df)
    val_opt   = expand_to_option_level(val_df)
    test_opt  = expand_to_option_level(test_df)
    print(f"Option rows → Train: {len(train_opt)} | "
          f"Val: {len(val_opt)} | Test: {len(test_opt)}")

    print("\nBuilding TF-IDF features …")
    X_train, X_val, X_test, tfidf_ctx, tfidf_opt = build_tfidf_features(
        train_opt, val_opt, test_opt, max_features=5000
    )
    print(f"TF-IDF shapes → Train: {X_train.shape} | "
          f"Val: {X_val.shape} | Test: {X_test.shape}")

    print("\nEncoding labels …")
    y_train, le = encode_labels(train_opt, col='answer')
    y_val,   _  = encode_labels(val_opt,   le, col='answer')
    y_test,  _  = encode_labels(test_opt,  le, col='answer')

    print(f"Classes: {le.classes_}")

    print("\nSaving artefacts …")
    save_processed(
        X_train, y_train, X_val, y_val, X_test, y_test,
        tfidf_ctx, tfidf_opt, [], le
    )
    print("\n✅ Preprocessing complete!")