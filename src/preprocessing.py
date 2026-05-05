import pandas as pd
import numpy as np
import ast
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#1. LOAD DATA
def load_data():
    train_df = pd.read_csv('../data/raw/train.csv')
    val_df = pd.read_csv('../data/raw/val.csv')
    test_df = pd.read_csv('../data/raw/test.csv')
    return train_df, val_df, test_df

#2. PARSE OPTIONS 
def parse_options(df):
    import re
    df = df.copy()
    
    def parse_option(opt_str):
        items = re.findall(r"'(.*?)'", str(opt_str))
        return items
    
    df['options_parsed'] = df['options'].apply(parse_option)
    df = df[df['options_parsed'].apply(lambda x: len(x) == 4)].reset_index(drop=True)
    df['A'] = df['options_parsed'].apply(lambda x: x[0])
    df['B'] = df['options_parsed'].apply(lambda x: x[1])
    df['C'] = df['options_parsed'].apply(lambda x: x[2])
    df['D'] = df['options_parsed'].apply(lambda x: x[3])
    return df

#3. CLEAN TEXT
def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#4. FEATURE ENGINEERING
def extract_features(df):
    df = df.copy()
    df['article_clean'] = df['article'].apply(clean_text)
    df['question_clean'] = df['question'].apply(clean_text)
    df['A_clean'] = df['A'].apply(clean_text)
    df['B_clean'] = df['B'].apply(clean_text)
    df['C_clean'] = df['C'].apply(clean_text)
    df['D_clean'] = df['D'].apply(clean_text)

    df['article_length'] = df['article'].apply(lambda x: len(str(x).split()))
    df['question_length'] = df['question'].apply(lambda x: len(str(x).split()))
    df['level'] = df['example_id'].apply(lambda x: 1 if 'high' in str(x) else 0)

    return df

#5. ONE HOT ENCODING
def build_ohe_features(df, vocab=None, max_features=5000):
    from collections import Counter
    import re

    def tokenize(text):
        return re.findall(r'\b[a-z]+\b', str(text).lower())

    if vocab is None:
        all_words = []
        for col in ['article_clean', 'question_clean']:
            for text in df[col]:
                all_words.extend(tokenize(text))
        counter = Counter(all_words)
        vocab = [w for w, _ in counter.most_common(max_features)]

    vocab_index = {w: i for i, w in enumerate(vocab)}

    def text_to_ohe(text):
        vec = np.zeros(len(vocab_index))
        for word in tokenize(text):
            if word in vocab_index:
                vec[vocab_index[word]] = 1
        return vec

    combined = (df['article_clean'] + ' ' + df['question_clean']).tolist()
    X = np.array([text_to_ohe(t) for t in combined])

    return X, vocab

# 6. LABEL ENCODING
def encode_labels(df):
    le = LabelEncoder()
    y = le.fit_transform(df['answer'])
    return y, le

#7. SAVE PROCESSED DATA
def save_processed(X_train, y_train, X_val, y_val, X_test, y_test, vocab, le):
    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/X_val.npy', X_val)
    np.save('../data/processed/y_val.npy', y_val)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_test.npy', y_test)
    joblib.dump(vocab, '../data/processed/vocab.pkl')
    joblib.dump(le, '../data/processed/label_encoder.pkl')
    print("Saved!")

#8. MAIN
if __name__ == '__main__':
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    print("Parsing options...")
    train_df = parse_options(train_df)
    val_df = parse_options(val_df)
    test_df = parse_options(test_df)

    print("Extracting features...")
    train_df = extract_features(train_df)
    val_df = extract_features(val_df)
    test_df = extract_features(test_df)

    print(f"Train size after parsing: {len(train_df)}")
    print(f"Val size after parsing: {len(val_df)}")
    print(f"Test size after parsing: {len(test_df)}")

    print("Building OHE features...")
    X_train, vocab = build_ohe_features(train_df, max_features=500)
    X_val, _ = build_ohe_features(val_df, vocab=vocab)
    X_test, _ = build_ohe_features(test_df, vocab=vocab)

    print("Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['answer'])
    y_val = le.transform(val_df['answer'])
    y_test = le.transform(test_df['answer'])

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("Saving...")
    save_processed(X_train, y_train, X_val, y_val, X_test, y_test, vocab, le)
    print("Preprocessing complete!")