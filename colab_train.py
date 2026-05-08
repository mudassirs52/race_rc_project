# ============================================================
# RACE RC Project — Google Colab Training Script
# Run cell by cell in Colab (T4 GPU recommended)
# ============================================================

# CELL 1: Install dependencies
"""
!pip install datasets scikit-learn imbalanced-learn nltk rouge-score sacrebleu gensim tqdm joblib plotly streamlit -q
"""

# CELL 2: Mount Drive & clone repo (if using Drive for checkpoints)
"""
from google.colab import drive
drive.mount('/content/drive')
import os
CHECKPOINT_DIR = '/content/drive/MyDrive/race_rc_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
"""

# CELL 3: Download RACE dataset
import os, json, re, math, time
import numpy as np
import pandas as pd
from collections import Counter

def download_race(out_dir='data/raw'):
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("race", "all")
    def save_split(split, fname):
        rows = []
        for item in ds[split]:
            opts = item['options']
            rows.append({'article': item['article'], 'question': item['question'],
                         'A': opts[0], 'B': opts[1], 'C': opts[2], 'D': opts[3],
                         'answer': item['answer']})
        pd.DataFrame(rows).to_csv(fname, index=False)
        print(f"  {fname}: {len(rows)} rows")
    save_split('train', f'{out_dir}/train.csv')
    save_split('validation', f'{out_dir}/dev.csv')
    save_split('test', f'{out_dir}/test.csv')
    print("Done!")

# CELL 4: Preprocessing
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def expand_option_level(df):
    rows = []
    for _, r in df.iterrows():
        art = clean(r['article']); q = clean(r['question'])
        ans = str(r['answer']).strip().upper()
        for L in 'ABCD':
            rows.append({'ctx': art + ' ' + q,
                         'option': clean(r[L]),
                         'answer': ans,
                         'is_correct': int(L == ans)})
    return pd.DataFrame(rows)

def build_features(train_df, val_df, test_df, max_feat=5000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    import joblib, os

    tfidf_ctx = TfidfVectorizer(max_features=max_feat, ngram_range=(1,2), sublinear_tf=True)
    tfidf_opt = TfidfVectorizer(max_features=max_feat//2, ngram_range=(1,2), sublinear_tf=True)

    Xc_tr = tfidf_ctx.fit_transform(train_df['ctx'])
    Xc_va = tfidf_ctx.transform(val_df['ctx'])
    Xc_te = tfidf_ctx.transform(test_df['ctx'])

    Xo_tr = tfidf_opt.fit_transform(train_df['option'])
    Xo_va = tfidf_opt.transform(val_df['option'])
    Xo_te = tfidf_opt.transform(test_df['option'])

    X_tr = hstack([Xc_tr, Xo_tr]); X_va = hstack([Xc_va, Xo_va]); X_te = hstack([Xc_te, Xo_te])

    os.makedirs('data/processed', exist_ok=True)
    from scipy.sparse import save_npz
    save_npz('data/processed/X_train.npz', X_tr)
    save_npz('data/processed/X_val.npz', X_va)
    save_npz('data/processed/X_test.npz', X_te)
    joblib.dump(tfidf_ctx, 'data/processed/tfidf_ctx.pkl')
    joblib.dump(tfidf_opt, 'data/processed/tfidf_opt.pkl')
    print(f"Features: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")
    return X_tr, X_va, X_te, tfidf_ctx, tfidf_opt

def encode_labels(train_df, val_df, test_df):
    from sklearn.preprocessing import LabelEncoder
    import joblib
    le = LabelEncoder()
    y_tr = le.fit_transform(train_df['answer'])
    y_va = le.transform(val_df['answer'])
    y_te = le.transform(test_df['answer'])
    joblib.dump(le, 'data/processed/label_encoder.pkl')
    np.save('data/processed/y_train.npy', y_tr)
    np.save('data/processed/y_val.npy', y_va)
    np.save('data/processed/y_test.npy', y_te)
    print(f"Classes: {le.classes_} | Train dist: {np.bincount(y_tr)}")
    return y_tr, y_va, y_te, le

# CELL 5: Train Model A
BERT_BASELINES = {
    "Random Chance":               {"accuracy": 0.250, "macro_f1": 0.250},
    "BERT-base (Liu 2019)":        {"accuracy": 0.665, "macro_f1": 0.660},
    "BERT-large (Liu 2019)":       {"accuracy": 0.727, "macro_f1": 0.723},
    "T5-base (Khashabi 2020)":     {"accuracy": 0.755, "macro_f1": 0.752},
}

def train_model_a(X_tr, y_tr, X_va, y_va, le,
                   ckpt_dir='models/model_a/checkpoints'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import ComplementNB
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    import joblib, os, matplotlib.pyplot as plt, seaborn as sns

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('models/model_a/plots', exist_ok=True)

    model_defs = {
        "Logistic_Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs',
            multi_class='multinomial', class_weight='balanced', random_state=42, n_jobs=-1),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
        "Linear_SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)),
        "Complement_NB": ComplementNB(alpha=0.1),
    }

    results = {}
    for name, model in model_defs.items():
        ckpt = os.path.join(ckpt_dir, f'{name}.pkl')
        if os.path.exists(ckpt):
            print(f"Loading checkpoint: {name}")
            model = joblib.load(ckpt)
        else:
            print(f"Training {name}...", end=' ')
            t0 = time.time()
            model.fit(X_tr, y_tr)
            print(f"{time.time()-t0:.1f}s")
            joblib.dump(model, ckpt)

        y_pred = model.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        f1  = f1_score(y_va, y_pred, average='macro', zero_division=0)
        em  = float(np.mean(y_va == y_pred))
        results[name] = {"accuracy": round(acc,4), "macro_f1": round(f1,4), "exact_match": round(em,4)}

        print(f"\n{name}: Acc={acc:.4f}  F1={f1:.4f}  EM={em:.4f}")
        print(classification_report(y_va, y_pred, target_names=le.classes_, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_va, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        fig.tight_layout()
        fig.savefig(f'models/model_a/plots/{name}_cm.png', dpi=150)
        plt.show()

    # Comparison table
    print("\n" + "="*65)
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'EM':>8}")
    print("-"*65)
    for n, m in results.items():
        print(f"{'[Ours] '+n:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['exact_match']:>8.4f}")
    print("-"*65)
    for n, m in BERT_BASELINES.items():
        print(f"{'[BERT] '+n:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}        —")

    # Bar chart
    names = [f"[Ours]\n{n}" for n in results] + [f"[Base]\n{n}" for n in BERT_BASELINES]
    accs  = [m['accuracy'] for m in results.values()] + [m['accuracy'] for m in BERT_BASELINES.values()]
    colors = ['#2196F3']*len(results) + ['#FF7043']*len(BERT_BASELINES)
    plt.figure(figsize=(12,5))
    bars = plt.bar(names, accs, color=colors, edgecolor='white')
    for b, a in zip(bars, accs):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{a:.3f}',
                 ha='center', fontsize=9)
    plt.axhline(0.25, color='gray', linestyle='--', label='Random (25%)')
    plt.ylabel('Accuracy'); plt.title('Traditional ML vs BERT/T5 Baselines')
    plt.ylim(0, 1.0); plt.legend(); plt.tight_layout()
    plt.savefig('models/model_a/accuracy_comparison.png', dpi=150)
    plt.show()

    import json
    with open('models/model_a/results.json','w') as f:
        json.dump({"traditional": results, "baselines": BERT_BASELINES}, f, indent=2)
    print("\n✅ Model A training complete!")
    return results

# CELL 6: Model B — Distractor & Hint Generation
STOP = {'a','an','the','is','are','was','were','be','to','of','in','on','at',
        'by','for','with','and','but','or','not','it','as','if','from','this','that'}

def tok(text): return re.findall(r'\b[a-z]+\b', str(text).lower())
def ctok(text): return [w for w in tok(text) if w not in STOP and len(w)>2]

def build_corpus_vocab(docs, max_feat=3000):
    df_cnt = Counter()
    N = len(docs)
    for d in docs:
        df_cnt.update(set(tok(d)))
    top = [w for w,_ in df_cnt.most_common(max_feat)]
    vocab = {w:i for i,w in enumerate(top)}
    idf   = {w: math.log((N+1)/(c+1))+1 for w,c in df_cnt.items() if w in vocab}
    return vocab, idf

def tfidf_vec(text, vocab, idf):
    tokens = tok(text); n = max(len(tokens),1)
    tf = Counter(tokens)
    v = np.zeros(len(vocab), dtype=np.float32)
    for w,c in tf.items():
        if w in vocab: v[vocab[w]] = (c/n)*idf.get(w,1.0)
    nm = np.linalg.norm(v); return v/nm if nm>0 else v

def ohe_vec(text, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    for w in tok(text):
        if w in vocab: v[vocab[w]] = 1.0
    nm = np.linalg.norm(v); return v/nm if nm>0 else v

def cosine(a, b):
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>0 else 0.0

def extract_candidates(passage, answer, top_k=30):
    words = tok(passage)
    freq = Counter(w for w in words if w not in STOP and len(w)>3)
    ans_toks = set(tok(answer))
    cands = set()
    for w,_ in freq.most_common(top_k): cands.add(w)
    for i in range(len(words)-1):
        if words[i] not in STOP and words[i+1] not in STOP:
            cands.add(f"{words[i]} {words[i+1]}")
    return [c for c in cands if not set(tok(c)).issubset(ans_toks)]

def get_distractors(passage, answer, vocab, idf, method='tfidf', n=3):
    cands = extract_candidates(passage, answer)
    if not cands: return ['N/A']*n
    ans_v = tfidf_vec(answer, vocab, idf) if method=='tfidf' else ohe_vec(answer, vocab)
    scored = [(cosine(tfidf_vec(c,vocab,idf) if method=='tfidf' else ohe_vec(c,vocab), ans_v), c)
              for c in cands]
    scored.sort(key=lambda x:-x[0])
    out, seen = [], set(tok(answer))
    for sim, c in scored:
        if len(out)>=n: break
        if 0.05<sim<0.95 and not set(tok(c)).issubset(seen):
            out.append(c); seen.update(tok(c))
    while len(out)<n: out.append('N/A')
    return out[:n]

def get_hints(passage, question, answer, n=3):
    q_words = set(ctok(question))
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage) if len(s.strip())>15]
    scored = [(len(q_words & set(ctok(s)))/(len(q_words)+1e-8), s) for s in sents]
    scored.sort(key=lambda x:-x[0])
    hints = [s for _,s in scored[:n+2]]
    hints.sort(key=lambda h: len(set(ctok(h)) & set(ctok(answer))))
    while len(hints)<n: hints.append("Re-read the passage carefully.")
    return hints[:n]

# CELL 7: Evaluate Model B with BLEU/ROUGE/METEOR
def ngrams(toks, n): return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def bleu(ref, hyp, max_n=4):
    r,h = tok(ref), tok(hyp)
    if not r or not h: return 0.0
    bp = math.exp(1-len(r)/len(h)) if len(h)<len(r) else 1.0
    precs = []
    for n in range(1,max_n+1):
        rn,hn = ngrams(r,n), ngrams(h,n)
        if not hn: precs.append(0.0); continue
        clip = sum(min(c,rn[g]) for g,c in hn.items())
        precs.append(clip/sum(hn.values()))
    if min(precs)==0: return 0.0
    return bp*math.exp(sum(math.log(p) for p in precs)/max_n)

def rouge_l(ref, hyp):
    r,h = tok(ref), tok(hyp)
    if not r or not h: return 0.0
    m,n = len(r),len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1]==h[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs=dp[m][n]; p=lcs/n; rec=lcs/m
    return (2*p*rec/(p+rec)) if p+rec>0 else 0.0

def meteor(ref, hyp):
    r,h = set(tok(ref)), tok(hyp)
    if not r or not h: return 0.0
    m = sum(1 for t in h if t in r)
    p=m/len(h); rec=m/len(r)
    return (10*p*rec/(9*p+rec)) if 9*p+rec>0 else 0.0

def eval_model_b(df, vocab, idf, method='tfidf', n_samples=200):
    sample = df.sample(min(n_samples, len(df)), random_state=42)
    bleus,rouges,meteors,hint_precs = [],[],[],[]
    for _,row in sample.iterrows():
        ans_l = str(row['answer']).strip().upper()
        correct = str(row[ans_l]) if ans_l in 'ABCD' else ''
        refs = [str(row[l]) for l in 'ABCD' if l!=ans_l]
        gen = get_distractors(str(row['article']), correct, vocab, idf, method)
        for g,r in zip(gen,refs):
            bleus.append(bleu(r,g)); rouges.append(rouge_l(r,g)); meteors.append(meteor(r,g))
        hints = get_hints(str(row['article']), str(row['question']), correct)
        gold_words = set(ctok(str(row['article'])[:200]))
        hits = sum(1 for h in hints if len(set(ctok(h))&gold_words)/(len(gold_words)+1e-8)>0.1)
        hint_precs.append(hits/3)

    res = {"Method": method, "BLEU": round(np.mean(bleus),4),
           "ROUGE-L": round(np.mean(rouges),4), "METEOR": round(np.mean(meteors),4),
           "Hint Prec@3": round(np.mean(hint_precs),4)}
    print(f"\nModel B [{method.upper()}] on {len(sample)} samples:")
    for k,v in res.items(): print(f"  {k:<15}: {v}")
    return res

# CELL 8: MAIN — Run everything
if __name__ == '__main__':
    print("Step 1: Download data")
    # download_race()   # uncomment in Colab

    print("\nStep 2: Load CSVs")
    train_df = pd.read_csv('data/raw/train.csv')
    val_df   = pd.read_csv('data/raw/dev.csv')
    test_df  = pd.read_csv('data/raw/test.csv')

    print("\nStep 3: Expand to option level")
    train_opt = expand_option_level(train_df)
    val_opt   = expand_option_level(val_df)
    test_opt  = expand_option_level(test_df)

    print("\nStep 4: Build TF-IDF features")
    X_tr, X_va, X_te, tfidf_ctx, tfidf_opt = build_features(train_opt, val_opt, test_opt)

    print("\nStep 5: Encode labels")
    y_tr, y_va, y_te, le = encode_labels(train_opt, val_opt, test_opt)

    print("\nStep 6: Train Model A")
    results_a = train_model_a(X_tr, y_tr, X_va, y_va, le)

    print("\nStep 7: Build corpus vocab for Model B")
    docs = (train_df['article'] + ' ' + train_df['question']).tolist()[:5000]
    vocab, idf = build_corpus_vocab(docs, max_feat=3000)

    print("\nStep 8: Evaluate Model B")
    all_b_results = []
    for method in ['tfidf', 'ohe', 'frequency']:
        r = eval_model_b(test_df, vocab, idf, method=method, n_samples=300)
        all_b_results.append(r)

    print("\nModel B Summary:")
    print(pd.DataFrame(all_b_results).to_string(index=False))

    # ── Step 9: Unsupervised & Semi-Supervised ──────────────────────────
    print("\nStep 9: Unsupervised & Semi-Supervised (20 marks)")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    # Reduce to dense
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_tr_d = normalize(svd.fit_transform(X_tr))
    X_va_d = normalize(svd.transform(X_va))

    def map_clusters(cids, y, n):
        mapping = {c: np.bincount(y[cids==c]).argmax()
                   for c in range(n) if (cids==c).any()}
        return np.array([mapping.get(c, 0) for c in cids])

    unsup_results = {}
    # K-Means
    km = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_tr_d)
    kp = map_clusters(km.predict(X_va_d), y_tr[:len(X_va_d)], 4)
    unsup_results['K-Means'] = {'accuracy': round(accuracy_score(y_va, kp),4),
                                 'macro_f1': round(f1_score(y_va,kp,average='macro',zero_division=0),4)}
    print(f"  K-Means: {unsup_results['K-Means']}")

    # GMM
    gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
    gmm.fit(X_tr_d)
    gp = map_clusters(gmm.predict(X_va_d), y_tr[:len(X_va_d)], 4)
    unsup_results['GMM-EM'] = {'accuracy': round(accuracy_score(y_va, gp),4),
                                'macro_f1': round(f1_score(y_va,gp,average='macro',zero_division=0),4)}
    print(f"  GMM-EM: {unsup_results['GMM-EM']}")

    # Label Spreading (10%)
    n_lab = int(len(y_tr)*0.10)
    y_semi = np.full(len(y_tr), -1, dtype=int)
    y_semi[np.random.RandomState(42).choice(len(y_tr), n_lab, replace=False)] = y_tr[:n_lab]
    ls = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2, max_iter=50, n_jobs=-1)
    ls.fit(X_tr_d, y_semi)
    lp = ls.predict(X_va_d)
    unsup_results['LabelSpreading(10%)'] = {'accuracy': round(accuracy_score(y_va,lp),4),
                                              'macro_f1': round(f1_score(y_va,lp,average='macro',zero_division=0),4)}
    print(f"  LabelSpreading(10%): {unsup_results['LabelSpreading(10%)']}")

    # Self-Training (20%)
    n_lab2 = int(len(y_tr)*0.20)
    y_semi2 = np.full(len(y_tr), -1, dtype=int)
    y_semi2[np.random.RandomState(42).choice(len(y_tr), n_lab2, replace=False)] = y_tr[:n_lab2]
    from sklearn.linear_model import LogisticRegression as LR2
    st = SelfTrainingClassifier(LR2(max_iter=500,class_weight='balanced',random_state=42,n_jobs=-1),
                                 threshold=0.85, max_iter=8)
    st.fit(X_tr, y_semi2)
    sp = st.predict(X_va)
    unsup_results['SelfTraining(20%)'] = {'accuracy': round(accuracy_score(y_va,sp),4),
                                           'macro_f1': round(f1_score(y_va,sp,average='macro',zero_division=0),4)}
    print(f"  SelfTraining(20%): {unsup_results['SelfTraining(20%)']}")

    # ── Step 10: Ensemble ────────────────────────────────────────────────
    print("\nStep 10: Ensemble Methods (5 marks)")
    from sklearn.ensemble import VotingClassifier, StackingClassifier

    # Load trained base models from checkpoints
    ckpt_base = {n: joblib.load(f'models/model_a/checkpoints/{n}.pkl')
                 for n in ['Logistic_Regression','Random_Forest','Linear_SVM','Complement_NB']
                 if os.path.exists(f'models/model_a/checkpoints/{n}.pkl')}

    if ckpt_base:
        estimators = list(ckpt_base.items())
        vh = VotingClassifier(estimators, voting='hard', n_jobs=-1)
        vh.fit(X_tr, y_tr)
        vhp = vh.predict(X_va)
        print(f"  Voting Hard: acc={accuracy_score(y_va,vhp):.4f}  f1={f1_score(y_va,vhp,average='macro',zero_division=0):.4f}")
        joblib.dump(vh, 'models/model_a/checkpoints/Voting_Hard.pkl')

        vs = VotingClassifier(estimators, voting='soft', n_jobs=-1)
        vs.fit(X_tr, y_tr)
        vsp = vs.predict(X_va)
        print(f"  Voting Soft: acc={accuracy_score(y_va,vsp):.4f}  f1={f1_score(y_va,vsp,average='macro',zero_division=0):.4f}")
        joblib.dump(vs, 'models/model_a/checkpoints/Voting_Soft.pkl')
    else:
        print("  No base model checkpoints found — run Model A first.")

    print("\n✅ All done!")
