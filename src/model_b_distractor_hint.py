"""
model_b_distractor_hint.py
==========================
Model B — Distractor & Hint Generator

Implements THREE classical ML approaches for distractor selection:
  1. TF-IDF Cosine Similarity Ranker (primary)
  2. OHE + Cosine Similarity (mandatory classical approach)
  3. Frequency-Based Substitution (bag-of-words)

Hint generation:
  - Extractive: rank sentences by keyword overlap (bag-of-words)
  - ML-scored: Logistic Regression on sentence features

Evaluation:
  - BLEU, ROUGE-L, METEOR scores for distractor quality
  - Precision@K for hint extraction
"""

import re
import os
import json
import math
import joblib
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','shall',
    'should','may','might','must','can','could','to','of','in',
    'on','at','by','for','with','about','against','between',
    'through','during','before','after','above','below','from',
    'up','down','out','off','over','under','again','further',
    'then','once','and','but','or','nor','so','yet','both',
    'either','neither','not','as','if','than','too','very',
    'just','this','that','these','those','it','its','itself',
    'he','she','they','them','their','we','our','you','your',
    'i','me','my','myself','yourself','himself','herself',
    'themselves','ourselves','what','which','who','whom',
    'when','where','why','how','all','each','every','few',
    'more','most','other','some','such','no','only','same',
}


def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]+\b', str(text).lower())


def content_words(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def sentences(text: str) -> List[str]:
    """Split text into sentences."""
    return re.split(r'(?<=[.!?])\s+', str(text).strip())


# ─────────────────────────────────────────────
# A. FEATURE VECTORS
# ─────────────────────────────────────────────

def tfidf_vector(tokens: List[str], idf: Dict[str, float],
                 vocab: Dict[str, int]) -> np.ndarray:
    """Compute a TF-IDF vector for a token list."""
    tf = Counter(tokens)
    n  = max(len(tokens), 1)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok, cnt in tf.items():
        if tok in vocab:
            vec[vocab[tok]] = (cnt / n) * idf.get(tok, 1.0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ohe_vector(tokens: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """One-Hot Encoding vector."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] = 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────
# B. CORPUS IDF  (built once from training data)
# ─────────────────────────────────────────────

class CorpusVocab:
    """
    Holds vocabulary + IDF scores computed from a corpus.
    Supports both TF-IDF and OHE vector extraction.
    """
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf:   Dict[str, float] = {}

    def fit(self, documents: List[str]):
        """documents: list of raw passage+question strings."""
        df_count = Counter()
        N = len(documents)
        for doc in documents:
            unique_toks = set(tokenize(doc))
            df_count.update(unique_toks)
        # top-N by document frequency
        top_words = [w for w, _ in df_count.most_common(self.max_features)]
        self.vocab = {w: i for i, w in enumerate(top_words)}
        self.idf   = {w: math.log((N + 1) / (cnt + 1)) + 1.0
                      for w, cnt in df_count.items()
                      if w in self.vocab}
        return self

    def tfidf(self, text: str) -> np.ndarray:
        return tfidf_vector(tokenize(text), self.idf, self.vocab)

    def ohe(self, text: str) -> np.ndarray:
        return ohe_vector(tokenize(text), self.vocab)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'CorpusVocab':
        return joblib.load(path)


# ─────────────────────────────────────────────
# C. CANDIDATE EXTRACTION FROM PASSAGE
# ─────────────────────────────────────────────

def extract_candidates(passage: str, correct_answer: str,
                       top_k: int = 20) -> List[str]:
    """
    Extract noun/content-word phrases from passage as distractor candidates.
    Uses frequency counting (classical ML, no NLP tools).
    Filters out the correct answer and very short phrases.
    """
    # Multi-word noun phrases heuristic: 2-3 consecutive content words
    words = tokenize(passage)
    candidates = set()

    # Unigrams (content words)
    freq = Counter(w for w in words if w not in STOPWORDS and len(w) > 3)
    for w, _ in freq.most_common(top_k):
        candidates.add(w)

    # Bigrams
    for i in range(len(words) - 1):
        if words[i] not in STOPWORDS and words[i+1] not in STOPWORDS:
            candidates.add(f"{words[i]} {words[i+1]}")

    # Remove the correct answer tokens/words
    ans_tokens = set(tokenize(correct_answer))
    candidates = {c for c in candidates
                  if not any(t in ans_tokens for t in tokenize(c))}
    return list(candidates)[:top_k * 2]


# ─────────────────────────────────────────────
# D. DISTRACTOR SELECTION — three approaches
# ─────────────────────────────────────────────

def select_distractors_tfidf(passage: str, question: str,
                              correct_answer: str,
                              cv: CorpusVocab, n: int = 3) -> List[str]:
    """
    Approach 1 — TF-IDF cosine similarity ranking.
    Rank candidates by similarity to correct answer; diverse top-3.
    """
    candidates = extract_candidates(passage, correct_answer)
    if not candidates:
        return ["N/A", "N/A", "N/A"]

    answer_vec = cv.tfidf(correct_answer)
    scored = []
    for c in candidates:
        sim = cosine(cv.tfidf(c), answer_vec)
        scored.append((sim, c))

    # Sort by similarity (descending) but not identical to answer
    scored.sort(key=lambda x: -x[0])
    distractors = []
    seen = set(tokenize(correct_answer))
    for sim, cand in scored:
        if len(distractors) >= n:
            break
        cand_toks = set(tokenize(cand))
        # Accept if moderately similar but not the answer
        if 0.05 < sim < 0.95 and not cand_toks.issubset(seen):
            distractors.append(cand)
            seen.update(cand_toks)

    # Pad if needed
    while len(distractors) < n:
        distractors.append("(not found)")
    return distractors[:n]


def select_distractors_ohe(passage: str, question: str,
                            correct_answer: str,
                            cv: CorpusVocab, n: int = 3) -> List[str]:
    """
    Approach 2 — OHE + Cosine Similarity (mandatory classical approach).
    """
    candidates = extract_candidates(passage, correct_answer)
    if not candidates:
        return ["N/A"] * n

    answer_vec = cv.ohe(correct_answer)
    scored = [(cosine(cv.ohe(c), answer_vec), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])

    distractors = []
    seen = set(tokenize(correct_answer))
    for sim, cand in scored:
        if len(distractors) >= n:
            break
        cand_toks = set(tokenize(cand))
        if 0.02 < sim < 0.95 and not cand_toks.issubset(seen):
            distractors.append(cand)
            seen.update(cand_toks)

    while len(distractors) < n:
        distractors.append("(not found)")
    return distractors[:n]


def select_distractors_frequency(passage: str, correct_answer: str,
                                  n: int = 3) -> List[str]:
    """
    Approach 3 — Frequency-based substitution.
    High-frequency content words in passage excluding the answer.
    """
    words = [w for w in tokenize(passage)
             if w not in STOPWORDS and len(w) > 3]
    freq = Counter(words)
    ans_tokens = set(tokenize(correct_answer))

    distractors = []
    for word, _ in freq.most_common(50):
        if word not in ans_tokens and word not in distractors:
            distractors.append(word)
        if len(distractors) >= n:
            break

    while len(distractors) < n:
        distractors.append("(not found)")
    return distractors[:n]


# ─────────────────────────────────────────────
# E. HINT GENERATION
# ─────────────────────────────────────────────

def score_sentences_bow(passage: str, question: str) -> List[Tuple[float, str]]:
    """
    Score each sentence in passage by keyword overlap with question.
    Returns sorted list of (score, sentence).
    """
    q_words = set(content_words(question))
    sents   = [s.strip() for s in sentences(passage) if len(s.strip()) > 20]
    scored  = []
    for s in sents:
        s_words = set(content_words(s))
        overlap = len(q_words & s_words)
        score   = overlap / (len(q_words) + 1e-8)
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return scored


def generate_hints(passage: str, question: str,
                   correct_answer: str, n_hints: int = 3) -> List[str]:
    """
    Generate graduated hints using extractive BOW scoring.
    Hint 1 = most general (lowest overlap), Hint 3 = near-explicit.
    """
    scored = score_sentences_bow(passage, question)
    if not scored:
        return ["No hints available."] * n_hints

    # Take top sentences and reverse-order for graduation
    top = scored[:n_hints + 2]
    hints = []
    for _, sent in reversed(top[:n_hints]):
        hints.append(sent)
    # Always make the last hint most specific (contains answer keywords)
    hints.sort(key=lambda h: len(set(content_words(h)) &
                               set(content_words(correct_answer))))
    while len(hints) < n_hints:
        hints.append("Refer to the passage for more context.")
    return hints[:n_hints]


# ─────────────────────────────────────────────
# F. FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────

def generate_distractors_and_hints(
        passage: str, question: str, correct_answer: str,
        cv: CorpusVocab,
        method: str = 'tfidf',
) -> Dict:
    """
    Main entry-point: given passage, question, correct answer → distractors + hints.
    method: 'tfidf' | 'ohe' | 'frequency'
    """
    if method == 'tfidf':
        distractors = select_distractors_tfidf(
            passage, question, correct_answer, cv)
    elif method == 'ohe':
        distractors = select_distractors_ohe(
            passage, question, correct_answer, cv)
    else:
        distractors = select_distractors_frequency(passage, correct_answer)

    hints = generate_hints(passage, question, correct_answer)

    return {
        "correct_answer": correct_answer,
        "distractors": distractors,
        "hints": hints,
        "method": method,
    }


# ─────────────────────────────────────────────
# G. EVALUATION — BLEU, ROUGE, METEOR
# ─────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simple corpus-BLEU approximation (unigram–4gram)."""
    ref_toks = tokenize(reference)
    hyp_toks = tokenize(hypothesis)
    if not hyp_toks or not ref_toks:
        return 0.0
    # Brevity penalty
    bp = math.exp(1 - len(ref_toks) / len(hyp_toks)) \
         if len(hyp_toks) < len(ref_toks) else 1.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_toks, n)
        hyp_ng = _ngrams(hyp_toks, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(cnt, ref_ng[ng]) for ng, cnt in hyp_ng.items())
        precisions.append(clipped / sum(hyp_ng.values()))
    # Geometric mean
    if min(precisions) == 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """ROUGE-L using LCS."""
    def lcs_len(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] \
                            else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    ref_toks = tokenize(reference)
    hyp_toks = tokenize(hypothesis)
    if not ref_toks or not hyp_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_len(ref_toks, hyp_toks)
    prec = lcs / len(hyp_toks) if hyp_toks else 0.0
    rec  = lcs / len(ref_toks)  if ref_toks  else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)}


def meteor_score(reference: str, hypothesis: str) -> float:
    """Simplified METEOR (unigram F-mean with α=0.9, γ=0.5)."""
    ref_toks = set(tokenize(reference))
    hyp_toks = tokenize(hypothesis)
    if not hyp_toks or not ref_toks:
        return 0.0
    matches = sum(1 for t in hyp_toks if t in ref_toks)
    prec = matches / len(hyp_toks)
    rec  = matches / len(ref_toks)
    f    = (10 * prec * rec) / (9 * prec + rec) if (9*prec + rec) > 0 else 0.0
    # Chunk penalty omitted for simplicity (conservative score)
    return round(f, 4)


def evaluate_distractors(generated: List[str],
                          reference: List[str]) -> Dict[str, float]:
    """
    Evaluate generated distractors against reference distractors.
    Returns average BLEU, ROUGE-L F1, METEOR.
    """
    bleus, rouges, meteors = [], [], []
    for gen, ref in zip(generated, reference):
        bleus.append(bleu_score(ref, gen))
        rouges.append(rouge_l(ref, gen)['f1'])
        meteors.append(meteor_score(ref, gen))
    return {
        "BLEU":    round(np.mean(bleus),   4),
        "ROUGE-L": round(np.mean(rouges),  4),
        "METEOR":  round(np.mean(meteors), 4),
    }


def evaluate_hints_precision(generated_hints: List[str],
                              gold_sentence: str, k: int = 3) -> float:
    """Precision@K: fraction of top-K hints overlapping with gold sentence."""
    gold_words = set(content_words(gold_sentence))
    hits = 0
    for h in generated_hints[:k]:
        h_words = set(content_words(h))
        if len(h_words & gold_words) / (len(gold_words) + 1e-8) > 0.2:
            hits += 1
    return round(hits / k, 4)


# ─────────────────────────────────────────────
# H. DISTRACTOR RANKER — Confusion Matrix
#    Frames distractor ranking as binary classification:
#    label=1 if the top-ranked candidate is NOT the correct answer
#    (good distractor), label=0 if it IS the correct answer (failure).
# ─────────────────────────────────────────────

def build_distractor_ranker_labels(df, cv: CorpusVocab,
                                    method: str = 'tfidf',
                                    n_samples: int = 500) -> Dict:
    """
    For each sample:
      - Generate top distractor
      - Check if it equals (or contains tokens of) correct answer
      - y_true=1 (correct distractor), y_pred=1 (model produced a distractor)
    Also builds a confusion matrix across 3 distractor slots.
    Returns precision, recall, F1, accuracy, confusion matrix arrays.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        confusion_matrix, classification_report,
    )

    sample = df.sample(min(n_samples, len(df)), random_state=42)
    y_true_all, y_pred_all = [], []

    for _, row in sample.iterrows():
        ans_l   = str(row['answer']).strip().upper()
        correct = str(row[ans_l]) if ans_l in 'ABCD' else ''
        refs    = [str(row[l]) for l in 'ABCD' if l != ans_l]

        gen = (select_distractors_tfidf(str(row['article']),
                                         str(row['question']), correct, cv)
               if method == 'tfidf'
               else select_distractors_ohe(str(row['article']),
                                            str(row['question']), correct, cv)
               if method == 'ohe'
               else select_distractors_frequency(str(row['article']), correct))

        ans_toks = set(tokenize(correct))
        for g, r in zip(gen, refs):
            # y_true: is reference a valid distractor (not the answer)?
            ref_toks = set(tokenize(r))
            y_true_all.append(1 if not ref_toks.issubset(ans_toks) else 0)
            # y_pred: did we generate something different from the answer?
            gen_toks = set(tokenize(g))
            y_pred_all.append(1 if (g not in ('N/A','(not found)') and
                                     not gen_toks.issubset(ans_toks)) else 0)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  Distractor Ranker Evaluation [{method.upper()}]")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Answer','Distractor'])}")

    # Plot confusion matrix
    os.makedirs('../models/model_b', exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Predicted Answer', 'Predicted Distractor'],
                yticklabels=['True Answer', 'True Distractor'])
    plt.title(f'Distractor Ranker CM — {method.upper()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../models/model_b/distractor_cm_{method}.png', dpi=150)
    plt.close()
    print(f"  Confusion matrix saved.")

    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4),
            "confusion_matrix": cm.tolist()}


# ─────────────────────────────────────────────
# MAIN — quick smoke test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Model B — Distractor & Hint Generator ===\n")

    sample_passage = (
        "The Amazon rainforest is the world's largest tropical rainforest, "
        "covering much of northwestern Brazil. It represents over half of "
        "the planet's remaining rainforests and comprises the largest and "
        "most biodiverse tract of tropical rainforest in the world. "
        "The Amazon basin is approximately 7,000,000 km² and contains "
        "the Amazon River and its tributaries."
    )
    sample_question = "Where is the Amazon rainforest mainly located?"
    correct_answer  = "Brazil"

    cv = CorpusVocab(max_features=200).fit([sample_passage])

    for method in ['tfidf', 'ohe', 'frequency']:
        result = generate_distractors_and_hints(
            sample_passage, sample_question, correct_answer, cv, method)
        print(f"[{method.upper()}]")
        print(f"  Distractors: {result['distractors']}")
        for i, h in enumerate(result['hints'], 1):
            print(f"    Hint {i}: {h[:80]}…" if len(h) > 80 else f"    Hint {i}: {h}")
        print()

    ref_distractors = ["Argentina", "Peru", "Colombia"]
    scores = evaluate_distractors(result['distractors'], ref_distractors)
    print(f"NLG scores: {scores}")
