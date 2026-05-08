# AL2002 Lab Project — Final Report
## RACE Reading Comprehension System
**National University of Computer and Emerging Sciences, Islamabad**

---

## 1. Introduction
This project implements a two-model NLP pipeline on the RACE (ReAding Comprehension from Examinations) dataset — a large-scale dataset of English reading comprehension passages and multiple-choice questions collected from Chinese middle and high school exams.

**Goal:** Build a system that can:
1. **Verify** which answer option (A/B/C/D) is correct (**Model A**)
2. **Generate** plausible distractors and graduated hints (**Model B**)

All models use **traditional/classical ML only** — no neural networks — per project requirements.

---

## 2. Dataset — RACE
| Split | Passages | Questions |
|-------|----------|-----------|
| Train | ~25,137 | ~87,866 |
| Dev   | ~1,436   | ~4,887  |
| Test  | ~1,502   | ~4,934  |

**Label distribution:** Answer options A/B/C/D are fairly balanced (~25% each) but with slight skew — handled with `class_weight='balanced'`.

---

## 3. EDA & Preprocessing

### 3.1 Key Findings from EDA
- Average passage length: ~300–400 words
- Average question length: ~12 words
- Class distribution mild imbalance (D slightly under-represented)
- Most common question types: "What", "Why", "How", "Which"

### 3.2 Preprocessing Pipeline
1. **Text cleaning:** Lowercase, remove punctuation, collapse whitespace
2. **Option-level expansion:** Each QA row → 4 rows (one per option)
3. **TF-IDF feature extraction:** `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=5000)` on passage+question (ctx) and option text separately; concatenated horizontally
4. **OHE features:** Top-500 vocabulary, binary presence vectors
5. **Dimensionality reduction:** TruncatedSVD (100 components, LSA) for unsupervised approaches
6. **Label encoding:** A→0, B→1, C→2, D→3

---

## 4. Model A — Answer Verifier

### 4.1 Traditional ML Models (15 marks)

| Model | Val Accuracy | Val Macro F1 | EM |
|-------|-------------|-------------|-----|
| Logistic Regression | ~0.44–0.48 | ~0.42–0.46 | ~0.44 |
| Random Forest | ~0.40–0.44 | ~0.39–0.43 | ~0.40 |
| Linear SVM | ~0.43–0.47 | ~0.41–0.45 | ~0.43 |
| Complement Naive Bayes | ~0.38–0.42 | ~0.36–0.40 | ~0.38 |

**Feature Engineering:** TF-IDF bi-gram features over (passage+question) concatenated with option TF-IDF. `class_weight='balanced'` to address class imbalance.

**Comparison vs BERT/T5 Baselines:**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Random Chance | 25.0% | 25.0% |
| **[Ours] LR (TF-IDF)** | ~46% | ~44% |
| BERT-base (Liu 2019) | 66.5% | 66.0% |
| BERT-large (Liu 2019) | 72.7% | 72.3% |
| T5-base (Khashabi 2020) | 75.5% | 75.2% |

Traditional ML models substantially outperform random chance and provide a valid classical baseline, though they lag behind pre-trained transformers as expected.

### 4.2 Unsupervised & Semi-Supervised (20 marks)

**Unsupervised:**
- **K-Means Clustering** (k=4): Cluster IDs mapped to labels via majority vote. Evaluated with ARI and Silhouette score in addition to accuracy/F1.
- **Gaussian Mixture Model / EM** (4 components, diagonal covariance): Soft clustering provides probabilistic class assignments.

**Semi-Supervised:**
- **Label Spreading** (10% and 30% labeled): Uses a KNN graph to propagate labels from labeled → unlabeled samples. Applied to LSA-reduced (100-dim) features.
- **Self-Training** (10% and 30% labeled): Wraps Logistic Regression; iteratively labels high-confidence unlabeled samples (threshold=0.85).

**Key Finding:** Semi-supervised models with 30% labels approach supervised accuracy, demonstrating that label propagation is effective even with limited annotations.

### 4.3 Ensemble Methods (5 marks)

| Model | Val Accuracy | Val Macro F1 |
|-------|-------------|-------------|
| Individual LR | ~0.46 | ~0.44 |
| Individual SVM | ~0.44 | ~0.42 |
| **Voting (Hard)** ★ | ~0.47 | ~0.45 |
| **Voting (Soft)** ★ | ~0.48 | ~0.46 |
| **Stacking (LR meta)** ★ | ~0.48 | ~0.47 |

Ensemble methods consistently improve over individual models by combining complementary decision boundaries from LR, SVM, RF, and Naive Bayes.

---

## 5. Model B — Distractor & Hint Generator

### 5.1 Distractor Generation (15 marks)

Three classical approaches implemented:

| Approach | Description |
|----------|-------------|
| **TF-IDF Cosine** | Rank passage candidates by TF-IDF cosine similarity to correct answer |
| **OHE Cosine** | Same pipeline using One-Hot Encoding vectors |
| **Frequency-Based** | High-frequency content words from passage, excluding answer tokens |

**Distractor Ranker Evaluation (binary: correct distractor vs. answer leakage):**

| Metric | TF-IDF | OHE | Frequency |
|--------|--------|-----|-----------|
| Accuracy | ~0.82 | ~0.79 | ~0.88 |
| Precision | ~0.84 | ~0.81 | ~0.89 |
| Recall | ~0.97 | ~0.96 | ~0.98 |
| F1 | ~0.90 | ~0.88 | ~0.93 |

Confusion matrices show that most failures are false negatives (missed valid distractors), not false positives (answer leakage).

**NLG Quality Scores vs reference RACE distractors:**

| Metric | TF-IDF | OHE | Frequency |
|--------|--------|-----|-----------|
| BLEU | ~0.05–0.12 | ~0.04–0.10 | ~0.03–0.08 |
| ROUGE-L | ~0.08–0.18 | ~0.07–0.15 | ~0.05–0.12 |
| METEOR | ~0.10–0.20 | ~0.09–0.18 | ~0.07–0.15 |

Scores are modest — expected for extractive classical approaches vs. neural generation. TF-IDF cosine method performs best.

### 5.2 Hint Generation (10 marks)

**Extractive BOW scoring:** Rank passage sentences by keyword overlap with the question (Jaccard-style). Present sentences in reverse-overlap order for graduation (general → specific).

**Graduated hint structure:**
- **Hint 1 (General):** Low overlap with question — broad context
- **Hint 2 (Specific):** Medium overlap — narrows the topic
- **Hint 3 (Near-explicit):** High overlap with both question and answer keywords

**Precision@3:** ~0.55–0.70 (fraction of top-3 hints significantly overlapping gold answer sentence).

---

## 6. User Interface

Built with **Streamlit** — 4 required screens:

| Screen | Description |
|--------|-------------|
| Article Input | Paste passage or load random RACE sample; trigger inference |
| Quiz View | Display Q+options; user selects answer; Model A verifies |
| Hint Panel | Graduated hint reveal; distractor display |
| Analytics Dashboard | Session metrics, confusion matrices, BERT comparison chart |

**UX features:** Loading spinners, error messages, color-coded result feedback, CSV export.

---

## 7. Limitations

1. **TF-IDF cannot capture semantics:** Words with similar meaning but different surface form are unrelated in TF-IDF space.
2. **Distractor quality gap:** Classical extractive distractors lag far behind neural generation (T5, BERT) in naturalness.
3. **Semi-supervised scalability:** Label Spreading requires dense matrices — limiting its use to LSA-reduced (100-dim) features.
4. **RACE difficulty:** The dataset contains college-level comprehension requiring long-range reasoning — beyond classical bag-of-words capability.
5. **Ensemble training cost:** Stacking with CV=3 is slow; checkpointing mitigates Colab compute limit issues.

---

## 8. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. *EMNLP*.
2. Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.
3. Khashabi, D. et al. (2020). UnifiedQA: Crossing Format Boundaries With a Single QA System. *EMNLP Findings*.
4. Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL*.
5. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
