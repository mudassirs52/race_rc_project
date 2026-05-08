"""
app.py  —  RACE Reading Comprehension UI
=========================================
4 screens:
  1. Article Input
  2. Question & Answer Quiz View
  3. Hint Panel
  4. Developer / Analytics Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, time, random, re
from datetime import datetime
from scipy.sparse import load_npz, hstack, issparse
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="RACE Reading Comprehension",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.main-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: rgba(100,181,246,0.15);
    border-left: 4px solid #64B5F6;
    border-radius: 8px; padding: 14px; margin: 6px 0;
}
.correct-badge { background:#1B5E20; color:#A5D6A7; padding:6px 14px;
    border-radius:20px; font-weight:700; }
.wrong-badge   { background:#B71C1C; color:#FFCDD2; padding:6px 14px;
    border-radius:20px; font-weight:700; }
.hint-box {
    background: rgba(255,193,7,0.1); border-left: 4px solid #FFC107;
    border-radius:8px; padding:12px; margin:8px 0; color:#FFF8E1;
}
h1,h2,h3 { color: #E3F2FD !important; }
.stButton>button {
    background: linear-gradient(90deg,#667eea,#764ba2);
    color:white; border:none; border-radius:10px;
    padding:10px 24px; font-weight:600; transition:0.3s;
}
.stButton>button:hover { opacity:0.85; transform:translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HELPERS / CONSTANTS
# ─────────────────────────────────────────────────────────
DATA_DIR  = "data/processed"
MODEL_DIR = "models/model_a/checkpoints"
RAW_DIR   = "data/raw"
RESULTS_PATH = "models/model_a/test_results.json"

STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','have','has',
    'had','do','does','will','would','can','could','to','of','in',
    'on','at','by','for','with','this','that','and','but','or','not',
    'it','its','as','if','from','we','you','he','she','they','what',
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    return re.findall(r'\b[a-z]+\b', clean(text))

def content_words(text):
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w)>2]

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text)) if len(s.strip())>15]

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "passage": "", "question": "", "options": {},
        "correct_letter": "", "user_answer": "",
        "checked": False, "hints": [], "hint_idx": 0,
        "distractors": [], "session_log": [],
        "latencies": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────
# LOAD ASSETS (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            name = fname[:-4].replace("_", " ").title()
            models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
    return models

@st.cache_resource(show_spinner=False)
def load_vectorizers():
    ctx_path = os.path.join(DATA_DIR, "tfidf_ctx.pkl")
    opt_path = os.path.join(DATA_DIR, "tfidf_opt.pkl")
    le_path  = os.path.join(DATA_DIR, "label_encoder.pkl")
    if not all(os.path.exists(p) for p in [ctx_path, opt_path, le_path]):
        return None, None, None
    return (joblib.load(ctx_path), joblib.load(opt_path),
            joblib.load(le_path))

@st.cache_data(show_spinner=False)
def load_race_samples():
    path = os.path.join(RAW_DIR, "test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────
def predict_answer(passage, question, options_dict,
                   tfidf_ctx, tfidf_opt, le, models):
    """Run all trained Model A classifiers; return majority vote."""
    if not models:
        return "N/A", {}, 0.0

    ctx_text = clean(passage) + " " + clean(question)
    votes = {l: 0 for l in "ABCD"}
    per_model = {}
    t0 = time.time()

    for name, model in models.items():
        preds = []
        for letter in "ABCD":
            opt_text = clean(options_dict.get(letter, ""))
            X_ctx = tfidf_ctx.transform([ctx_text])
            X_opt = tfidf_opt.transform([opt_text])
            X = hstack([X_ctx, X_opt])
            pred = model.predict(X)[0]
            preds.append((letter, pred))
        # Pick the letter predicted most often (or use proba if available)
        try:
            proba_rows = []
            for letter in "ABCD":
                opt_text = clean(options_dict.get(letter, ""))
                X_ctx = tfidf_ctx.transform([ctx_text])
                X_opt = tfidf_opt.transform([opt_text])
                X = hstack([X_ctx, X_opt])
                p = model.predict_proba(X)[0]
                proba_rows.append(p)
            # Sum probability of each letter class
            letter_scores = {}
            for i, letter in enumerate("ABCD"):
                idx = list(le.classes_).index(letter) if letter in le.classes_ else i
                letter_scores[letter] = proba_rows[i][idx]
            best = max(letter_scores, key=letter_scores.get)
        except Exception:
            best = preds[0][1] if preds else "A"
            # decode
            try:
                best = le.inverse_transform([int(best)])[0]
            except Exception:
                pass

        per_model[name] = str(best)
        if best in votes:
            votes[best] += 1

    latency = time.time() - t0
    best_vote = max(votes, key=votes.get)
    return best_vote, per_model, latency


def generate_distractors(passage, question, correct_answer, n=3):
    """Frequency-based distractor extraction (no external deps needed)."""
    words = [w for w in tokenize(passage)
             if w not in STOPWORDS and len(w)>3]
    from collections import Counter
    freq = Counter(words)
    ans_toks = set(tokenize(correct_answer))
    distractors = []
    for word, _ in freq.most_common(60):
        if word not in ans_toks and word not in distractors:
            distractors.append(word)
        if len(distractors) >= n:
            break
    while len(distractors) < n:
        distractors.append("N/A")
    return distractors[:n]


def generate_hints(passage, question, correct_answer, n=3):
    """BOW extractive hint generation."""
    q_words = set(content_words(question))
    sents = split_sentences(passage)
    scored = []
    for s in sents:
        overlap = len(q_words & set(content_words(s)))
        scored.append((overlap / (len(q_words)+1e-8), s))
    scored.sort(key=lambda x: -x[0])
    hints = [s for _, s in scored[:n+2]]
    hints.sort(key=lambda h: len(set(content_words(h)) &
                               set(content_words(correct_answer))))
    while len(hints) < n:
        hints.append("Re-read the passage carefully for clues.")
    return hints[:n]


# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 RACE-RC System")
    st.markdown("**AL2002 — AI Lab Project**")
    st.divider()
    screen = st.radio("Navigate", [
        "📝 Article Input",
        "❓ Quiz View",
        "💡 Hint Panel",
        "📊 Analytics Dashboard",
    ])
    st.divider()
    st.caption("Traditional ML · TF-IDF · OHE")

# Load assets
with st.spinner("Loading models…"):
    models      = load_models()
    tfidf_ctx, tfidf_opt, le = load_vectorizers()
    race_df     = load_race_samples()

models_ready = bool(models) and tfidf_ctx is not None

# ═════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ═════════════════════════════════════════════════════════
if screen == "📝 Article Input":
    st.title("📝 Article Input")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎲 Load Random RACE Sample") and race_df is not None:
            row = race_df.sample(1).iloc[0]
            st.session_state.passage         = str(row['article'])
            st.session_state.question        = str(row['question'])
            st.session_state.options         = {l: str(row[l]) for l in 'ABCD'}
            st.session_state.correct_letter  = str(row['answer']).strip().upper()
            st.session_state.checked         = False
            st.session_state.user_answer     = ""
            st.session_state.hint_idx        = 0
            st.rerun()

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    passage = st.text_area("📄 Paste or type a reading passage:",
                            value=st.session_state.passage,
                            height=200, key="passage_input")
    question = st.text_input("❓ Question:", value=st.session_state.question)

    col_a, col_b = st.columns(2)
    with col_a:
        opt_a = st.text_input("Option A:", value=st.session_state.options.get('A',''))
        opt_c = st.text_input("Option C:", value=st.session_state.options.get('C',''))
    with col_b:
        opt_b = st.text_input("Option B:", value=st.session_state.options.get('B',''))
        opt_d = st.text_input("Option D:", value=st.session_state.options.get('D',''))

    correct = st.selectbox("✅ Correct Answer (for training/eval):",
                           ['A','B','C','D'],
                           index='ABCD'.index(st.session_state.correct_letter)
                           if st.session_state.correct_letter in 'ABCD' else 0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Submit & Run Inference"):
        if not passage.strip():
            st.error("Please enter a passage.")
        else:
            st.session_state.passage        = passage
            st.session_state.question       = question
            st.session_state.options        = {'A':opt_a,'B':opt_b,'C':opt_c,'D':opt_d}
            st.session_state.correct_letter = correct
            st.session_state.checked        = False
            st.session_state.user_answer    = ""
            st.session_state.hint_idx       = 0

            with st.spinner("Running Model A & Model B …"):
                t0 = time.time()
                # Model B — distractors & hints
                st.session_state.distractors = generate_distractors(
                    passage, question, st.session_state.options.get(correct,''))
                st.session_state.hints = generate_hints(
                    passage, question, st.session_state.options.get(correct,''))
                latency = time.time() - t0
                st.session_state.latencies.append(latency)

            st.success(f"✅ Inference complete ({latency:.2f}s). Go to **Quiz View**.")


# ═════════════════════════════════════════════════════════
# SCREEN 2 — QUIZ VIEW
# ═════════════════════════════════════════════════════════
elif screen == "❓ Quiz View":
    st.title("❓ Quiz View")

    if not st.session_state.passage:
        st.warning("No passage loaded. Go to **Article Input** first.")
        st.stop()

    with st.expander("📖 Reading Passage", expanded=True):
        st.write(st.session_state.passage[:1000] +
                 ("…" if len(st.session_state.passage)>1000 else ""))

    st.markdown(f"### {st.session_state.question}")

    opts = st.session_state.options
    # Shuffle options including distractors if original are empty
    display_opts = {}
    for l in 'ABCD':
        display_opts[l] = opts.get(l, '')

    choice = st.radio("Select your answer:",
                      [f"**{l}** — {display_opts[l]}" for l in 'ABCD'],
                      index=None)

    col1, col2 = st.columns([1, 4])
    with col1:
        check = st.button("✔️ Check Answer")

    if check and choice:
        selected = choice.split('**')[1]
        st.session_state.user_answer = selected
        st.session_state.checked = True

        correct_l = st.session_state.correct_letter
        is_correct = (selected == correct_l)

        # Log
        st.session_state.session_log.append({
            "timestamp": datetime.now().isoformat(),
            "question": st.session_state.question[:60],
            "user_answer": selected,
            "correct_answer": correct_l,
            "correct": is_correct,
        })

        if is_correct:
            st.markdown('<span class="correct-badge">✅ CORRECT!</span>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<span class="wrong-badge">❌ Wrong — Correct: {correct_l}</span>',
                        unsafe_allow_html=True)
            st.info("💡 Try the **Hint Panel** for clues!")

        # Model A prediction
        if models_ready:
            with st.spinner("Running Model A verifier…"):
                pred, per_model, lat = predict_answer(
                    st.session_state.passage, st.session_state.question,
                    opts, tfidf_ctx, tfidf_opt, le, models)
            st.markdown(f"**Model A Prediction:** `{pred}` (latency: {lat:.3f}s)")
            with st.expander("Per-model breakdown"):
                for name, p in per_model.items():
                    st.write(f"- **{name}**: `{p}`")
        else:
            st.info("Models not loaded — run preprocessing and training first.")


# ═════════════════════════════════════════════════════════
# SCREEN 3 — HINT PANEL
# ═════════════════════════════════════════════════════════
elif screen == "💡 Hint Panel":
    st.title("💡 Hint Panel")

    if not st.session_state.passage:
        st.warning("No passage loaded. Go to **Article Input** first.")
        st.stop()

    hints = st.session_state.hints
    if not hints:
        hints = generate_hints(
            st.session_state.passage,
            st.session_state.question,
            st.session_state.options.get(st.session_state.correct_letter,''))
        st.session_state.hints = hints

    st.markdown("Hints are revealed **gradually** — from general to specific.")
    st.progress(st.session_state.hint_idx / max(len(hints), 1))

    for i in range(st.session_state.hint_idx):
        label = ["🟡 General Hint", "🟠 Specific Hint", "🔴 Near-Explicit Hint"][i % 3]
        st.markdown(f'<div class="hint-box"><b>{label} {i+1}:</b><br>{hints[i]}</div>',
                    unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.hint_idx < len(hints):
            if st.button("💡 Reveal Next Hint"):
                st.session_state.hint_idx += 1
                st.rerun()
    with col2:
        if st.session_state.hint_idx >= len(hints):
            if st.button("🔓 Reveal Answer"):
                ans_l = st.session_state.correct_letter
                ans_text = st.session_state.options.get(ans_l,'')
                st.success(f"**Answer: {ans_l}** — {ans_text}")

    # Generated distractors
    st.divider()
    st.subheader("🎭 Generated Distractors (Model B)")
    distractors = st.session_state.distractors
    if distractors:
        for i, d in enumerate(distractors, 1):
            st.markdown(f"- **Distractor {i}**: `{d}`")
    else:
        st.info("Run inference from Article Input to see distractors.")


# ═════════════════════════════════════════════════════════
# SCREEN 4 — ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════
elif screen == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    # ── Session stats ──
    log = st.session_state.session_log
    st.subheader("Session Performance")
    c1, c2, c3, c4 = st.columns(4)
    total   = len(log)
    correct = sum(1 for r in log if r['correct'])
    acc     = correct/total if total else 0
    avg_lat = np.mean(st.session_state.latencies) if st.session_state.latencies else 0

    c1.metric("Questions Answered", total)
    c2.metric("Correct", correct)
    c3.metric("Session Accuracy", f"{acc:.1%}")
    c4.metric("Avg Latency", f"{avg_lat:.3f}s")

    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode()
        st.download_button("⬇ Export Session CSV", csv,
                           "session_results.csv", "text/csv")

    st.divider()

    # ── Model A saved results ──
    st.subheader("Model A — Test Set Results vs BERT/T5 Baselines")
    results = load_results()
    if results:
        trad = results.get("traditional_models", {})
        base = results.get("bert_baselines", {})

        all_names, all_accs, all_f1s, colors = [], [], [], []
        for n, m in trad.items():
            all_names.append("[Ours] " + n)
            all_accs.append(m.get("accuracy", 0))
            all_f1s.append(m.get("macro_f1", 0))
            colors.append("#42A5F5")
        for n, m in base.items():
            all_names.append("[Base] " + n)
            all_accs.append(m.get("accuracy", 0))
            all_f1s.append(m.get("macro_f1", 0))
            colors.append("#EF5350")

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=all_names, y=all_accs,
                             marker_color=colors, opacity=0.85))
        fig.add_trace(go.Bar(name='Macro F1', x=all_names, y=all_f1s,
                             marker_color=colors, opacity=0.55))
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font_color='white',
            title="Accuracy & Macro F1 — Traditional ML vs BERT/T5",
            yaxis=dict(range=[0,1]), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        rows = []
        for n, m in {**trad, **base}.items():
            rows.append({"Model": n,
                         "Accuracy": m.get("accuracy","—"),
                         "Macro F1": m.get("macro_f1","—"),
                         "Exact Match": m.get("exact_match","—")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    else:
        st.info("No saved results found. Run `src/evaluate.py` after training.")

    # ── Confusion matrix images ──
    st.divider()
    st.subheader("Confusion Matrices")
    cm_dir = "models/model_a/confusion_matrices"
    if os.path.isdir(cm_dir):
        imgs = [f for f in os.listdir(cm_dir) if f.endswith('.png')]
        if imgs:
            cols = st.columns(min(len(imgs), 2))
            for i, img in enumerate(imgs):
                cols[i % 2].image(os.path.join(cm_dir, img),
                                   caption=img.replace('_', ' ').replace('.png',''),
                                   use_container_width=True)
    else:
        st.info("Confusion matrices not yet generated.")

    # ── Comparison bar ──
    st.divider()
    st.subheader("Accuracy vs Random Chance Baseline")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc * 100,
        title={"text": "Session Accuracy (%)"},
        gauge={
            "axis": {"range": [0,100]},
            "steps": [{"range":[0,25],"color":"red"},
                      {"range":[25,65],"color":"orange"},
                      {"range":[65,100],"color":"green"}],
            "threshold": {"line":{"color":"white","width":3},
                          "thickness":0.75, "value":25},
        }
    ))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
    st.plotly_chart(fig2, use_container_width=True)
