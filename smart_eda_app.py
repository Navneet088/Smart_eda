import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import json, re, textwrap

# LLM provider imports (optional — loaded on demand)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic as anthropic_sdk
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder,
    OrdinalEncoder, BinaryEncoder
)
try:
    import category_encoders as ce
    HAS_CE = True
except ImportError:
    HAS_CE = False
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)

# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartEDA Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #0d0f1a; }

.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 100%);
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #38bdf8 !important;
    letter-spacing: -0.02em;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

.metric-card {
    background: linear-gradient(145deg, #1e293b, #162032);
    border: 1px solid #1e40af33;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px #0003;
}

.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-card .lbl {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #818cf8;
    border-left: 3px solid #818cf8;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.tag {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}

.tag-cat {
    background: #2d1b4e;
    color: #c4b5fd;
}

.tag-warn {
    background: #422006;
    color: #fbbf24;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px #1d4ed844 !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px #1d4ed866 !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b) !important;
    border-right: 1px solid #1e40af33 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
}

.info-box {
    background: #0f2744;
    border: 1px solid #1e40af55;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #bae6fd;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.success-box {
    background: #052e16;
    border: 1px solid #15803d55;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #86efac;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.warn-box {
    background: #1c1407;
    border: 1px solid #b4530055;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #fcd34d;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* ── WORKFLOW PIPELINE ── */
.wf-wrap {
    display: flex;
    align-items: stretch;
    gap: 0;
    width: 100%;
    overflow-x: auto;
    padding: 1.4rem 0 1rem;
    scrollbar-width: thin;
    scrollbar-color: #1e40af #0d0f1a;
}

.wf-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 110px;
    flex: 1;
    position: relative;
    animation: wf-fadein 0.5s ease both;
}

@keyframes wf-fadein {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}

.wf-circle {
    width: 58px; height: 58px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    position: relative;
    z-index: 2;
    border: 2px solid transparent;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.wf-node.done .wf-circle {
    background: linear-gradient(135deg, #0f4c2a, #166534);
    border-color: #22c55e;
    box-shadow: 0 0 18px #22c55e55;
    animation: wf-pulse-done 2.5s ease infinite;
}

.wf-node.active .wf-circle {
    background: linear-gradient(135deg, #1e3a8a, #3730a3);
    border-color: #60a5fa;
    box-shadow: 0 0 22px #3b82f677;
    animation: wf-pulse-active 1.4s ease infinite;
}

.wf-node.pending .wf-circle {
    background: #1e293b;
    border-color: #334155;
    opacity: 0.55;
}

@keyframes wf-pulse-done {
    0%,100% { box-shadow: 0 0 16px #22c55e44; }
    50%      { box-shadow: 0 0 28px #22c55e88; }
}

@keyframes wf-pulse-active {
    0%,100% { box-shadow: 0 0 18px #3b82f655; }
    50%      { box-shadow: 0 0 34px #3b82f699; transform: scale(1.06); }
}

.wf-label {
    margin-top: 0.55rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-align: center;
    line-height: 1.3;
    max-width: 100px;
}

.wf-node.done  .wf-label { color: #4ade80; }
.wf-node.active .wf-label { color: #93c5fd; }
.wf-node.pending .wf-label { color: #475569; }

.wf-badge {
    font-size: 0.6rem;
    margin-top: 0.25rem;
    padding: 1px 8px;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
}
.wf-node.done  .wf-badge { background:#14532d; color:#86efac; }
.wf-node.active .wf-badge { background:#1e3a8a; color:#bfdbfe; }
.wf-node.pending .wf-badge { background:#1e293b; color:#475569; }

.wf-arrow {
    display: flex; align-items: center;
    padding: 0 2px;
    margin-top: -16px;   /* align with circle center */
}

.wf-arrow svg { display: block; }

.wf-node.done   ~ .wf-arrow .arrow-line { stroke: #22c55e; }
.wf-node.active ~ .wf-arrow .arrow-line { stroke: #3b82f6; }
.wf-node.pending~ .wf-arrow .arrow-line { stroke: #334155; }

/* connector line between nodes */
.wf-connector {
    flex: 1;
    height: 2px;
    margin-top: 28px;    /* half of circle height */
    min-width: 18px;
    position: relative;
    z-index: 1;
}
.wf-connector.done   { background: linear-gradient(90deg,#22c55e,#16a34a); box-shadow:0 0 6px #22c55e55; }
.wf-connector.active { background: linear-gradient(90deg,#3b82f6,#6366f1);
    animation: wf-flow 1.2s linear infinite;
    background-size: 200% 100%;
}
.wf-connector.pending { background: #1e293b; }

@keyframes wf-flow {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

/* step detail strip */
.wf-detail-strip {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 1.2rem;
    font-size: 0.8rem;
    color: #64748b;
}
.wf-chip {
    background: #1e293b;
    color: #94a3b8;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    border: 1px solid #334155;
}
.wf-chip.done  { background:#14532d; color:#86efac; border-color:#166534; }
.wf-chip.active{ background:#1e3a8a; color:#bfdbfe; border-color:#1d4ed8;
    animation: wf-chip-blink 1.5s ease infinite; }
@keyframes wf-chip-blink {
    0%,100%{ opacity:1; } 50%{ opacity:0.65; }
}

/* sidebar mini pipeline */
.sb-pipeline {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0.5rem 0;
}
.sb-step {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
    padding: 5px 8px;
    border-radius: 8px;
    border: 1px solid transparent;
    transition: all 0.3s;
}
.sb-step.done   { background:#052e16; border-color:#166534; color:#4ade80; }
.sb-step.active { background:#1e3a8a22; border-color:#3b82f6; color:#93c5fd;
    animation: wf-chip-blink 1.5s ease infinite; }
.sb-step.pending{ color:#475569; }
.sb-dot {
    width:8px; height:8px; border-radius:50%; flex-shrink:0;
}
.sb-step.done   .sb-dot { background:#22c55e; box-shadow:0 0 6px #22c55e; }
.sb-step.active .sb-dot { background:#3b82f6; box-shadow:0 0 6px #3b82f6; }
.sb-step.pending .sb-dot { background:#334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key in ["df", "df_processed", "target_col", "task_type", "preprocessing_log",
            "auto_analysed", "auto_findings", "auto_stats", "auto_Xauto", "auto_yauto",
            "sug_models", "sug_scaler", "sug_features", "auto_results", "best_auto_model",
            "auto_Xtest", "auto_ytest",
            "llm_provider", "llm_api_key", "llm_chat_history", "llm_model_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "preprocessing_log" not in st.session_state or st.session_state.preprocessing_log is None:
    st.session_state.preprocessing_log = []
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

# ─────────────────────────────────────────────
# WORKFLOW HELPER
# ─────────────────────────────────────────────
def get_pipeline_status():
    df       = st.session_state.df
    dfp      = st.session_state.df_processed
    target   = st.session_state.target_col
    log      = st.session_state.preprocessing_log or []
    has_data     = df is not None
    has_target   = target is not None and has_data
    has_miss     = has_data and any("[Missing]" in s or "Auto-fill" in s for s in log)
    has_preproc  = has_data and any(s for s in log if "[Missing]" not in s and "Auto-fill" not in s and "Removed duplicate" not in s and "Feature selection" not in s)
    has_featsel  = has_data and any("Feature selection" in s for s in log)
    models_done  = st.session_state.models_trained
    if has_data:
        nulls_remaining = dfp.isnull().sum().sum() if dfp is not None else 0
        miss_done = has_miss or nulls_remaining == 0
    else:
        miss_done = False
    steps = [
        {"id":"upload",  "icon":"📂","label":"Data\nUpload",      "done": has_data,    "active": not has_data},
        {"id":"overview","icon":"📋","label":"Overview\n& EDA",    "done": has_data,    "active": False},
        {"id":"missing", "icon":"🧹","label":"Missing\nValues",    "done": miss_done,   "active": has_data and not miss_done},
        {"id":"preproc", "icon":"⚙️", "label":"Preprocess\ning",   "done": has_preproc, "active": miss_done and not has_preproc},
        {"id":"featsel", "icon":"🎯","label":"Feature\nSelection", "done": has_featsel, "active": has_preproc and not has_featsel},
        {"id":"model",   "icon":"🤖","label":"Modeling\n& Eval",   "done": models_done, "active": has_featsel and not models_done},
        {"id":"done",    "icon":"🏆","label":"Pipeline\nComplete", "done": models_done, "active": False},
    ]
    return steps

def render_workflow_bar():
    steps = get_pipeline_status()
    nodes_html = ""
    for i, s in enumerate(steps):
        state = "done" if s["done"] else ("active" if s["active"] else "pending")
        badge = "✓ Done" if s["done"] else ("● Active" if s["active"] else "○ Pending")
        delay = i * 0.09
        nodes_html += f"""
        <div class="wf-node {state}" style="animation-delay:{delay}s">
            <div class="wf-circle">{s["icon"]}</div>
            <div class="wf-label">{s["label"].replace(chr(10),"<br>")}</div>
            <div class="wf-badge">{badge}</div>
        </div>"""
        if i < len(steps) - 1:
            conn = "done" if s["done"] else ("active" if s["active"] else "pending")
            nodes_html += f"<div class=\"wf-connector {conn}\"></div>"
    log = st.session_state.preprocessing_log or []
    chips = "".join(f"<span class=\"wf-chip done\">{e}</span>" for e in log[-8:]) if log else "<span class=\"wf-chip\">No steps yet</span>"
    return f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #1e40af33;
                border-radius:16px;padding:1.2rem 1.6rem;box-shadow:0 4px 32px #0006;margin-bottom:1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">⚡ Live Pipeline Status</div>
        <div class="wf-wrap">{nodes_html}</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#475569;margin:0.4rem 0;">APPLIED STEPS →</div>
        <div class="wf-detail-strip">{chips}</div>
    </div>"""

def render_sidebar_pipeline():
    steps = get_pipeline_status()
    rows = ""
    for s in steps:
        state = "done" if s["done"] else ("active" if s["active"] else "pending")
        label = s["label"].replace(chr(10), " ")
        rows += f"<div class=\"sb-step {state}\"><div class=\"sb-dot\"></div>{s['icon']} {label}</div>"
    return f"<div class=\"sb-pipeline\">{rows}</div>"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 SmartEDA Pro")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;'>End-to-end ML pipeline</div>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    if uploaded_file:
        sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
        if st.button("🚀 Load Dataset"):
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
                st.session_state.df = df
                st.session_state.df_processed = df.copy()
                st.session_state.preprocessing_log = []
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 🗺️ Pipeline")
    st.markdown(render_sidebar_pipeline(), unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown("### 🎯 Target Column")
        cols = st.session_state.df_processed.columns.tolist()
        target = st.selectbox("Select target", ["None"] + cols)
        if target != "None":
            st.session_state.target_col = target
            unique_vals = st.session_state.df_processed[target].nunique()
            if unique_vals <= 15:
                st.session_state.task_type = "Classification"
                st.markdown('<span class="tag tag-cat">🏷️ Classification</span>', unsafe_allow_html=True)
            else:
                st.session_state.task_type = "Regression"
                st.markdown('<span class="tag">📈 Regression</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">SmartEDA Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload → Explore → Preprocess → Model → Predict</div>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if st.session_state.df is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="val">01</div>
            <div class="lbl">Upload CSV</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="val">02</div>
            <div class="lbl">Explore & Clean</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="val">03</div>
            <div class="lbl">Model & Evaluate</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="info-box">👈 Upload a CSV from the sidebar to get started.</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# WORKFLOW BAR
# ─────────────────────────────────────────────
st.markdown(render_workflow_bar(), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
df = st.session_state.df_processed

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📋 Overview",
    "📊 EDA & Visuals",
    "🧹 Missing Values",
    "⚙️ Preprocessing",
    "🔍 Post-Process EDA",
    "🎯 Feature Selection",
    "🤖 Modeling",
    "🧠 Auto Suggest & Predict",
    "🤖 AI Data Analyst"
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="val">{df.shape[0]:,}</div><div class="lbl">Rows</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="val">{df.shape[1]}</div><div class="lbl">Columns</div></div>""", unsafe_allow_html=True)
    with c3:
        nulls = df.isnull().sum().sum()
        st.markdown(f"""<div class="metric-card"><div class="val">{nulls}</div><div class="lbl">Missing Values</div></div>""", unsafe_allow_html=True)
    with c4:
        dups = df.duplicated().sum()
        st.markdown(f"""<div class="metric-card"><div class="val">{dups}</div><div class="lbl">Duplicates</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">First 5 Rows (head)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    st.markdown('<div class="section-header">Last 5 Rows (tail)</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(), use_container_width=True)

    st.markdown('<div class="section-header">Statistical Description</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown('<div class="section-header">Column Info (dtypes + null count)</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
        "Unique Values": df.nunique().values,
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown('<div class="section-header">Column Types</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric:**")
        for c in num_cols:
            st.markdown(f'<span class="tag">🔢 {c}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Categorical:**")
        for c in cat_cols:
            st.markdown(f'<span class="tag tag-cat">🏷️ {c}</span>', unsafe_allow_html=True)

    if dups > 0:
        st.markdown('<div class="warn-box">⚠️ Duplicate rows detected.</div>', unsafe_allow_html=True)
        if st.button("🗑️ Remove Duplicates"):
            st.session_state.df_processed = st.session_state.df_processed.drop_duplicates()
            st.session_state.preprocessing_log.append("Removed duplicate rows")
            st.rerun()

# ══════════════════════════════════════════════
# TAB 2 — EDA & VISUALS
# ══════════════════════════════════════════════
with tab2:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.markdown('<div class="section-header">Distribution Plots</div>', unsafe_allow_html=True)
    if num_cols:
        sel_num = st.selectbox("Select numeric column", num_cols, key="eda_num")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1e293b")
        for ax in axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#334155")
        axes[0].hist(df[sel_num].dropna(), bins=30, color="#38bdf8", edgecolor="#1e293b", alpha=0.85)
        axes[0].set_title(f"Histogram: {sel_num}", color="#e2e8f0")
        axes[1].boxplot(df[sel_num].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#1d4ed8", color="#38bdf8"),
                        whiskerprops=dict(color="#38bdf8"),
                        medianprops=dict(color="#f472b6", linewidth=2),
                        capprops=dict(color="#38bdf8"))
        axes[1].set_title(f"Boxplot: {sel_num}", color="#e2e8f0")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Categorical Value Counts</div>', unsafe_allow_html=True)
    if cat_cols:
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="eda_cat")
        vc = df[sel_cat].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", axis="both")
        ax.spines[:].set_color("#334155")
        colors = plt.cm.cool(np.linspace(0.2, 0.8, len(vc)))
        ax.barh(vc.index.astype(str), vc.values, color=colors)
        ax.set_title(f"Value Counts: {sel_cat}", color="#e2e8f0")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(min(len(num_cols), 14), min(len(num_cols), 10)), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.8},
                    linewidths=0.5, linecolor="#334155")
        ax.tick_params(colors="#94a3b8")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Scatter Plot</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis", num_cols, key="scatter_x")
        with c2:
            y_col = st.selectbox("Y axis", num_cols, index=1, key="scatter_y")
        hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="scatter_hue")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        if hue_col != "None":
            for val in df[hue_col].dropna().unique():
                sub = df[df[hue_col] == val]
                ax.scatter(sub[x_col], sub[y_col], label=str(val), alpha=0.6, s=20)
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        else:
            ax.scatter(df[x_col], df[y_col], color="#38bdf8", alpha=0.5, s=20)
        ax.set_xlabel(x_col, color="#94a3b8")
        ax.set_ylabel(y_col, color="#94a3b8")
        ax.set_title(f"{x_col} vs {y_col}", color="#e2e8f0")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Missing Values Heatmap</div>', unsafe_allow_html=True)
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 4), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False,
                    cmap=["#1e293b", "#f472b6"], ax=ax)
        ax.set_title("Missing Value Map (pink = missing)", color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        st.pyplot(fig)
        plt.close()
    else:
        st.markdown('<div class="success-box">✅ No missing values!</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — MISSING VALUES
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Missing Value Summary</div>', unsafe_allow_html=True)
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Missing %": miss_pct.values})
    miss_df = miss_df[miss_df["Missing"] > 0].reset_index(drop=True)

    if miss_df.empty:
        st.markdown('<div class="success-box">✅ Dataset has no missing values!</div>', unsafe_allow_html=True)
    else:
        st.dataframe(miss_df, use_container_width=True)

        cols_with_miss = miss_df["Column"].tolist()
        num_miss = [c for c in cols_with_miss if c in df.select_dtypes(include=np.number).columns]
        cat_miss = [c for c in cols_with_miss if c in df.select_dtypes(include="object").columns]

        st.markdown("---")
        st.markdown('<div class="section-header">Fill Missing Values — Numeric Columns</div>', unsafe_allow_html=True)
        if num_miss:
            for col in num_miss:
                pct = miss_pct[col]
                st.markdown(f"**`{col}`** — {int(miss[col])} missing ({pct}%)")
                if pct > 50:
                    st.markdown(f'<span class="tag tag-warn">⚠️ >50% missing — consider dropping</span>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    strategy = st.selectbox(
                        "Strategy", ["Mean", "Median", "Mode", "Constant Value", "Drop Column", "Drop Rows"],
                        key=f"num_strat_{col}"
                    )
                with c2:
                    const_val = 0
                    if strategy == "Constant Value":
                        const_val = st.number_input("Constant", key=f"num_const_{col}", value=0.0)
                with c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Apply", key=f"apply_num_{col}"):
                        tmp = st.session_state.df_processed
                        if strategy == "Mean":
                            tmp[col].fillna(tmp[col].mean(), inplace=True)
                        elif strategy == "Median":
                            tmp[col].fillna(tmp[col].median(), inplace=True)
                        elif strategy == "Mode":
                            tmp[col].fillna(tmp[col].mode()[0], inplace=True)
                        elif strategy == "Constant Value":
                            tmp[col].fillna(const_val, inplace=True)
                        elif strategy == "Drop Column":
                            tmp.drop(columns=[col], inplace=True)
                        elif strategy == "Drop Rows":
                            tmp.dropna(subset=[col], inplace=True)
                        st.session_state.df_processed = tmp
                        st.session_state.preprocessing_log.append(f"[Missing] {col}: {strategy}")
                        st.rerun()
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Fill Missing Values — Categorical Columns</div>', unsafe_allow_html=True)
        if cat_miss:
            for col in cat_miss:
                st.markdown(f"**`{col}`** — {int(miss[col])} missing ({miss_pct[col]}%)")
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    strategy = st.selectbox(
                        "Strategy", ["Mode", "Constant Value", "Drop Column", "Drop Rows"],
                        key=f"cat_strat_{col}"
                    )
                with c2:
                    const_str = ""
                    if strategy == "Constant Value":
                        const_str = st.text_input("Constant", key=f"cat_const_{col}", value="Unknown")
                with c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Apply", key=f"apply_cat_{col}"):
                        tmp = st.session_state.df_processed
                        if strategy == "Mode":
                            tmp[col].fillna(tmp[col].mode()[0], inplace=True)
                        elif strategy == "Constant Value":
                            tmp[col].fillna(const_str, inplace=True)
                        elif strategy == "Drop Column":
                            tmp.drop(columns=[col], inplace=True)
                        elif strategy == "Drop Rows":
                            tmp.dropna(subset=[col], inplace=True)
                        st.session_state.df_processed = tmp
                        st.session_state.preprocessing_log.append(f"[Missing] {col}: {strategy}")
                        st.rerun()
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        if st.button("🧹 Fill ALL numeric with Median + categorical with Mode"):
            tmp = st.session_state.df_processed
            for col in tmp.select_dtypes(include=np.number).columns:
                tmp[col].fillna(tmp[col].median(), inplace=True)
            for col in tmp.select_dtypes(include="object").columns:
                if tmp[col].isnull().sum() > 0:
                    tmp[col].fillna(tmp[col].mode()[0], inplace=True)
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append("Auto-fill: numeric→Median, categorical→Mode")
            st.rerun()

# ══════════════════════════════════════════════
# TAB 4 — PREPROCESSING
# ══════════════════════════════════════════════
with tab4:
    df = st.session_state.df_processed
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    target_c = st.session_state.target_col
    num_cols_feat = [c for c in num_cols if c != target_c] if target_c in num_cols else num_cols

    # ────────────────────────────────────────
    # SECTION A — SCALING
    # ────────────────────────────────────────
    st.markdown('<div class="section-header">🔢 A. Scaling — Numeric Columns</div>', unsafe_allow_html=True)
    if num_cols_feat:
        scale_cols = st.multiselect("Select columns to scale", num_cols_feat, key="scale_cols")
        scaler_type = st.radio("Scaler", [
            "Standard Scaler (Z-score)",
            "Min-Max Scaler [0,1]",
            "Robust Scaler (IQR)",
            "Max-Abs Scaler [-1,1]",
            "Normalizer (L2 row-wise)"
        ], horizontal=False, key="scaler_radio")
        with st.expander("ℹ️ Scaler Descriptions"):
            st.markdown("""
| Scaler | Formula | Best For |
|---|---|---|
| Standard | (x-μ)/σ | Normal distribution, SVM, LR |
| Min-Max | (x-min)/(max-min) | NN, KNN |
| Robust | (x-Q2)/(Q3-Q1) | Outliers present |
| Max-Abs | x/max(|x|) | Sparse data |
| Normalizer | x/||x||₂ | Text, cosine similarity |
""")
        if st.button("⚡ Apply Scaler", key="btn_scaler") and scale_cols:
            from sklearn.preprocessing import MaxAbsScaler, Normalizer
            tmp = st.session_state.df_processed.copy()
            if scaler_type.startswith("Standard"):       scaler = StandardScaler()
            elif scaler_type.startswith("Min"):          scaler = MinMaxScaler()
            elif scaler_type.startswith("Robust"):       scaler = RobustScaler()
            elif scaler_type.startswith("Max-Abs"):      scaler = MaxAbsScaler()
            else:                                        scaler = Normalizer()
            tmp[scale_cols] = scaler.fit_transform(tmp[scale_cols])
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Scaling ({scaler_type}): {scale_cols}")
            st.success(f"✅ {scaler_type} applied to {scale_cols}")

    st.markdown("---")

    # ────────────────────────────────────────
    # SECTION B — ALL TRANSFORMERS
    # ────────────────────────────────────────
    st.markdown('<div class="section-header">📐 B. Transformers</div>', unsafe_allow_html=True)
    if num_cols_feat:
        tr_cols = st.multiselect("Select columns to transform", num_cols_feat, key="tr_cols")
        tr_type = st.radio("Transformer", [
            "Yeo-Johnson (Power)",
            "Box-Cox (Power, +ve only)",
            "Quantile → Normal",
            "Quantile → Uniform",
            "Log1p (log(x+1))",
            "Square Root (√x)",
            "Cube Root (∛x)",
            "Square (x²)",
        ], horizontal=False, key="tr_radio")
        with st.expander("ℹ️ Transformer Descriptions"):
            st.markdown("""
| Transformer | Use When |
|---|---|
| Yeo-Johnson | Skewed data, any values |
| Box-Cox | Skewed, positive values only |
| Quantile→Normal | Force Gaussian distribution |
| Quantile→Uniform | Force uniform [0,1] |
| Log1p | Right-skewed, non-negative |
| Square Root | Moderate right skew |
| Cube Root | Negative values ok |
| Square | Left-skewed data |
""")
        if st.button("⚡ Apply Transformer", key="btn_transform") and tr_cols:
            tmp = st.session_state.df_processed.copy()
            try:
                if tr_type.startswith("Yeo"):
                    tmp[tr_cols] = PowerTransformer(method="yeo-johnson").fit_transform(tmp[tr_cols])
                elif tr_type.startswith("Box"):
                    tmp[tr_cols] = PowerTransformer(method="box-cox").fit_transform(tmp[tr_cols])
                elif "Normal" in tr_type:
                    tmp[tr_cols] = QuantileTransformer(output_distribution="normal", random_state=42).fit_transform(tmp[tr_cols])
                elif "Uniform" in tr_type:
                    tmp[tr_cols] = QuantileTransformer(output_distribution="uniform", random_state=42).fit_transform(tmp[tr_cols])
                elif tr_type.startswith("Log"):
                    tmp[tr_cols] = np.log1p(tmp[tr_cols])
                elif "Square Root" in tr_type:
                    tmp[tr_cols] = np.sqrt(tmp[tr_cols].clip(0))
                elif "Cube" in tr_type:
                    tmp[tr_cols] = np.cbrt(tmp[tr_cols])
                elif "Square" in tr_type:
                    tmp[tr_cols] = np.square(tmp[tr_cols])
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Transform ({tr_type}): {tr_cols}")
                st.success(f"✅ {tr_type} applied!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # ────────────────────────────────────────
    # SECTION C — ALL ENCODINGS
    # ────────────────────────────────────────
    st.markdown('<div class="section-header">🏷️ C. Encoding — Categorical Columns</div>', unsafe_allow_html=True)
    df = st.session_state.df_processed
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        enc_col = st.selectbox("Select column to encode", cat_cols, key="enc_col")
        enc_type = st.radio("Encoding method", [
            "Label Encoding",
            "One-Hot Encoding",
            "Ordinal (custom order)",
            "Binary Encoding",
            "Frequency Encoding",
            "Target Encoding",
            "Hash Encoding",
        ], horizontal=False, key="enc_radio")
        with st.expander("ℹ️ Encoding Descriptions"):
            st.markdown("""
| Encoding | Creates | Best For |
|---|---|---|
| Label | 1 col (int) | Ordinal, Tree models |
| One-Hot | N binary cols | Nominal, Linear models |
| Ordinal | 1 col (custom int) | Ordered categories |
| Binary | log2(N) cols | High cardinality |
| Frequency | 1 col (count%) | High cardinality |
| Target | 1 col (mean target) | High cardinality |
| Hash | N cols (hash trick) | Very high cardinality |
""")

        if enc_type == "Ordinal (custom order)":
            unique_vals = df[enc_col].dropna().unique().tolist()
            st.write("Assign integer rank to each category:")
            ord_map = {}
            cols_row = st.columns(min(len(unique_vals), 4))
            for i, v in enumerate(unique_vals):
                with cols_row[i % 4]:
                    ord_map[v] = st.number_input(f"`{v}`", value=i, key=f"ord_{enc_col}_{v}")

        if enc_type == "Target Encoding":
            if st.session_state.target_col and st.session_state.target_col in st.session_state.df_processed.columns:
                st.info(f"Will encode using target: `{st.session_state.target_col}`")
            else:
                st.warning("Set a target column in the sidebar for Target Encoding.")

        if enc_type == "Hash Encoding":
            n_hash = st.slider("Hash components (n_components)", 2, 16, 4, key="hash_n")

        if st.button("⚡ Apply Encoding", key="btn_encode"):
            tmp = st.session_state.df_processed.copy()
            try:
                if enc_type == "Label Encoding":
                    le = LabelEncoder()
                    tmp[enc_col] = le.fit_transform(tmp[enc_col].astype(str))
                    st.session_state.preprocessing_log.append(f"Label Encode: {enc_col}")
                    st.success(f"✅ Label encoded `{enc_col}`")

                elif enc_type == "One-Hot Encoding":
                    dummies = pd.get_dummies(tmp[enc_col], prefix=enc_col, drop_first=False)
                    tmp = pd.concat([tmp.drop(columns=[enc_col]), dummies], axis=1)
                    st.session_state.preprocessing_log.append(f"One-Hot Encode: {enc_col} → {dummies.shape[1]} cols")
                    st.success(f"✅ One-hot encoded `{enc_col}` → {dummies.shape[1]} columns")

                elif enc_type == "Ordinal (custom order)":
                    tmp[enc_col] = tmp[enc_col].map(ord_map)
                    st.session_state.preprocessing_log.append(f"Ordinal Encode: {enc_col}")
                    st.success(f"✅ Ordinal encoded `{enc_col}`")

                elif enc_type == "Binary Encoding":
                    le = LabelEncoder()
                    codes = le.fit_transform(tmp[enc_col].astype(str))
                    n_bits = max(1, int(np.ceil(np.log2(len(le.classes_) + 1))))
                    for b in range(n_bits):
                        tmp[f"{enc_col}_bin{b}"] = (codes >> b) & 1
                    tmp = tmp.drop(columns=[enc_col])
                    st.session_state.preprocessing_log.append(f"Binary Encode: {enc_col} → {n_bits} cols")
                    st.success(f"✅ Binary encoded `{enc_col}` → {n_bits} bit columns")

                elif enc_type == "Frequency Encoding":
                    freq = tmp[enc_col].value_counts(normalize=True)
                    tmp[enc_col] = tmp[enc_col].map(freq)
                    st.session_state.preprocessing_log.append(f"Frequency Encode: {enc_col}")
                    st.success(f"✅ Frequency encoded `{enc_col}`")

                elif enc_type == "Target Encoding":
                    tgt = st.session_state.target_col
                    if tgt and tgt in tmp.columns:
                        means = tmp.groupby(enc_col)[tgt].mean()
                        tmp[enc_col] = tmp[enc_col].map(means)
                        st.session_state.preprocessing_log.append(f"Target Encode: {enc_col}")
                        st.success(f"✅ Target encoded `{enc_col}`")
                    else:
                        st.error("Target column not set.")

                elif enc_type == "Hash Encoding":
                    import hashlib
                    def hash_encode(val, n):
                        return int(hashlib.md5(str(val).encode()).hexdigest(), 16) % n
                    for i in range(n_hash):
                        tmp[f"{enc_col}_hash{i}"] = tmp[enc_col].apply(lambda x: int(hashlib.md5(f"{x}_{i}".encode()).hexdigest(),16) % 2)
                    tmp = tmp.drop(columns=[enc_col])
                    st.session_state.preprocessing_log.append(f"Hash Encode: {enc_col} → {n_hash} cols")
                    st.success(f"✅ Hash encoded `{enc_col}` → {n_hash} columns")

                st.session_state.df_processed = tmp
                st.rerun()
            except Exception as e:
                st.error(f"Encoding error: {e}")
    else:
        st.markdown('<div class="info-box">ℹ️ No categorical columns remaining.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ────────────────────────────────────────
    # SECTION D — OUTLIER DETECTION & REMOVAL (ALL COLUMNS)
    # ────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 D. Outlier Detection & Removal</div>', unsafe_allow_html=True)
    df = st.session_state.df_processed
    num_cols_out = df.select_dtypes(include=np.number).columns.tolist()

    if num_cols_out:
        # Auto-detect outliers in ALL columns
        st.markdown("**📊 Outlier Summary — All Numeric Columns**")
        out_method_global = st.radio("Detection Method", ["IQR (1.5×)", "Z-Score (|z|>3)", "Modified Z-Score (|mz|>3.5)"], horizontal=True, key="out_method_global")

        outlier_summary = []
        for col in num_cols_out:
            col_data = df[col].dropna()
            if out_method_global.startswith("IQR"):
                Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                IQR = Q3 - Q1
                mask = (col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)
            elif out_method_global.startswith("Z"):
                z = np.abs((col_data - col_data.mean()) / col_data.std())
                mask = z > 3
            else:
                med = col_data.median()
                mad = np.median(np.abs(col_data - med))
                mz = 0.6745 * np.abs(col_data - med) / (mad + 1e-8)
                mask = mz > 3.5
            n_out = mask.sum()
            outlier_summary.append({
                "Column": col,
                "Total Rows": len(col_data),
                "Outliers": int(n_out),
                "Outlier %": round(n_out / len(col_data) * 100, 2),
                "Has Outliers": "⚠️ Yes" if n_out > 0 else "✅ No"
            })

        out_df = pd.DataFrame(outlier_summary)
        st.dataframe(out_df, use_container_width=True)

        cols_with_outliers = [r["Column"] for r in outlier_summary if r["Outliers"] > 0]
        if cols_with_outliers:
            st.markdown(f'<div class="warn-box">⚠️ {len(cols_with_outliers)} column(s) have outliers: {", ".join(cols_with_outliers)}</div>', unsafe_allow_html=True)

            # Visualize outliers
            st.markdown("**Boxplots — Columns with Outliers**")
            n_plots = len(cols_with_outliers)
            n_cols_plot = min(3, n_plots)
            n_rows_plot = (n_plots + n_cols_plot - 1) // n_cols_plot
            fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                                     figsize=(5*n_cols_plot, 4*n_rows_plot),
                                     facecolor="#1e293b")
            axes = np.array(axes).flatten() if n_plots > 1 else [axes]
            for i, col in enumerate(cols_with_outliers):
                ax = axes[i]
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#94a3b8")
                ax.spines[:].set_color("#334155")
                bp = ax.boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="#1d4ed8", color="#38bdf8"),
                    whiskerprops=dict(color="#f472b6"),
                    medianprops=dict(color="#fbbf24", linewidth=2),
                    flierprops=dict(marker="o", color="#f87171", markersize=5),
                    capprops=dict(color="#38bdf8"))
                ax.set_title(col, color="#e2e8f0", fontsize=10)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.markdown("**Remove Outliers:**")

            c1, c2 = st.columns(2)
            with c1:
                remove_type = st.radio("Apply to", ["Single column", "All columns with outliers"], horizontal=True, key="out_apply")
            with c2:
                if remove_type == "Single column":
                    out_single_col = st.selectbox("Pick column", cols_with_outliers, key="out_single")

            cap_or_remove = st.radio("Action", ["Remove rows", "Cap (Winsorize to IQR bounds)"], horizontal=True, key="out_action")

            if st.button("🗑️ Apply Outlier Treatment", key="btn_outlier"):
                tmp = st.session_state.df_processed.copy()
                before = len(tmp)
                target_cols = cols_with_outliers if remove_type == "All columns with outliers" else [out_single_col]

                for col in target_cols:
                    col_data = tmp[col].dropna()
                    if out_method_global.startswith("IQR"):
                        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    elif out_method_global.startswith("Z"):
                        mean_, std_ = col_data.mean(), col_data.std()
                        lo, hi = mean_ - 3*std_, mean_ + 3*std_
                    else:
                        med = col_data.median()
                        mad = np.median(np.abs(col_data - med)) + 1e-8
                        lo = med - 3.5 * mad / 0.6745
                        hi = med + 3.5 * mad / 0.6745

                    if cap_or_remove == "Remove rows":
                        tmp = tmp[(tmp[col] >= lo) & (tmp[col] <= hi)]
                    else:
                        tmp[col] = tmp[col].clip(lo, hi)

                removed = before - len(tmp)
                st.session_state.df_processed = tmp.reset_index(drop=True)
                action_word = "removed" if cap_or_remove == "Remove rows" else "capped"
                st.session_state.preprocessing_log.append(
                    f"Outlier {action_word} ({out_method_global}): {target_cols}, {removed} rows affected"
                )
                st.success(f"✅ {removed} rows {action_word} in {target_cols}")
                st.rerun()

            if st.button("🗑️ Remove ALL Outliers (all columns, IQR)", key="btn_outlier_all"):
                tmp = st.session_state.df_processed.copy()
                before = len(tmp)
                for col in tmp.select_dtypes(include=np.number).columns:
                    Q1, Q3 = tmp[col].quantile(0.25), tmp[col].quantile(0.75)
                    IQR = Q3 - Q1
                    tmp = tmp[(tmp[col] >= Q1-1.5*IQR) & (tmp[col] <= Q3+1.5*IQR)]
                removed = before - len(tmp)
                st.session_state.df_processed = tmp.reset_index(drop=True)
                st.session_state.preprocessing_log.append(f"Remove ALL outliers IQR: {removed} rows removed")
                st.success(f"✅ Removed {removed} total outlier rows across all columns")
                st.rerun()
        else:
            st.markdown('<div class="success-box">✅ No outliers found in any numeric column!</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── LOG
    st.markdown('<div class="section-header">📝 Preprocessing Log</div>', unsafe_allow_html=True)
    if st.session_state.preprocessing_log:
        for i, step in enumerate(st.session_state.preprocessing_log, 1):
            st.markdown(f'<span class="tag">#{i}</span> {step}', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No preprocessing steps applied yet.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Current Dataset Preview:**")
    st.dataframe(st.session_state.df_processed.head(10), use_container_width=True)

    if st.button("↩️ Reset to Original", key="btn_reset"):
        st.session_state.df_processed = st.session_state.df.copy()
        st.session_state.preprocessing_log = []
        st.rerun()

# ══════════════════════════════════════════════
# TAB 5 — POST-PREPROCESSING EDA
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔍 Post-Preprocessing EDA & Visuals</div>', unsafe_allow_html=True)
    df_post = st.session_state.df_processed
    num_post = df_post.select_dtypes(include=np.number).columns.tolist()
    cat_post = df_post.select_dtypes(include="object").columns.tolist()

    st.markdown('<div class="info-box">ℹ️ These charts reflect your dataset AFTER all preprocessing steps.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="val">{df_post.shape[0]:,}</div><div class="lbl">Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="val">{df_post.shape[1]}</div><div class="lbl">Columns</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="val">{df_post.isnull().sum().sum()}</div><div class="lbl">Missing</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="val">{len(num_post)}</div><div class="lbl">Numeric Cols</div></div>', unsafe_allow_html=True)

    if num_post:
        # 1) Distribution grid — all numeric columns
        st.markdown('<div class="section-header">📊 Distribution — All Numeric Columns</div>', unsafe_allow_html=True)
        n_num = len(num_post)
        n_cols_g = min(3, n_num)
        n_rows_g = (n_num + n_cols_g - 1) // n_cols_g
        fig, axes = plt.subplots(n_rows_g, n_cols_g,
                                  figsize=(5*n_cols_g, 3.5*n_rows_g),
                                  facecolor="#0d0f1a")
        axes_flat = np.array(axes).flatten() if n_num > 1 else [axes]
        colors_cycle = ["#38bdf8","#818cf8","#f472b6","#34d399","#fbbf24","#a78bfa","#fb7185","#60a5fa"]
        for i, col in enumerate(num_post):
            ax = axes_flat[i]
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.spines[:].set_color("#334155")
            ax.hist(df_post[col].dropna(), bins=25, color=colors_cycle[i%len(colors_cycle)],
                    edgecolor="#0d0f1a", alpha=0.85)
            ax.set_title(col, color="#e2e8f0", fontsize=9)
            skew_val = df_post[col].skew()
            ax.set_xlabel(f"skew={skew_val:.2f}", color="#64748b", fontsize=7)
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 2) Boxplot grid — outlier check after preprocessing
        st.markdown('<div class="section-header">📦 Boxplots — Outlier Check After Preprocessing</div>', unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(n_rows_g, n_cols_g,
                                    figsize=(5*n_cols_g, 3.5*n_rows_g),
                                    facecolor="#0d0f1a")
        axes2_flat = np.array(axes2).flatten() if n_num > 1 else [axes2]
        for i, col in enumerate(num_post):
            ax = axes2_flat[i]
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.spines[:].set_color("#334155")
            ax.boxplot(df_post[col].dropna(), patch_artist=True,
                boxprops=dict(facecolor=colors_cycle[i%len(colors_cycle)]+"55", color=colors_cycle[i%len(colors_cycle)]),
                whiskerprops=dict(color=colors_cycle[i%len(colors_cycle)]),
                medianprops=dict(color="#fbbf24", linewidth=2),
                flierprops=dict(marker="o", color="#f87171", markersize=4),
                capprops=dict(color=colors_cycle[i%len(colors_cycle)]))
            ax.set_title(col, color="#e2e8f0", fontsize=9)
        for j in range(i+1, len(axes2_flat)):
            axes2_flat[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # 3) Correlation heatmap
        st.markdown('<div class="section-header">🔥 Correlation Heatmap (Post-Processing)</div>', unsafe_allow_html=True)
        if len(num_post) >= 2:
            corr_post = df_post[num_post].corr()
            fig3, ax3 = plt.subplots(figsize=(min(len(num_post),14), min(len(num_post),10)), facecolor="#1e293b")
            ax3.set_facecolor("#1e293b")
            sns.heatmap(corr_post, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax3, annot_kws={"size":8},
                        cbar_kws={"shrink":0.8},
                        linewidths=0.5, linecolor="#334155")
            ax3.tick_params(colors="#94a3b8")
            ax3.set_title("Correlation Matrix — Processed Data", color="#e2e8f0")
            st.pyplot(fig3)
            plt.close()

        # 4) Pairplot (limited to 5 cols)
        st.markdown('<div class="section-header">🔗 Pairplot (up to 5 numeric columns)</div>', unsafe_allow_html=True)
        pair_cols = num_post[:5]
        target_pair = st.session_state.target_col
        try:
            pair_df = df_post[pair_cols].dropna()
            if target_pair and target_pair in df_post.columns and df_post[target_pair].nunique() <= 10:
                pair_df[target_pair] = df_post[target_pair].loc[pair_df.index]
                fig_pair = sns.pairplot(pair_df, hue=target_pair,
                    plot_kws={"alpha":0.5,"s":15},
                    diag_kws={"fill":True})
            else:
                fig_pair = sns.pairplot(pair_df,
                    plot_kws={"alpha":0.5,"s":15,"color":"#38bdf8"},
                    diag_kws={"fill":True,"color":"#818cf8"})
            fig_pair.figure.patch.set_facecolor("#0d0f1a")
            for ax_ in fig_pair.axes.flatten():
                if ax_:
                    ax_.set_facecolor("#1e293b")
                    ax_.tick_params(colors="#94a3b8", labelsize=7)
                    ax_.spines[:].set_color("#334155")
            st.pyplot(fig_pair.figure)
            plt.close()
        except Exception as e:
            st.warning(f"Pairplot error: {e}")

        # 5) Skewness & Kurtosis table
        st.markdown('<div class="section-header">📐 Skewness & Kurtosis</div>', unsafe_allow_html=True)
        sk_df = pd.DataFrame({
            "Column": num_post,
            "Skewness": [round(df_post[c].skew(), 3) for c in num_post],
            "Kurtosis": [round(df_post[c].kurtosis(), 3) for c in num_post],
            "Distribution": ["Normal ✅" if abs(df_post[c].skew()) < 0.5 else ("Moderate ⚠️" if abs(df_post[c].skew()) < 1 else "Skewed ❌") for c in num_post]
        })
        st.dataframe(sk_df, use_container_width=True)

        # 6) Before vs After comparison
        if st.session_state.df is not None and st.session_state.preprocessing_log:
            st.markdown('<div class="section-header">📈 Before vs After Preprocessing</div>', unsafe_allow_html=True)
            df_orig = st.session_state.df
            orig_num = [c for c in num_post if c in df_orig.columns]
            if orig_num:
                compare_col = st.selectbox("Select column to compare", orig_num, key="compare_col")
                fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1e293b")
                for ax_ in axes_cmp:
                    ax_.set_facecolor("#1e293b")
                    ax_.tick_params(colors="#94a3b8")
                    ax_.spines[:].set_color("#334155")
                axes_cmp[0].hist(df_orig[compare_col].dropna(), bins=25, color="#f472b6", edgecolor="#0d0f1a", alpha=0.8)
                axes_cmp[0].set_title(f"BEFORE: {compare_col}", color="#e2e8f0")
                axes_cmp[0].set_xlabel(f"skew={df_orig[compare_col].skew():.3f}", color="#94a3b8")
                axes_cmp[1].hist(df_post[compare_col].dropna(), bins=25, color="#38bdf8", edgecolor="#0d0f1a", alpha=0.8)
                axes_cmp[1].set_title(f"AFTER: {compare_col}", color="#e2e8f0")
                axes_cmp[1].set_xlabel(f"skew={df_post[compare_col].skew():.3f}", color="#94a3b8")
                plt.tight_layout()
                st.pyplot(fig_cmp)
                plt.close()

    if cat_post:
        st.markdown('<div class="section-header">🏷️ Categorical Columns After Encoding</div>', unsafe_allow_html=True)
        for col in cat_post[:6]:
            vc = df_post[col].value_counts().head(15)
            fig_c, ax_c = plt.subplots(figsize=(8, 3), facecolor="#1e293b")
            ax_c.set_facecolor("#1e293b")
            ax_c.tick_params(colors="#94a3b8", labelsize=8)
            ax_c.spines[:].set_color("#334155")
            ax_c.barh(vc.index.astype(str), vc.values, color="#818cf8", alpha=0.85)
            ax_c.set_title(col, color="#e2e8f0")
            ax_c.invert_yaxis()
            st.pyplot(fig_c)
            plt.close()

    st.markdown("---")
    st.markdown("**Processed Dataset:**")
    st.dataframe(df_post.head(10), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 — FEATURE SELECTION
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Feature Selection</div>', unsafe_allow_html=True)
    df = st.session_state.df_processed
    target = st.session_state.target_col
    task = st.session_state.task_type

    if target is None:
        st.markdown('<div class="warn-box">⚠️ Please set a target column in the sidebar first.</div>', unsafe_allow_html=True)
        st.stop()

    if target not in df.columns:
        st.error("Target column not found in processed dataset. It may have been dropped.")
        st.stop()

    X = df.drop(columns=[target])
    y = df[target]

    # Only keep numeric X
    X_num = X.select_dtypes(include=np.number)

    if X_num.empty:
        st.markdown('<div class="warn-box">⚠️ No numeric feature columns available. Please encode categorical columns first.</div>', unsafe_allow_html=True)
        st.stop()

    # Drop rows with NaN in X_num or y
    valid_idx = X_num.dropna().index.intersection(y.dropna().index)
    X_clean = X_num.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    st.markdown(f"**Features available:** {X_clean.shape[1]}  |  **Samples:** {X_clean.shape[0]}")

    # Correlation with target
    st.markdown('<div class="section-header">Correlation with Target</div>', unsafe_allow_html=True)
    try:
        corrs = X_clean.corrwith(y_clean).abs().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(3, len(corrs)*0.35)), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        colors = plt.cm.cool(np.linspace(0.2, 0.9, len(corrs)))
        ax.barh(corrs.index, corrs.values, color=colors)
        ax.set_xlabel("Absolute Correlation", color="#94a3b8")
        ax.set_title("Feature Correlation with Target", color="#e2e8f0")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not compute correlation: {e}")

    # SelectKBest
    st.markdown('<div class="section-header">SelectKBest</div>', unsafe_allow_html=True)
    k = st.slider("Number of top features (k)", 1, max(1, X_clean.shape[1]), min(5, X_clean.shape[1]))
    fs_method = st.radio(
        "Score function",
        ["F-statistic", "Mutual Information"],
        horizontal=True
    )
    if st.button("🎯 Run Feature Selection"):
        try:
            if task == "Classification":
                sf = f_classif if fs_method == "F-statistic" else mutual_info_classif
            else:
                sf = f_regression if fs_method == "F-statistic" else mutual_info_regression
            selector = SelectKBest(score_func=sf, k=k)
            selector.fit(X_clean, y_clean)
            scores = pd.Series(selector.scores_, index=X_clean.columns).sort_values(ascending=False)
            selected = X_clean.columns[selector.get_support()].tolist()

            st.markdown("**Feature Scores:**")
            fig, ax = plt.subplots(figsize=(10, max(3, len(scores)*0.35)), facecolor="#1e293b")
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#334155")
            bar_colors = ["#38bdf8" if c in selected else "#334155" for c in scores.index]
            ax.barh(scores.index, scores.values, color=bar_colors)
            ax.set_title(f"Top {k} Features Highlighted", color="#e2e8f0")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()

            st.markdown(f'<div class="success-box">✅ Selected features: {selected}</div>', unsafe_allow_html=True)

            if st.button("✂️ Keep only selected features + target", key="apply_feat_sel"):
                keep = selected + [target]
                st.session_state.df_processed = st.session_state.df_processed[keep]
                st.session_state.preprocessing_log.append(f"Feature selection: kept {selected}")
                st.rerun()
        except Exception as e:
            st.error(f"Feature selection error: {e}")

# ══════════════════════════════════════════════
# TAB 7 — MODELING
# ══════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
    df = st.session_state.df_processed
    target = st.session_state.target_col
    task = st.session_state.task_type

    if target is None:
        st.markdown('<div class="warn-box">⚠️ Set a target column in the sidebar.</div>', unsafe_allow_html=True)
        st.stop()

    if target not in df.columns:
        st.error("Target column not found. It may have been dropped during preprocessing.")
        st.stop()

    X = df.drop(columns=[target]).select_dtypes(include=np.number)
    y = df[target]
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.empty:
        st.warning("No numeric feature columns. Please encode categorical columns first.")
        st.stop()

    st.markdown(f"**Task:** `{task}`  |  **Features:** {X.shape[1]}  |  **Samples:** {X.shape[0]}")

    test_size = st.slider("Test split size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)

    # Model lists
    if task == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }
    else:
        model_options = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
        }

    selected_models = st.multiselect("Select models to train", list(model_options.keys()), default=list(model_options.keys())[:3])

    if st.button("🚀 Train Models"):
        if not selected_models:
            st.warning("Select at least one model.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        results = []
        best_model = None
        best_score = -np.inf

        prog = st.progress(0, text="Training models...")
        for i, name in enumerate(selected_models):
            model = model_options[name]
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if task == "Classification":
                    test_score = accuracy_score(y_test, y_pred)
                    cv_score = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy").mean()
                    results.append({"Model": name, "Test Accuracy": round(test_score, 4), "CV Accuracy (mean)": round(cv_score, 4)})
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    cv_score = cross_val_score(model, X, y, cv=cv_folds, scoring="r2").mean()
                    results.append({"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R²": round(r2, 4), "CV R² (mean)": round(cv_score, 4)})
                if test_score if task == "Classification" else r2 > best_score:
                    best_score = test_score if task == "Classification" else r2
                    best_model = (name, model, y_pred)
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})
            prog.progress((i+1)/len(selected_models), text=f"Trained: {name}")

        prog.empty()
        st.session_state.models_trained = True
        res_df = pd.DataFrame(results)
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
        st.dataframe(res_df, use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        score_col = "Test Accuracy" if task == "Classification" else "R²"
        if score_col in res_df.columns:
            vals = res_df.set_index("Model")[score_col].dropna()
            colors_bar = plt.cm.cool(np.linspace(0.2, 0.9, len(vals)))
            ax.bar(vals.index, vals.values, color=colors_bar)
            ax.set_ylabel(score_col, color="#94a3b8")
            ax.set_title("Model Performance", color="#e2e8f0")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)
            plt.close()

        # Best model detail
        if best_model:
            name, model, y_pred = best_model
            st.markdown(f'<div class="success-box">🏆 Best Model: <b>{name}</b></div>', unsafe_allow_html=True)

            if task == "Classification":
                st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                # Confusion matrix
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1e293b")
                ax.set_facecolor("#1e293b")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            cbar_kws={"shrink": 0.8},
                            linewidths=0.5, linecolor="#334155",
                            annot_kws={"color": "white", "size": 12})
                ax.set_xlabel("Predicted", color="#94a3b8")
                ax.set_ylabel("Actual", color="#94a3b8")
                ax.set_title(f"Confusion Matrix — {name}", color="#e2e8f0")
                ax.tick_params(colors="#94a3b8")
                st.pyplot(fig)
                plt.close()

            else:
                st.markdown('<div class="section-header">Actual vs Predicted</div>', unsafe_allow_html=True)
                fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#1e293b")
                for ax in axes:
                    ax.set_facecolor("#1e293b")
                    ax.tick_params(colors="#94a3b8")
                    ax.spines[:].set_color("#334155")
                axes[0].scatter(y_test, y_pred, color="#38bdf8", alpha=0.5, s=20)
                min_val = min(y_test.min(), np.array(y_pred).min())
                max_val = max(y_test.max(), np.array(y_pred).max())
                axes[0].plot([min_val, max_val], [min_val, max_val], color="#f472b6", linewidth=2, label="Perfect")
                axes[0].set_xlabel("Actual", color="#94a3b8")
                axes[0].set_ylabel("Predicted", color="#94a3b8")
                axes[0].set_title("Actual vs Predicted", color="#e2e8f0")
                axes[0].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
                residuals = np.array(y_test) - np.array(y_pred)
                axes[1].hist(residuals, bins=30, color="#818cf8", edgecolor="#1e293b", alpha=0.85)
                axes[1].set_title("Residuals", color="#e2e8f0")
                axes[1].set_xlabel("Residual", color="#94a3b8")
                st.pyplot(fig)
                plt.close()

            # Feature importance
            if hasattr(model, "feature_importances_"):
                st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
                fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, max(3, len(fi)*0.35)), facecolor="#1e293b")
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#94a3b8")
                ax.spines[:].set_color("#334155")
                colors_fi = plt.cm.cool(np.linspace(0.2, 0.9, len(fi)))
                ax.barh(fi.index, fi.values, color=colors_fi)
                ax.invert_yaxis()
                ax.set_title("Feature Importances", color="#e2e8f0")
                st.pyplot(fig)
                plt.close()

# ══════════════════════════════════════════════
# TAB 8 — AUTO SUGGEST & PREDICT
# ══════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">🧠 Smart Auto Analyst — Data ko Analyse karo, Best Config Suggest karo, Predict karo</div>', unsafe_allow_html=True)

    df_auto = st.session_state.df_processed
    target_auto = st.session_state.target_col
    task_auto   = st.session_state.task_type

    if target_auto is None or target_auto not in df_auto.columns:
        st.markdown('<div class="warn-box">⚠️ Pehle sidebar mein Target Column set karo.</div>', unsafe_allow_html=True)
        st.stop()

    # ────────────────────────────────────────────────────────
    # STEP 1 — DATA ANALYSIS REPORT
    # ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #1e40af44;
                border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#38bdf8;margin-bottom:0.4rem;">
            STEP 1 — AUTO DATA ANALYSIS
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;">
            App tumhare data ko scan karta hai aur ek detailed report banata hai.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Analyse My Data", key="btn_analyse"):
        st.session_state.auto_analysed = True

    if st.session_state.get("auto_analysed", False):

        X_auto = df_auto.drop(columns=[target_auto]).select_dtypes(include=np.number)
        y_auto = df_auto[target_auto]
        valid_idx = X_auto.dropna().index.intersection(y_auto.dropna().index)
        X_auto = X_auto.loc[valid_idx]
        y_auto = y_auto.loc[valid_idx]

        n_samples   = len(df_auto)
        n_features  = X_auto.shape[1]
        n_classes   = y_auto.nunique() if task_auto == "Classification" else None
        missing_pct = df_auto.isnull().sum().sum() / (df_auto.shape[0]*df_auto.shape[1]) * 100
        num_outliers_total = 0
        skew_cols = []
        for col in X_auto.columns:
            Q1,Q3 = X_auto[col].quantile(0.25), X_auto[col].quantile(0.75)
            IQR = Q3-Q1
            num_outliers_total += ((X_auto[col]<Q1-1.5*IQR)|(X_auto[col]>Q3+1.5*IQR)).sum()
            if abs(X_auto[col].skew()) > 1:
                skew_cols.append(col)

        class_balance = None
        imbalanced = False
        if task_auto == "Classification":
            vc = y_auto.value_counts(normalize=True)
            class_balance = vc.to_dict()
            imbalanced = vc.min() < 0.15

        # ── Analysis Cards
        st.markdown('<div class="section-header">📊 Data Profile Report</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="val">{n_samples:,}</div><div class="lbl">Samples</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="val">{n_features}</div><div class="lbl">Features</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="val">{round(missing_pct,1)}%</div><div class="lbl">Missing %</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="val">{num_outliers_total}</div><div class="lbl">Outlier Cells</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Build analysis findings
        findings = []
        if n_samples < 500:
            findings.append(("⚠️", "Small dataset", f"Only {n_samples} rows — avoid complex models. Use simple/regularized models."))
        elif n_samples < 5000:
            findings.append(("ℹ️", "Medium dataset", f"{n_samples} rows — most models will work well."))
        else:
            findings.append(("✅", "Large dataset", f"{n_samples} rows — can use complex models like Gradient Boosting."))

        if n_features > 20:
            findings.append(("⚠️", "High dimensionality", f"{n_features} features — feature selection strongly recommended."))
        elif n_features < 3:
            findings.append(("⚠️", "Very few features", f"Only {n_features} features — model accuracy may be limited."))

        if missing_pct > 0:
            findings.append(("⚠️", "Missing values exist", f"{round(missing_pct,1)}% cells missing — fill before training."))
        else:
            findings.append(("✅", "No missing values", "Data is complete."))

        if skew_cols:
            findings.append(("⚠️", "Skewed columns", f"{len(skew_cols)} columns have high skew: {skew_cols[:3]} — apply log/power transform."))

        if num_outliers_total > 0:
            findings.append(("⚠️", "Outliers detected", f"{num_outliers_total} outlier values — consider robust scaler or removal."))
        else:
            findings.append(("✅", "No outliers", "Data looks clean."))

        if task_auto == "Classification":
            if imbalanced:
                findings.append(("🔴", "Class imbalance!", f"Minority class < 15% — use class_weight='balanced' or SMOTE."))
            else:
                findings.append(("✅", "Balanced classes", "No class imbalance detected."))

        for icon,title,desc in findings:
            color = "#052e16" if "✅" in icon else ("#1c1407" if "⚠️" in icon else "#1a0a0a")
            border = "#166534" if "✅" in icon else ("#b45309" if "⚠️" in icon else "#991b1b")
            txt = "#86efac" if "✅" in icon else ("#fcd34d" if "⚠️" in icon else "#fca5a5")
            st.markdown(f"""
            <div style="background:{color};border:1px solid {border};border-radius:10px;
                        padding:0.7rem 1rem;margin:0.3rem 0;display:flex;gap:0.8rem;align-items:flex-start;">
                <span style="font-size:1.1rem">{icon}</span>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{txt};font-weight:700">{title}</div>
                    <div style="font-size:0.8rem;color:#94a3b8;margin-top:2px">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.session_state.auto_findings = findings
        st.session_state.auto_Xauto   = X_auto
        st.session_state.auto_yauto   = y_auto
        st.session_state.auto_stats   = dict(n_samples=n_samples, n_features=n_features,
                                              missing_pct=missing_pct, skew_cols=skew_cols,
                                              num_outliers_total=num_outliers_total,
                                              imbalanced=imbalanced)

    # ────────────────────────────────────────────────────────
    # STEP 2 — AUTO SUGGESTIONS
    # ────────────────────────────────────────────────────────
    if st.session_state.get("auto_findings"):
        st.markdown("---")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #4f46e544;
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#818cf8;margin-bottom:0.4rem;">
                STEP 2 — SMART SUGGESTIONS
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;">
                Data analysis ke basis par app ne yeh config suggest ki hai.
            </div>
        </div>
        """, unsafe_allow_html=True)

        stats = st.session_state.auto_stats
        n_samples   = stats["n_samples"]
        n_features  = stats["n_features"]
        skew_cols   = stats["skew_cols"]
        num_out     = stats["num_outliers_total"]
        imbalanced  = stats["imbalanced"]

        # ── Suggested Scaler
        if num_out > 10:
            sug_scaler = "Robust Scaler (IQR)"
            sug_scaler_reason = "Outliers detected — Robust Scaler is best"
        else:
            sug_scaler = "Standard Scaler (Z-score)"
            sug_scaler_reason = "Clean data — Standard Scaler works well"

        # ── Suggested Transformer
        if len(skew_cols) > 0:
            sug_transform = "Yeo-Johnson (Power)"
            sug_transform_reason = f"Skewed columns detected ({len(skew_cols)}) — Power transform recommended"
        else:
            sug_transform = "None needed"
            sug_transform_reason = "Data is not highly skewed"

        # ── Suggested Models
        sug_models = []
        sug_reasons = []
        if task_auto == "Classification":
            if n_samples < 500:
                sug_models = ["Logistic Regression", "Naive Bayes", "Decision Tree"]
                sug_reasons = ["Small data — simple models work best", "Fast & works with small data", "Interpretable, low overfitting"]
            elif n_samples < 5000:
                sug_models = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
                sug_reasons = ["Handles mixed data well", "High accuracy on tabular data", "Good baseline"]
            else:
                sug_models = ["Gradient Boosting", "Random Forest", "SVM"]
                sug_reasons = ["Best for large tabular data", "Robust & accurate", "Good for high dimensions"]
            if imbalanced:
                sug_models.insert(0, "Random Forest (balanced)")
                sug_reasons.insert(0, "⚠️ Class imbalance detected — needs class_weight='balanced'")
        else:
            if n_samples < 500:
                sug_models = ["Ridge Regression", "Lasso Regression", "Decision Tree"]
                sug_reasons = ["Regularization prevents overfitting", "Auto feature selection", "Handles non-linearity"]
            elif n_samples < 5000:
                sug_models = ["Random Forest", "Gradient Boosting", "Ridge Regression"]
                sug_reasons = ["Best for tabular regression", "High accuracy", "Stable baseline"]
            else:
                sug_models = ["Gradient Boosting", "Random Forest", "SVR"]
                sug_reasons = ["Top performer on large data", "Robust to noise", "Good for complex patterns"]

        # ── Best features suggestion
        X_sug = st.session_state.auto_Xauto
        y_sug = st.session_state.auto_yauto
        try:
            from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
            if task_auto == "Classification":
                fi_model = RFC(n_estimators=50, random_state=42)
            else:
                fi_model = RFR(n_estimators=50, random_state=42)
            fi_model.fit(X_sug, y_sug)
            fi_series = pd.Series(fi_model.feature_importances_, index=X_sug.columns).sort_values(ascending=False)
            top_features = fi_series[fi_series.cumsum() <= 0.85].index.tolist()
            if not top_features:
                top_features = fi_series.head(min(5, len(fi_series))).index.tolist()
            sug_features = top_features
        except:
            sug_features = X_sug.columns.tolist()[:5]

        # Display suggestions
        st.markdown('<div class="section-header">💡 Suggested Configuration</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e40af44;border-radius:12px;padding:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#38bdf8;margin-bottom:0.6rem;">⚙️ RECOMMENDED SCALER</div>
                <div style="font-size:1rem;color:#e2e8f0;font-weight:600">{sug_scaler}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{sug_scaler_reason}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #4f46e544;border-radius:12px;padding:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#818cf8;margin-bottom:0.6rem;">📐 RECOMMENDED TRANSFORM</div>
                <div style="font-size:1rem;color:#e2e8f0;font-weight:600">{sug_transform}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{sug_transform_reason}</div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #16603044;border-radius:12px;padding:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#4ade80;margin-bottom:0.6rem;">🎯 TOP FEATURES (85% importance)</div>
                {"".join(f'<span style="background:#14532d;color:#86efac;border-radius:20px;padding:2px 10px;font-size:0.72rem;margin:2px;display:inline-block;">{f}</span>' for f in sug_features)}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 Suggested Models (ranked)</div>', unsafe_allow_html=True)
        for i,(m,r) in enumerate(zip(sug_models, sug_reasons)):
            medal = ["🥇","🥈","🥉","4️⃣"][min(i,3)]
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                        padding:0.7rem 1rem;margin:0.3rem 0;display:flex;gap:1rem;align-items:center;">
                <span style="font-size:1.3rem">{medal}</span>
                <div style="flex:1">
                    <div style="font-family:'Space Mono',monospace;font-size:0.82rem;color:#e2e8f0">{m}</div>
                    <div style="font-size:0.75rem;color:#64748b">{r}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.session_state.sug_models   = sug_models
        st.session_state.sug_scaler   = sug_scaler
        st.session_state.sug_features = sug_features

    # ────────────────────────────────────────────────────────
    # STEP 3 — ONE-CLICK AUTO TRAIN
    # ────────────────────────────────────────────────────────
    if st.session_state.get("sug_models"):
        st.markdown("---")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #16603044;
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#4ade80;margin-bottom:0.4rem;">
                STEP 3 — AUTO TRAIN WITH SUGGESTED CONFIG
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;">
                Ek click mein sabhi suggested models train honge aur best model select hoga.
            </div>
        </div>
        """, unsafe_allow_html=True)

        test_sz = st.slider("Test split", 0.1, 0.4, 0.2, 0.05, key="auto_test_sz")
        cv_auto = st.slider("CV folds", 2, 10, 5, key="auto_cv")

        if st.button("🚀 Auto Train Best Models", key="btn_auto_train"):
            X_tr = st.session_state.auto_Xauto[st.session_state.sug_features] if st.session_state.sug_features else st.session_state.auto_Xauto
            y_tr = st.session_state.auto_yauto

            # Build model map from suggested names
            all_clf = {
                "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
                "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
                "Random Forest (balanced)": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
                "SVM": SVC(class_weight="balanced", probability=True),
                "KNN": KNeighborsClassifier(),
            }
            all_reg = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
            }

            model_map = all_clf if task_auto == "Classification" else all_reg
            models_to_run = [m for m in st.session_state.sug_models if m in model_map]
            if not models_to_run:
                models_to_run = list(model_map.keys())[:3]

            X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=test_sz, random_state=42)
            auto_results = []
            best_auto_model = None
            best_auto_score = -np.inf

            prog2 = st.progress(0, text="Auto training...")
            for i, mname in enumerate(models_to_run):
                mdl = model_map[mname]
                try:
                    mdl.fit(X_train, y_train)
                    y_pred_a = mdl.predict(X_test)
                    if task_auto == "Classification":
                        sc = accuracy_score(y_test, y_pred_a)
                        cv_sc = cross_val_score(mdl, X_tr, y_tr, cv=cv_auto, scoring="accuracy").mean()
                        f1 = cross_val_score(mdl, X_tr, y_tr, cv=cv_auto, scoring="f1_weighted").mean()
                        auto_results.append({"Model": mname, "Test Acc": round(sc,4), "CV Acc": round(cv_sc,4), "CV F1": round(f1,4)})
                        score_v = sc
                    else:
                        r2_v = r2_score(y_test, y_pred_a)
                        rmse_v = np.sqrt(mean_squared_error(y_test, y_pred_a))
                        cv_sc = cross_val_score(mdl, X_tr, y_tr, cv=cv_auto, scoring="r2").mean()
                        auto_results.append({"Model": mname, "R²": round(r2_v,4), "RMSE": round(rmse_v,4), "CV R²": round(cv_sc,4)})
                        score_v = r2_v
                    if score_v > best_auto_score:
                        best_auto_score = score_v
                        best_auto_model = (mname, mdl, y_pred_a, y_test, X_test)
                except Exception as ex:
                    auto_results.append({"Model": mname, "Error": str(ex)})
                prog2.progress((i+1)/len(models_to_run), text=f"✅ Trained: {mname}")

            prog2.empty()
            st.session_state.auto_results      = auto_results
            st.session_state.best_auto_model   = best_auto_model
            st.session_state.auto_Xtest        = X_test
            st.session_state.auto_ytest        = y_test
            st.session_state.models_trained    = True

        # Show results if trained
        if st.session_state.get("auto_results"):
            st.markdown('<div class="section-header">📊 Auto Training Results</div>', unsafe_allow_html=True)
            res_df2 = pd.DataFrame(st.session_state.auto_results)
            st.dataframe(res_df2, use_container_width=True)

            score_key = "Test Acc" if task_auto == "Classification" else "R²"
            if score_key in res_df2.columns:
                fig_r, ax_r = plt.subplots(figsize=(8,3), facecolor="#1e293b")
                ax_r.set_facecolor("#1e293b"); ax_r.tick_params(colors="#94a3b8"); ax_r.spines[:].set_color("#334155")
                vals_r = res_df2.set_index("Model")[score_key].dropna()
                col_r = ["#22c55e" if v==vals_r.max() else "#38bdf8" for v in vals_r.values]
                ax_r.bar(vals_r.index, vals_r.values, color=col_r)
                ax_r.set_ylabel(score_key, color="#94a3b8")
                ax_r.set_title("Auto Model Comparison (🟢 = Best)", color="#e2e8f0")
                plt.xticks(rotation=20, ha="right", color="#94a3b8")
                st.pyplot(fig_r); plt.close()

            if st.session_state.best_auto_model:
                bname, bmdl, by_pred, by_test, bX_test = st.session_state.best_auto_model
                st.markdown(f'<div class="success-box">🏆 Best Model: <b>{bname}</b> — Score: {round(best_auto_score,4) if "best_auto_score" in dir() else ""}</div>', unsafe_allow_html=True)

                if task_auto == "Classification":
                    st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
                    rpt = classification_report(by_test, by_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(rpt).transpose().round(3), use_container_width=True)
                    cm_a = confusion_matrix(by_test, by_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6,5), facecolor="#1e293b")
                    ax_cm.set_facecolor("#1e293b")
                    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Greens", ax=ax_cm,
                                linewidths=0.5, linecolor="#334155", annot_kws={"color":"white","size":12})
                    ax_cm.set_xlabel("Predicted", color="#94a3b8"); ax_cm.set_ylabel("Actual", color="#94a3b8")
                    ax_cm.set_title(f"Confusion Matrix — {bname}", color="#e2e8f0")
                    ax_cm.tick_params(colors="#94a3b8")
                    st.pyplot(fig_cm); plt.close()
                else:
                    fig_ap, axes_ap = plt.subplots(1,2, figsize=(12,4), facecolor="#1e293b")
                    for ax_ in axes_ap:
                        ax_.set_facecolor("#1e293b"); ax_.tick_params(colors="#94a3b8"); ax_.spines[:].set_color("#334155")
                    axes_ap[0].scatter(by_test, by_pred, color="#38bdf8", alpha=0.5, s=20)
                    mn,mx = min(by_test.min(),np.array(by_pred).min()), max(by_test.max(),np.array(by_pred).max())
                    axes_ap[0].plot([mn,mx],[mn,mx], color="#f472b6", lw=2, label="Perfect")
                    axes_ap[0].set_title("Actual vs Predicted", color="#e2e8f0")
                    axes_ap[0].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
                    residuals_a = np.array(by_test) - np.array(by_pred)
                    axes_ap[1].hist(residuals_a, bins=25, color="#818cf8", edgecolor="#1e293b", alpha=0.85)
                    axes_ap[1].set_title("Residuals Distribution", color="#e2e8f0")
                    st.pyplot(fig_ap); plt.close()

                if hasattr(bmdl, "feature_importances_"):
                    fi2 = pd.Series(bmdl.feature_importances_, index=st.session_state.sug_features if st.session_state.sug_features else st.session_state.auto_Xauto.columns).sort_values(ascending=False)
                    fig_fi2, ax_fi2 = plt.subplots(figsize=(8, max(3,len(fi2)*0.4)), facecolor="#1e293b")
                    ax_fi2.set_facecolor("#1e293b"); ax_fi2.tick_params(colors="#94a3b8"); ax_fi2.spines[:].set_color("#334155")
                    ax_fi2.barh(fi2.index, fi2.values, color=plt.cm.cool(np.linspace(0.2,0.9,len(fi2))))
                    ax_fi2.invert_yaxis(); ax_fi2.set_title("Feature Importances (Auto Best Model)", color="#e2e8f0")
                    st.pyplot(fig_fi2); plt.close()

    # ────────────────────────────────────────────────────────
    # STEP 4 — LIVE PREDICTION
    # ────────────────────────────────────────────────────────
    if st.session_state.get("best_auto_model"):
        st.markdown("---")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #f472b644;
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#f472b6;margin-bottom:0.4rem;">
                STEP 4 — LIVE PREDICTION
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;">
                Apni values daalo — best model turant predict karega!
            </div>
        </div>
        """, unsafe_allow_html=True)

        bname, bmdl, _, _, _ = st.session_state.best_auto_model
        feat_cols = st.session_state.sug_features if st.session_state.sug_features else st.session_state.auto_Xauto.columns.tolist()

        st.markdown(f"**Model:** `{bname}`  |  **Features:** {feat_cols}")
        st.markdown('<div class="section-header">Enter Feature Values</div>', unsafe_allow_html=True)

        input_vals = {}
        df_ref = st.session_state.auto_Xauto
        n_feat = len(feat_cols)
        n_inp_cols = min(3, n_feat)
        inp_col_list = st.columns(n_inp_cols)
        for i, col in enumerate(feat_cols):
            with inp_col_list[i % n_inp_cols]:
                col_min = float(df_ref[col].min()) if col in df_ref else 0.0
                col_max = float(df_ref[col].max()) if col in df_ref else 100.0
                col_mean = float(df_ref[col].mean()) if col in df_ref else 0.0
                input_vals[col] = st.number_input(
                    f"{col}",
                    value=round(col_mean, 4),
                    min_value=round(col_min - abs(col_min), 4),
                    max_value=round(col_max + abs(col_max), 4),
                    key=f"pred_inp_{col}",
                    help=f"Range: [{round(col_min,2)}, {round(col_max,2)}]  |  Mean: {round(col_mean,2)}"
                )

        if st.button("🎯 Predict Now!", key="btn_predict"):
            input_df = pd.DataFrame([input_vals])
            try:
                prediction = bmdl.predict(input_df)[0]
                if task_auto == "Classification":
                    proba = None
                    if hasattr(bmdl, "predict_proba"):
                        proba = bmdl.predict_proba(input_df)[0]
                        classes = bmdl.classes_
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#052e16,#0f2744);
                                border:1px solid #22c55e;border-radius:14px;padding:1.5rem 2rem;
                                text-align:center;margin:1rem 0;">
                        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#4ade80;margin-bottom:0.4rem">PREDICTION RESULT</div>
                        <div style="font-size:2.2rem;font-weight:700;color:#22c55e">{prediction}</div>
                        <div style="font-size:0.8rem;color:#64748b;margin-top:4px">Model: {bname}</div>
                    </div>""", unsafe_allow_html=True)
                    if proba is not None:
                        st.markdown("**Confidence per class:**")
                        for cls, prob in zip(classes, proba):
                            bar_w = int(prob*100)
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:10px;margin:4px 0">
                                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#94a3b8;min-width:100px">{cls}</div>
                                <div style="flex:1;background:#1e293b;border-radius:20px;height:10px;overflow:hidden">
                                    <div style="width:{bar_w}%;height:100%;background:{'#22c55e' if cls==prediction else '#38bdf8'};border-radius:20px"></div>
                                </div>
                                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#e2e8f0;min-width:40px">{round(prob*100,1)}%</div>
                            </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#052e16,#0f2744);
                                border:1px solid #22c55e;border-radius:14px;padding:1.5rem 2rem;
                                text-align:center;margin:1rem 0;">
                        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#4ade80;margin-bottom:0.4rem">PREDICTED VALUE</div>
                        <div style="font-size:2.5rem;font-weight:700;color:#22c55e">{round(float(prediction),4)}</div>
                        <div style="font-size:0.8rem;color:#64748b;margin-top:4px">Model: {bname}</div>
                    </div>""", unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Prediction error: {ex}")

        st.markdown("---")
        st.markdown('<div class="section-header">📦 Batch Prediction (CSV Upload)</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_upload")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            missing_cols = [c for c in feat_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
            else:
                try:
                    batch_preds = bmdl.predict(batch_df[feat_cols])
                    batch_df["🎯 Prediction"] = batch_preds
                    if task_auto == "Classification" and hasattr(bmdl, "predict_proba"):
                        proba_batch = bmdl.predict_proba(batch_df[feat_cols])
                        batch_df["Confidence"] = proba_batch.max(axis=1).round(3)
                    st.success(f"✅ {len(batch_df)} rows predicted!")
                    st.dataframe(batch_df.head(50), use_container_width=True)
                    csv_out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions CSV", csv_out, "predictions.csv", "text/csv", key="dl_preds")
                except Exception as ex:
                    st.error(f"Batch prediction error: {ex}")

# ══════════════════════════════════════════════
# TAB 9 — AI DATA ANALYST (LLM)
# ══════════════════════════════════════════════
with tab9:

    # ── init chat history
    if "llm_chat_history" not in st.session_state or st.session_state.llm_chat_history is None:
        st.session_state.llm_chat_history = []

    # ── helper: build rich data context for LLM
    def build_data_context():
        df_c = st.session_state.df_processed
        if df_c is None:
            return "No dataset loaded yet."
        target = st.session_state.target_col
        task   = st.session_state.task_type
        log    = st.session_state.preprocessing_log or []
        num_c  = df_c.select_dtypes(include=np.number).columns.tolist()
        cat_c  = df_c.select_dtypes(include="object").columns.tolist()
        desc   = df_c.describe(include="all").round(3).to_string()
        miss   = df_c.isnull().sum()
        corr_str = ""
        if len(num_c) >= 2:
            corr   = df_c[num_c].corr().round(3)
            top_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    top_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
            top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            corr_str = "\n".join([f"  {a} ↔ {b}: {r:.3f}" for a,b,r in top_pairs[:10]])
        skew_info = "\n".join([f"  {c}: skew={round(df_c[c].skew(),3)}" for c in num_c[:10]])
        ctx = f"""
=== DATASET CONTEXT ===
Shape: {df_c.shape[0]} rows × {df_c.shape[1]} columns
Task type: {task or 'Not set'}
Target column: {target or 'Not set'}
Numeric columns: {num_c}
Categorical columns: {cat_c}
Missing values: {miss[miss>0].to_dict() or 'None'}
Preprocessing steps applied: {log or 'None'}

=== STATISTICAL SUMMARY ===
{desc}

=== SKEWNESS (numeric cols) ===
{skew_info}

=== TOP CORRELATIONS ===
{corr_str or 'Not enough numeric columns'}

=== SAMPLE DATA (first 5 rows) ===
{df_c.head(5).to_string()}
"""
        return ctx.strip()

    # ── helper: call LLM
    def call_llm(provider, api_key, model_name, messages, data_ctx):
        system_prompt = f"""You are an expert Data Scientist and ML Engineer acting as an interactive AI Data Analyst.
You have full access to the user's dataset. Your job is to:
1. Analyze the data deeply and answer questions about it
2. Suggest the best preprocessing steps, models, and hyperparameters
3. Explain patterns, correlations, outliers, and distributions
4. Write Python/pandas code snippets when useful
5. Give actionable, specific recommendations — not generic advice
6. Be conversational, clear, and precise

CURRENT DATASET CONTEXT:
{data_ctx}

Always reference specific column names, values, and statistics from the dataset in your answers.
Format code blocks with ```python ... ```.
"""
        try:
            if provider == "Claude (Anthropic)":
                import anthropic as ac
                client = ac.Anthropic(api_key=api_key)
                claude_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=claude_msgs
                )
                return response.content[0].text

            elif provider == "GPT (OpenAI)":
                import openai as oa
                client = oa.OpenAI(api_key=api_key)
                oai_msgs = [{"role": "system", "content": system_prompt}]
                oai_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=oai_msgs,
                    max_tokens=2048,
                    temperature=0.3
                )
                return response.choices[0].message.content

            elif provider == "Gemini (Google)":
                import google.generativeai as genai_sdk
                genai_sdk.configure(api_key=api_key)
                gm = genai_sdk.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                history_gemini = []
                for m in messages[:-1]:
                    role = "user" if m["role"] == "user" else "model"
                    history_gemini.append({"role": role, "parts": [m["content"]]})
                chat = gm.start_chat(history=history_gemini)
                response = chat.send_message(messages[-1]["content"])
                return response.text

            elif provider == "Ollama (Local)":
                import requests as req
                payload = {
                    "model": model_name,
                    "messages": [{"role": "system", "content": system_prompt}] +
                                [{"role": m["role"], "content": m["content"]} for m in messages],
                    "stream": False
                }
                resp = req.post("http://localhost:11434/api/chat", json=payload, timeout=120)
                return resp.json()["message"]["content"]

            elif provider == "Groq":
                from groq import Groq
                client = Groq(api_key=api_key)
                groq_msgs = [{"role": "system", "content": system_prompt}]
                groq_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=groq_msgs,
                    max_tokens=2048,
                    temperature=0.3
                )
                return response.choices[0].message.content

            elif provider == "HuggingFace (Inference API)":
                import requests as req
                headers = {"Authorization": f"Bearer {api_key}"}
                full_prompt = system_prompt + "\n\n"
                for m in messages:
                    full_prompt += f"{'Human' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
                full_prompt += "Assistant:"
                payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 1024}}
                resp = req.post(
                    f"https://api-inference.huggingface.co/models/{model_name}",
                    headers=headers, json=payload, timeout=60
                )
                result = resp.json()
                if isinstance(result, list):
                    return result[0].get("generated_text", str(result))
                return str(result)

        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    # ─────────────────────────────────
    # UI HEADER
    # ─────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#1a0a2e);border:1px solid #7c3aed44;
                border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                    background:linear-gradient(90deg,#a78bfa,#f472b6,#38bdf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:700;margin-bottom:0.4rem;">
            🤖 AI Data Analyst
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;">
            Choose any LLM provider → paste API key → ask anything about your data.
            The AI has full context of your dataset, statistics, correlations & preprocessing steps.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────
    # PROVIDER CONFIG (collapsible)
    # ─────────────────────────────────
    with st.expander("⚙️ LLM Provider Configuration", expanded=st.session_state.llm_provider is None):

        PROVIDERS = {
            "Claude (Anthropic)": {
                "models": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
                "key_label": "Anthropic API Key",
                "key_url": "https://console.anthropic.com/",
                "needs_key": True,
                "color": "#f472b6"
            },
            "GPT (OpenAI)": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "key_label": "OpenAI API Key",
                "key_url": "https://platform.openai.com/api-keys",
                "needs_key": True,
                "color": "#4ade80"
            },
            "Gemini (Google)": {
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
                "key_label": "Google AI Studio Key",
                "key_url": "https://aistudio.google.com/",
                "needs_key": True,
                "color": "#38bdf8"
            },
            "Groq": {
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
                "key_label": "Groq API Key",
                "key_url": "https://console.groq.com/",
                "needs_key": True,
                "color": "#fbbf24"
            },
            "HuggingFace (Inference API)": {
                "models": ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta", "tiiuae/falcon-7b-instruct"],
                "key_label": "HuggingFace Token",
                "key_url": "https://huggingface.co/settings/tokens",
                "needs_key": True,
                "color": "#a78bfa"
            },
            "Ollama (Local)": {
                "models": ["llama3", "mistral", "codellama", "gemma2", "phi3", "llava"],
                "key_label": "No key needed (runs locally)",
                "key_url": "https://ollama.ai/",
                "needs_key": False,
                "color": "#34d399"
            },
        }

        # Provider pill selector
        st.markdown("**Select Provider:**")
        p_cols = st.columns(len(PROVIDERS))
        for i, (pname, pinfo) in enumerate(PROVIDERS.items()):
            with p_cols[i]:
                selected_mark = "✓ " if st.session_state.llm_provider == pname else ""
                if st.button(f"{selected_mark}{pname.split(' ')[0]}", key=f"prov_{i}",
                             use_container_width=True):
                    st.session_state.llm_provider = pname
                    st.session_state.llm_model_name = PROVIDERS[pname]["models"][0]
                    st.rerun()

        if st.session_state.llm_provider:
            pinfo = PROVIDERS[st.session_state.llm_provider]
            st.markdown(f"**Selected:** `{st.session_state.llm_provider}`")

            # Model picker
            model_choice = st.selectbox(
                "Model",
                pinfo["models"],
                index=0,
                key="llm_model_sel"
            )
            # Allow custom model name
            custom_model = st.text_input(
                "Or enter custom model name (leave blank to use above)",
                value="",
                key="llm_custom_model"
            )
            st.session_state.llm_model_name = custom_model.strip() if custom_model.strip() else model_choice

            # API Key
            if pinfo["needs_key"]:
                api_key_input = st.text_input(
                    pinfo["key_label"],
                    type="password",
                    placeholder="Paste your API key here...",
                    key="llm_api_input"
                )
                if api_key_input:
                    st.session_state.llm_api_key = api_key_input
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#64748b;">Get key: <a href="{pinfo["key_url"]}" target="_blank" style="color:#38bdf8;">{pinfo["key_url"]}</a></div>',
                    unsafe_allow_html=True
                )
            else:
                st.session_state.llm_api_key = "local"
                st.markdown('<div class="success-box">✅ Ollama runs locally — no API key needed. Make sure Ollama is running on port 11434.</div>', unsafe_allow_html=True)

            # Install hint
            install_map = {
                "Claude (Anthropic)": "pip install anthropic",
                "GPT (OpenAI)": "pip install openai",
                "Gemini (Google)": "pip install google-generativeai",
                "Groq": "pip install groq",
                "HuggingFace (Inference API)": "pip install requests",
                "Ollama (Local)": "pip install requests",
            }
            st.markdown(f'<div style="font-size:0.75rem;color:#64748b;margin-top:6px;">Install: <code>{install_map[st.session_state.llm_provider]}</code></div>', unsafe_allow_html=True)

            if st.button("💾 Save Config", key="save_llm_config"):
                st.success(f"✅ Config saved: {st.session_state.llm_provider} / {st.session_state.llm_model_name}")

    # ─────────────────────────────────
    # DATA CONTEXT PREVIEW
    # ─────────────────────────────────
    with st.expander("🗃️ Data Context sent to LLM (click to preview)"):
        ctx_preview = build_data_context()
        st.code(ctx_preview[:3000] + ("\n... [truncated]" if len(ctx_preview) > 3000 else ""), language="text")

    # ─────────────────────────────────
    # QUICK PROMPT BUTTONS
    # ─────────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Analysis Prompts</div>', unsafe_allow_html=True)
    quick_prompts = [
        ("📊 Full EDA Summary", "Give me a complete EDA summary of this dataset. Include key statistics, distributions, outliers, correlations, and any data quality issues you notice."),
        ("🧹 Preprocessing Plan", "Based on the dataset, suggest the best step-by-step preprocessing pipeline. Include which columns need scaling, encoding, outlier treatment, and why."),
        ("🤖 Best Model", "Which ML model would work best for this dataset and why? Consider dataset size, feature types, task type, and any class imbalance."),
        ("🔍 Feature Analysis", "Analyze the feature importance and correlations. Which features are most predictive? Are there any redundant or irrelevant features I should drop?"),
        ("⚠️ Data Issues", "What are the main data quality issues in this dataset? Missing values, outliers, skewness, multicollinearity — give me a prioritized action list."),
        ("💡 Insights", "What are the top 5 most interesting insights or patterns you can find in this dataset? Be specific with column names and values."),
        ("🐍 Code", "Write Python pandas code to fully preprocess this dataset — handle missing values, encode categoricals, scale numerics, and split into train/test."),
        ("📈 Model Improvement", "How can I improve my model's accuracy for this dataset? Suggest hyperparameter tuning, feature engineering, and ensembling strategies."),
    ]

    qp_cols = st.columns(4)
    for i, (label, prompt) in enumerate(quick_prompts):
        with qp_cols[i % 4]:
            if st.button(label, key=f"qp_{i}", use_container_width=True):
                st.session_state.llm_chat_history.append({"role": "user", "content": prompt})
                st.session_state.llm_pending = True
                st.rerun()

    # ─────────────────────────────────
    # CHAT INTERFACE
    # ─────────────────────────────────
    st.markdown('<div class="section-header">💬 Chat with AI Analyst</div>', unsafe_allow_html=True)

    # Render chat history
    chat_history = st.session_state.llm_chat_history or []
    for msg in chat_history:
        is_user = msg["role"] == "user"
        bg      = "#1e3a8a22" if is_user else "#14532d22"
        border  = "#3b82f655" if is_user else "#22c55e55"
        icon    = "🧑‍💻" if is_user else "🤖"
        label   = "You" if is_user else f"AI ({st.session_state.llm_model_name or 'LLM'})"
        content = msg["content"]

        # Render code blocks nicely
        parts = re.split(r"(```[\s\S]*?```)", content)
        rendered = ""
        for part in parts:
            if part.startswith("```"):
                lang_match = re.match(r"```(\w+)?\n?([\s\S]*?)```", part)
                code_content = lang_match.group(2) if lang_match else part[3:-3]
                rendered += f'<pre style="background:#0d0f1a;border:1px solid #334155;border-radius:8px;padding:0.8rem;font-size:0.78rem;color:#e2e8f0;overflow-x:auto;margin:6px 0">{code_content}</pre>'
            else:
                escaped = part.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                rendered += f'<span style="font-size:0.85rem">{escaped}</span>'

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:12px;
                    padding:1rem 1.2rem;margin:0.5rem 0;">
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                        color:{'#93c5fd' if is_user else '#4ade80'};margin-bottom:0.5rem;">
                {icon} {label}
            </div>
            <div style="color:#e2e8f0;line-height:1.6">{rendered}</div>
        </div>
        """, unsafe_allow_html=True)

    # Auto-call LLM if pending
    if st.session_state.get("llm_pending") and chat_history:
        if not st.session_state.llm_provider:
            st.markdown('<div class="warn-box">⚠️ Please select an LLM provider and enter API key above.</div>', unsafe_allow_html=True)
            st.session_state.llm_pending = False
        elif st.session_state.llm_provider != "Ollama (Local)" and not st.session_state.llm_api_key:
            st.markdown('<div class="warn-box">⚠️ Please enter your API key in the configuration above.</div>', unsafe_allow_html=True)
            st.session_state.llm_pending = False
        else:
            with st.spinner(f"🤖 {st.session_state.llm_provider} is analyzing your data..."):
                data_ctx = build_data_context()
                response = call_llm(
                    provider   = st.session_state.llm_provider,
                    api_key    = st.session_state.llm_api_key,
                    model_name = st.session_state.llm_model_name,
                    messages   = chat_history,
                    data_ctx   = data_ctx
                )
                st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
                st.session_state.llm_pending = False
                st.rerun()

    # Message input
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input(
        "Ask anything about your data... (e.g. 'Which features have highest correlation?')",
        key="llm_chat_input"
    )
    if user_input:
        st.session_state.llm_chat_history.append({"role": "user", "content": user_input})
        st.session_state.llm_pending = True
        st.rerun()

    # Chat controls
    c_a, c_b, c_c = st.columns([2, 1, 1])
    with c_b:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.llm_chat_history = []
            st.session_state.llm_pending = False
            st.rerun()
    with c_c:
        if chat_history:
            chat_text = "\n\n".join([f"{'USER' if m['role']=='user' else 'AI'}: {m['content']}" for m in chat_history])
            st.download_button("⬇️ Export Chat", chat_text.encode(), "ai_analysis.txt", "text/plain", key="dl_chat")

    # ─────────────────────────────────
    # AUTO FULL REPORT
    # ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📄 Generate Full AI Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ Ek click mein LLM poore dataset ka comprehensive report generate karega — EDA, issues, recommendations, model suggestions sab kuch.</div>', unsafe_allow_html=True)

    if st.button("📊 Generate Full AI Data Report", key="btn_full_report"):
        full_report_prompt = """Generate a comprehensive Data Analysis Report for this dataset with the following sections:

## 1. Executive Summary
Brief overview of the dataset and key findings.

## 2. Data Quality Assessment
- Missing values analysis
- Outlier detection
- Data type issues
- Duplicate rows

## 3. Statistical Insights
- Key distributions (normal/skewed)
- Important correlations
- Surprising patterns or anomalies

## 4. Feature Analysis
- Most important features
- Features to drop or engineer
- Multicollinearity issues

## 5. Preprocessing Recommendations
Step-by-step pipeline with reasons for each step.

## 6. Model Recommendations
- Best 3 models for this task with reasons
- Expected performance range
- Key hyperparameters to tune

## 7. Action Items (Priority Order)
Numbered list of what to do next.

Be specific — use actual column names and statistics from the dataset."""

        st.session_state.llm_chat_history.append({"role": "user", "content": full_report_prompt})
        st.session_state.llm_pending = True
        st.rerun()

