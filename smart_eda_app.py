import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, LabelEncoder, OneHotEncoder
)
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
for key in ["df", "df_processed", "target_col", "task_type", "preprocessing_log"]:
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview",
    "📊 EDA & Visuals",
    "🧹 Missing Values",
    "⚙️ Preprocessing",
    "🎯 Feature Selection",
    "🤖 Modeling"
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
    if st.session_state.target_col in num_cols:
        num_cols_feat = [c for c in num_cols if c != st.session_state.target_col]
    else:
        num_cols_feat = num_cols

    # ── SCALING
    st.markdown('<div class="section-header">🔢 Scaling — Numeric Columns</div>', unsafe_allow_html=True)
    if num_cols_feat:
        scale_cols = st.multiselect("Select columns to scale", num_cols_feat, key="scale_cols")
        scaler_type = st.radio("Scaler", ["Standard Scaler (Z-score)", "Min-Max Scaler", "Robust Scaler"], horizontal=True)

        with st.expander("ℹ️ Scaler Info"):
            st.markdown("""
- **Standard Scaler** — mean=0, std=1. Best for normally distributed data.
- **Min-Max Scaler** — scales to [0,1]. Sensitive to outliers.
- **Robust Scaler** — uses median/IQR. Best when outliers are present.
""")
        if st.button("⚡ Apply Scaler") and scale_cols:
            tmp = st.session_state.df_processed.copy()
            if scaler_type.startswith("Standard"):
                scaler = StandardScaler()
            elif scaler_type.startswith("Min"):
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            tmp[scale_cols] = scaler.fit_transform(tmp[scale_cols])
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Scaling ({scaler_type}): {scale_cols}")
            st.success(f"✅ {scaler_type} applied to {scale_cols}")

    st.markdown("---")

    # ── POWER TRANSFORM
    st.markdown('<div class="section-header">📐 Power Transformation</div>', unsafe_allow_html=True)
    if num_cols_feat:
        pt_cols = st.multiselect("Select columns for power transform", num_cols_feat, key="pt_cols")
        pt_type = st.radio("Method", ["Yeo-Johnson", "Box-Cox (positive values only)"], horizontal=True)
        with st.expander("ℹ️ Power Transform Info"):
            st.markdown("""
- **Yeo-Johnson** — works on any values. Makes distribution more Gaussian.
- **Box-Cox** — only positive values. Strong normalization effect.
""")
        if st.button("⚡ Apply Power Transform") and pt_cols:
            tmp = st.session_state.df_processed.copy()
            method = "yeo-johnson" if pt_type.startswith("Yeo") else "box-cox"
            try:
                pt = PowerTransformer(method=method)
                tmp[pt_cols] = pt.fit_transform(tmp[pt_cols])
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Power Transform ({method}): {pt_cols}")
                st.success(f"✅ Power transform applied!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # ── ENCODING
    st.markdown('<div class="section-header">🏷️ Encoding — Categorical Columns</div>', unsafe_allow_html=True)
    if cat_cols:
        enc_col = st.selectbox("Select column to encode", cat_cols, key="enc_col")
        enc_type = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding", "Ordinal (manual order)"], horizontal=True)

        with st.expander("ℹ️ Encoding Info"):
            st.markdown("""
- **Label Encoding** — converts categories to integers (0,1,2...). Good for ordinal data or tree models.
- **One-Hot Encoding** — creates binary dummy columns. Best for nominal categories + linear models.
- **Ordinal** — assign custom integer order to categories.
""")

        if enc_type == "Ordinal (manual order)":
            unique_vals = df[enc_col].dropna().unique().tolist()
            st.write("Drag-order not available in Streamlit — assign integer ranks below:")
            ord_map = {}
            cols_row = st.columns(min(len(unique_vals), 4))
            for i, v in enumerate(unique_vals):
                with cols_row[i % 4]:
                    ord_map[v] = st.number_input(f"`{v}`", value=i, key=f"ord_{enc_col}_{v}")

        if st.button("⚡ Apply Encoding"):
            tmp = st.session_state.df_processed.copy()
            if enc_type == "Label Encoding":
                le = LabelEncoder()
                tmp[enc_col] = le.fit_transform(tmp[enc_col].astype(str))
                st.session_state.preprocessing_log.append(f"Label Encode: {enc_col}")
                st.success(f"✅ Label encoded `{enc_col}`")
            elif enc_type == "One-Hot Encoding":
                dummies = pd.get_dummies(tmp[enc_col], prefix=enc_col, drop_first=False)
                tmp = pd.concat([tmp.drop(columns=[enc_col]), dummies], axis=1)
                st.session_state.preprocessing_log.append(f"One-Hot Encode: {enc_col}")
                st.success(f"✅ One-hot encoded `{enc_col}` → {dummies.shape[1]} columns")
            elif enc_type == "Ordinal (manual order)":
                tmp[enc_col] = tmp[enc_col].map(ord_map)
                st.session_state.preprocessing_log.append(f"Ordinal Encode: {enc_col}")
                st.success(f"✅ Ordinal encoded `{enc_col}`")
            st.session_state.df_processed = tmp
            st.rerun()

    st.markdown("---")

    # ── OUTLIER REMOVAL
    st.markdown('<div class="section-header">🚨 Outlier Removal</div>', unsafe_allow_html=True)
    df = st.session_state.df_processed
    num_cols2 = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols2:
        out_col = st.selectbox("Select column", num_cols2, key="out_col")
        out_method = st.radio("Method", ["IQR (1.5x)", "Z-Score (|z|>3)"], horizontal=True)
        if st.button("🗑️ Remove Outliers"):
            tmp = st.session_state.df_processed.copy()
            before = len(tmp)
            if out_method.startswith("IQR"):
                Q1 = tmp[out_col].quantile(0.25)
                Q3 = tmp[out_col].quantile(0.75)
                IQR = Q3 - Q1
                tmp = tmp[~((tmp[out_col] < Q1 - 1.5*IQR) | (tmp[out_col] > Q3 + 1.5*IQR))]
            else:
                z = np.abs((tmp[out_col] - tmp[out_col].mean()) / tmp[out_col].std())
                tmp = tmp[z <= 3]
            removed = before - len(tmp)
            st.session_state.df_processed = tmp.reset_index(drop=True)
            st.session_state.preprocessing_log.append(f"Outlier removal ({out_method}): {out_col}, removed {removed} rows")
            st.success(f"✅ Removed {removed} outlier rows from `{out_col}`")

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

    if st.button("↩️ Reset to Original"):
        st.session_state.df_processed = st.session_state.df.copy()
        st.session_state.preprocessing_log = []
        st.rerun()

# ══════════════════════════════════════════════
# TAB 5 — FEATURE SELECTION
# ══════════════════════════════════════════════
with tab5:
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
# TAB 6 — MODELING
# ══════════════════════════════════════════════
with tab6:
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
