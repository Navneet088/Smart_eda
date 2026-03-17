import streamlit as st

# ── Page config (must be first Streamlit call)
st.set_page_config(
    page_title="SmartEDA Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Internal utilities
from utils.styles   import inject_css
from utils.state    import init_state
from utils.workflow import render_workflow_bar, render_sidebar_pipeline

# ── Tab renderers
from tabs.tab_overview      import render_tab_overview
from tabs.tab_eda           import render_tab_eda
from tabs.tab_missing       import render_tab_missing
from tabs.tab_preprocessing import render_tab_preprocessing
from tabs.tab_posteda       import render_tab_posteda
from tabs.tab_features      import render_tab_features
from tabs.tab_modeling      import render_tab_modeling
from tabs.tab_automl               import render_tab_automl
from tabs.tab_llm                  import render_tab_llm
from tabs.tab_feature_engineering  import render_tab_feature_engineering

# ══════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════
inject_css()
init_state()

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
import pandas as pd

with st.sidebar:
    st.markdown("## 🔬 SmartEDA Pro")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;'>End-to-end ML pipeline</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    if uploaded_file:
        sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
        if st.button("🚀 Load Dataset"):
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
                st.session_state.df           = df
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
        cols   = st.session_state.df_processed.columns.tolist()
        target = st.selectbox("Select target", ["None"] + cols)
        if target != "None":
            st.session_state.target_col = target
            unique_vals = st.session_state.df_processed[target].nunique()
            if unique_vals <= 15:
                st.session_state.task_type = "Classification"
                st.markdown('<span class="tag tag-cat">🏷️ Classification</span>',
                            unsafe_allow_html=True)
            else:
                st.session_state.task_type = "Regression"
                st.markdown('<span class="tag">📈 Regression</span>',
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown('<div class="hero-title">SmartEDA Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload → Explore → Preprocess → Model → Predict</div>',
            unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if st.session_state.df is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="val">01</div><div class="lbl">Upload CSV</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="val">02</div><div class="lbl">Explore & Clean</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="val">03</div><div class="lbl">Model & Evaluate</div></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="info-box">👈 Upload a CSV from the sidebar to get started.</div>',
                unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════
# LIVE WORKFLOW BAR
# ══════════════════════════════════════════════
st.markdown(render_workflow_bar(), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📋 Overview",
    "📊 EDA & Visuals",
    "🧹 Missing Values",
    "⚙️ Preprocessing",
    "🔍 Post-Process EDA",
    "🎯 Feature Selection",
    "⚗️ Feature Engineering",
    "🤖 Modeling",
    "🧠 Auto Suggest & Predict",
    "🤖 AI Data Analyst",
])

with tab1:  render_tab_overview()
with tab2:  render_tab_eda()
with tab3:  render_tab_missing()
with tab4:  render_tab_preprocessing()
with tab5:  render_tab_posteda()
with tab6:  render_tab_features()
with tab7:  render_tab_feature_engineering()
with tab8:  render_tab_modeling()
with tab9:  render_tab_automl()
with tab10: render_tab_llm()
