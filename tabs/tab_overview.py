import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, LabelEncoder,
    OneHotEncoder, OrdinalEncoder, MaxAbsScaler, Normalizer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

def render_tab_overview():
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════
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