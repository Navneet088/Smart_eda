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

def render_tab_features():
    # TAB 6 — FEATURE SELECTION
    # ══════════════════════════════════════════════
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