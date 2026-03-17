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

def render_tab_eda():
    # TAB 2 — EDA & VISUALS
    # ══════════════════════════════════════════════
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