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

def render_tab_posteda():
    df = st.session_state.df_processed
    # TAB 5 — POST-PREPROCESSING EDA
    # ══════════════════════════════════════════════
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
