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

def render_tab_preprocessing():
    # TAB 4 — PREPROCESSING
    # ══════════════════════════════════════════════
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