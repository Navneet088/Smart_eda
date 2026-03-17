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

def render_tab_missing():
    # TAB 3 — MISSING VALUES
    # ══════════════════════════════════════════════
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

        st.markdown("---")
        st.markdown('<div class="section-header">🗑️ Drop Null Values</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            before = len(st.session_state.df_processed)
            null_rows = st.session_state.df_processed.isnull().any(axis=1).sum()
            st.markdown(f"""<div class="metric-card">
                <div class="val">{null_rows}</div>
                <div class="lbl">Rows with any null</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            all_null_rows = st.session_state.df_processed.isnull().all(axis=1).sum()
            st.markdown(f"""<div class="metric-card">
                <div class="val">{all_null_rows}</div>
                <div class="lbl">Rows all null</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            null_cols = st.session_state.df_processed.isnull().any(axis=0).sum()
            st.markdown(f"""<div class="metric-card">
                <div class="val">{null_cols}</div>
                <div class="lbl">Cols with nulls</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("🗑️ Drop ALL rows with ANY null", use_container_width=True, key="drop_any_null"):
                tmp = st.session_state.df_processed.copy()
                before = len(tmp)
                tmp = tmp.dropna(how="any").reset_index(drop=True)
                removed = before - len(tmp)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Dropped rows with ANY null: {removed} rows removed")
                st.success(f"✅ {removed} rows dropped. Remaining: {len(tmp)}")
                st.rerun()

        with col_b:
            if st.button("🗑️ Drop rows where ALL values null", use_container_width=True, key="drop_all_null"):
                tmp = st.session_state.df_processed.copy()
                before = len(tmp)
                tmp = tmp.dropna(how="all").reset_index(drop=True)
                removed = before - len(tmp)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Dropped rows where ALL null: {removed} rows removed")
                st.success(f"✅ {removed} rows dropped. Remaining: {len(tmp)}")
                st.rerun()

        with col_c:
            thresh_val = st.number_input(
                "Keep rows with at least N non-null values",
                min_value=1,
                max_value=int(st.session_state.df_processed.shape[1]),
                value=int(st.session_state.df_processed.shape[1]),
                key="drop_thresh"
            )
            if st.button("🗑️ Drop by Threshold", use_container_width=True, key="drop_thresh_btn"):
                tmp = st.session_state.df_processed.copy()
                before = len(tmp)
                tmp = tmp.dropna(thresh=thresh_val).reset_index(drop=True)
                removed = before - len(tmp)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Dropped rows thresh<{thresh_val}: {removed} rows removed")
                st.success(f"✅ {removed} rows dropped. Remaining: {len(tmp)}")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        # Drop NULL columns
        st.markdown("**Drop Columns with too many nulls:**")
        col_thresh = st.slider(
            "Drop columns where null % is MORE than:",
            min_value=0, max_value=100, value=50, step=5,
            key="col_null_thresh",
            format="%d%%"
        )
        null_pct_cols = (st.session_state.df_processed.isnull().sum() / len(st.session_state.df_processed) * 100)
        cols_to_drop = null_pct_cols[null_pct_cols > col_thresh].index.tolist()
        if cols_to_drop:
            st.markdown(f'<div class="warn-box">⚠️ Columns that will be dropped: <b>{cols_to_drop}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ No columns exceed {col_thresh}% null threshold.</div>', unsafe_allow_html=True)

        if st.button(f"🗑️ Drop Columns > {col_thresh}% null", key="drop_null_cols", disabled=len(cols_to_drop)==0):
            tmp = st.session_state.df_processed.copy()
            tmp.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Dropped columns >{col_thresh}% null: {cols_to_drop}")
            st.success(f"✅ Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
            st.rerun()

    # ══════════════════════════════════════════════