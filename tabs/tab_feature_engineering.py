import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def render_tab_feature_engineering():

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#1a0a2e);border:1px solid #7c3aed44;
                border-radius:16px;padding:1.2rem 1.8rem;margin-bottom:1.2rem;">
        <div style="font-family:'Space Mono',monospace;font-size:1rem;
                    background:linear-gradient(90deg,#a78bfa,#38bdf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:700;margin-bottom:0.3rem;">⚗️ Feature Engineering — Auto + Manual</div>
        <div style="font-size:0.83rem;color:#94a3b8;">
            Automatically generate, transform, combine & select the best features for your model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df       = st.session_state.df_processed
    target   = st.session_state.target_col
    task     = st.session_state.task_type
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    feat_num = [c for c in num_cols if c != target]

    # ─────────────────────────────────────────────
    # SECTION A — AUTO FEATURE GENERATION
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 A. Auto Feature Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ One click — app automatically generates the most useful features based on your data.</div>', unsafe_allow_html=True)

    auto_opts = st.multiselect("Select auto-generation types", [
        "Statistical Features (mean, std, min, max per row)",
        "Interaction Features (col × col)",
        "Ratio Features (col / col)",
        "Difference Features (col - col)",
        "Polynomial Features (degree 2)",
        "Lag Features (shift by 1,2,3)",
        "Rolling Window (mean & std, window=3)",
        "Date/Time Features (if datetime columns exist)",
        "Text Length Features (if text columns exist)",
        "Binning / Bucketing (numeric → categories)",
    ], default=[
        "Statistical Features (mean, std, min, max per row)",
        "Interaction Features (col × col)",
        "Ratio Features (col / col)",
    ], key="auto_feat_opts")

    if st.button("⚗️ Run Auto Feature Generation", key="btn_auto_feat"):
        tmp = st.session_state.df_processed.copy()
        new_features = []

        if "Statistical Features (mean, std, min, max per row)" in auto_opts:
            if len(feat_num) >= 2:
                tmp["feat_row_mean"] = tmp[feat_num].mean(axis=1)
                tmp["feat_row_std"]  = tmp[feat_num].std(axis=1)
                tmp["feat_row_min"]  = tmp[feat_num].min(axis=1)
                tmp["feat_row_max"]  = tmp[feat_num].max(axis=1)
                tmp["feat_row_range"]= tmp[feat_num].max(axis=1) - tmp[feat_num].min(axis=1)
                new_features += ["feat_row_mean","feat_row_std","feat_row_min","feat_row_max","feat_row_range"]

        if "Interaction Features (col × col)" in auto_opts:
            pairs = list(itertools.combinations(feat_num[:6], 2))
            for c1, c2 in pairs[:10]:
                name = f"{c1}_x_{c2}"
                tmp[name] = tmp[c1] * tmp[c2]
                new_features.append(name)

        if "Ratio Features (col / col)" in auto_opts:
            pairs = list(itertools.combinations(feat_num[:6], 2))
            for c1, c2 in pairs[:8]:
                name = f"{c1}_div_{c2}"
                tmp[name] = tmp[c1] / (tmp[c2].replace(0, np.nan) + 1e-8)
                new_features.append(name)

        if "Difference Features (col - col)" in auto_opts:
            pairs = list(itertools.combinations(feat_num[:6], 2))
            for c1, c2 in pairs[:8]:
                name = f"{c1}_minus_{c2}"
                tmp[name] = tmp[c1] - tmp[c2]
                new_features.append(name)

        if "Polynomial Features (degree 2)" in auto_opts:
            poly_cols = feat_num[:5]
            if poly_cols:
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                poly_arr = poly.fit_transform(tmp[poly_cols].fillna(0))
                poly_names = poly.get_feature_names_out(poly_cols)
                # Only add new columns (skip originals)
                new_poly = [n for n in poly_names if n not in tmp.columns]
                poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=tmp.index)
                for n in new_poly:
                    tmp[n] = poly_df[n]
                    new_features.append(n)

        if "Lag Features (shift by 1,2,3)" in auto_opts:
            for col in feat_num[:4]:
                for lag in [1, 2, 3]:
                    name = f"{col}_lag{lag}"
                    tmp[name] = tmp[col].shift(lag)
                    new_features.append(name)
            tmp = tmp.fillna(method="bfill").fillna(0)

        if "Rolling Window (mean & std, window=3)" in auto_opts:
            for col in feat_num[:4]:
                tmp[f"{col}_roll3_mean"] = tmp[col].rolling(3, min_periods=1).mean()
                tmp[f"{col}_roll3_std"]  = tmp[col].rolling(3, min_periods=1).std().fillna(0)
                new_features += [f"{col}_roll3_mean", f"{col}_roll3_std"]

        if "Date/Time Features (if datetime columns exist)" in auto_opts:
            dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            for col in dt_cols:
                tmp[f"{col}_year"]    = tmp[col].dt.year
                tmp[f"{col}_month"]   = tmp[col].dt.month
                tmp[f"{col}_day"]     = tmp[col].dt.day
                tmp[f"{col}_weekday"] = tmp[col].dt.weekday
                tmp[f"{col}_quarter"] = tmp[col].dt.quarter
                tmp[f"{col}_is_weekend"] = (tmp[col].dt.weekday >= 5).astype(int)
                new_features += [f"{col}_year", f"{col}_month", f"{col}_day",
                                  f"{col}_weekday", f"{col}_quarter", f"{col}_is_weekend"]
            # Try parsing object cols as datetime
            for col in cat_cols:
                try:
                    parsed = pd.to_datetime(tmp[col], errors="raise")
                    tmp[f"{col}_year"]    = parsed.dt.year
                    tmp[f"{col}_month"]   = parsed.dt.month
                    tmp[f"{col}_weekday"] = parsed.dt.weekday
                    new_features += [f"{col}_year", f"{col}_month", f"{col}_weekday"]
                except Exception:
                    pass

        if "Text Length Features (if text columns exist)" in auto_opts:
            for col in cat_cols:
                if tmp[col].dtype == object:
                    tmp[f"{col}_len"]       = tmp[col].astype(str).str.len()
                    tmp[f"{col}_wordcount"] = tmp[col].astype(str).str.split().str.len()
                    new_features += [f"{col}_len", f"{col}_wordcount"]

        if "Binning / Bucketing (numeric → categories)" in auto_opts:
            for col in feat_num[:5]:
                name = f"{col}_bin"
                tmp[name] = pd.qcut(tmp[col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
                tmp[name] = tmp[name].astype(str)
                new_features.append(name)

        # Remove duplicate/inf cols
        new_features = [f for f in new_features if f in tmp.columns]
        tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp.fillna(0, inplace=True)

        st.session_state.df_processed = tmp
        st.session_state.preprocessing_log.append(
            f"Auto FE: +{len(new_features)} features generated"
        )
        st.success(f"✅ {len(new_features)} new features created! Dataset: {tmp.shape[0]} × {tmp.shape[1]}")
        with st.expander("📋 New Features Created"):
            for f in new_features:
                st.markdown(f'<span class="tag">✨ {f}</span>', unsafe_allow_html=True)
        st.rerun()

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION B — MANUAL FEATURE CREATION
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">✏️ B. Manual Feature Creation</div>', unsafe_allow_html=True)

    df = st.session_state.df_processed
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feat_num = [c for c in num_cols if c != target]

    manual_type = st.radio("Create feature by:", [
        "Math Formula (custom expression)",
        "Combine 2 Columns",
        "Bin a Column",
        "Group Aggregation",
    ], horizontal=True, key="manual_fe_type")

    if manual_type == "Math Formula (custom expression)":
        st.markdown('<div class="info-box">ℹ️ Use column names directly. Example: <code>col1 * col2 + col3 / 2</code></div>', unsafe_allow_html=True)
        new_col_name = st.text_input("New column name", value="new_feature", key="manual_col_name")
        formula = st.text_input("Formula (use column names as variables)", key="manual_formula",
                                placeholder="e.g.  age * income  or  np.log1p(salary)")
        st.markdown("**Available columns:** " + ", ".join([f"`{c}`" for c in feat_num[:15]]))
        if st.button("➕ Create Feature", key="btn_manual_formula"):
            try:
                tmp = st.session_state.df_processed.copy()
                local_vars = {col: tmp[col] for col in tmp.columns}
                local_vars["np"] = np
                local_vars["pd"] = pd
                tmp[new_col_name] = eval(formula, {"__builtins__": {}}, local_vars)
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Manual FE: {new_col_name} = {formula}")
                st.success(f"✅ Created `{new_col_name}` from formula")
                st.rerun()
            except Exception as e:
                st.error(f"Formula error: {e}")

    elif manual_type == "Combine 2 Columns":
        c1, c2, c3 = st.columns(3)
        with c1: col_a = st.selectbox("Column A", feat_num, key="comb_a")
        with c2:
            op = st.selectbox("Operation", ["+", "-", "×", "÷", "max", "min", "mean", "mod %"], key="comb_op")
        with c3: col_b = st.selectbox("Column B", feat_num, key="comb_b")
        new_name = st.text_input("New column name",
            value=f"{col_a}_{op}_{col_b}".replace("×","x").replace("÷","div"), key="comb_name")
        if st.button("➕ Create Combined Feature", key="btn_comb"):
            tmp = st.session_state.df_processed.copy()
            a, b = tmp[col_a], tmp[col_b]
            if op == "+":           tmp[new_name] = a + b
            elif op == "-":         tmp[new_name] = a - b
            elif op == "×":         tmp[new_name] = a * b
            elif op == "÷":         tmp[new_name] = a / (b.replace(0, np.nan) + 1e-8)
            elif op == "max":       tmp[new_name] = np.maximum(a, b)
            elif op == "min":       tmp[new_name] = np.minimum(a, b)
            elif op == "mean":      tmp[new_name] = (a + b) / 2
            elif op == "mod %":     tmp[new_name] = a % (b.replace(0, np.nan) + 1e-8)
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Manual FE: {new_name} = {col_a} {op} {col_b}")
            st.success(f"✅ Created `{new_name}`")
            st.rerun()

    elif manual_type == "Bin a Column":
        bin_col  = st.selectbox("Column to bin", feat_num, key="bin_col")
        bin_type = st.radio("Bin type", ["Equal Width", "Equal Frequency (quantile)", "Custom boundaries"], horizontal=True, key="bin_type")
        n_bins   = st.slider("Number of bins", 2, 20, 5, key="n_bins")
        bin_name = st.text_input("New column name", value=f"{bin_col}_binned", key="bin_name")
        custom_bounds = None
        if bin_type == "Custom boundaries":
            bounds_str = st.text_input("Enter boundaries (comma separated)", "0,25,50,75,100", key="bin_bounds")
            try:
                custom_bounds = [float(x.strip()) for x in bounds_str.split(",")]
            except:
                st.error("Invalid boundary values")

        if st.button("➕ Create Binned Feature", key="btn_bin"):
            tmp = st.session_state.df_processed.copy()
            try:
                if bin_type == "Equal Width":
                    tmp[bin_name] = pd.cut(tmp[bin_col], bins=n_bins, labels=False)
                elif bin_type == "Equal Frequency (quantile)":
                    tmp[bin_name] = pd.qcut(tmp[bin_col], q=n_bins, labels=False, duplicates="drop")
                elif bin_type == "Custom boundaries" and custom_bounds:
                    tmp[bin_name] = pd.cut(tmp[bin_col], bins=custom_bounds, labels=False)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Binning: {bin_name} from {bin_col} ({bin_type}, {n_bins} bins)")
                st.success(f"✅ Created `{bin_name}`")
                st.rerun()
            except Exception as e:
                st.error(f"Binning error: {e}")

    elif manual_type == "Group Aggregation":
        st.markdown('<div class="info-box">ℹ️ Group by a categorical column and aggregate a numeric column — like SQL GROUP BY.</div>', unsafe_allow_html=True)
        if cat_cols and feat_num:
            g1, g2, g3 = st.columns(3)
            with g1: grp_col  = st.selectbox("Group by (categorical)", cat_cols, key="grp_col")
            with g2: agg_col  = st.selectbox("Aggregate (numeric)", feat_num, key="agg_col")
            with g3: agg_func = st.selectbox("Function", ["mean","median","std","min","max","sum","count","nunique"], key="agg_func")
            new_grp_name = st.text_input("New column name", value=f"{grp_col}_{agg_func}_{agg_col}", key="grp_name")
            if st.button("➕ Create Group Feature", key="btn_grp"):
                tmp = st.session_state.df_processed.copy()
                grp_map = tmp.groupby(grp_col)[agg_col].agg(agg_func)
                tmp[new_grp_name] = tmp[grp_col].map(grp_map)
                st.session_state.df_processed = tmp
                st.session_state.preprocessing_log.append(f"Group FE: {new_grp_name} = {agg_func}({agg_col}) by {grp_col}")
                st.success(f"✅ Created `{new_grp_name}`")
                st.rerun()
        else:
            st.warning("Need at least 1 categorical + 1 numeric column.")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION C — PCA DIMENSIONALITY REDUCTION
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 C. PCA — Dimensionality Reduction</div>', unsafe_allow_html=True)

    df = st.session_state.df_processed
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feat_num = [c for c in num_cols if c != target]

    if len(feat_num) >= 3:
        pca_cols = st.multiselect("Select columns for PCA", feat_num, default=feat_num[:min(8, len(feat_num))], key="pca_cols")
        n_components = st.slider("Number of PCA components", 1, max(1, len(pca_cols)-1 if len(pca_cols) > 1 else 1), min(2, len(pca_cols)), key="pca_n")

        if len(pca_cols) >= 2:
            # Show explained variance preview
            try:
                from sklearn.decomposition import PCA as PCA_
                X_pca = df[pca_cols].dropna()
                pca_preview = PCA_().fit(X_pca)
                exp_var = np.cumsum(pca_preview.explained_variance_ratio_)

                fig, ax = plt.subplots(figsize=(7, 3), facecolor="#1e293b")
                ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#334155")
                ax.plot(range(1, len(exp_var)+1), exp_var*100, color="#38bdf8", marker="o", markersize=4)
                ax.axhline(95, color="#f472b6", linestyle="--", linewidth=1, label="95% variance")
                ax.set_xlabel("Components", color="#94a3b8"); ax.set_ylabel("Cumulative Variance %", color="#94a3b8")
                ax.set_title("PCA Explained Variance", color="#e2e8f0"); ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
                st.pyplot(fig); plt.close()
            except Exception:
                pass

            keep_original = st.checkbox("Keep original columns too", value=False, key="pca_keep")
            if st.button("📉 Apply PCA", key="btn_pca"):
                try:
                    tmp = st.session_state.df_processed.copy()
                    X_pca = tmp[pca_cols].fillna(0)
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(X_pca)
                    pca_df = pd.DataFrame(pca_result,
                                          columns=[f"PCA_{i+1}" for i in range(n_components)],
                                          index=tmp.index)
                    if not keep_original:
                        tmp.drop(columns=pca_cols, inplace=True)
                    tmp = pd.concat([tmp, pca_df], axis=1)
                    st.session_state.df_processed = tmp
                    exp_var_total = sum(pca.explained_variance_ratio_)*100
                    st.session_state.preprocessing_log.append(
                        f"PCA: {len(pca_cols)} cols → {n_components} components ({exp_var_total:.1f}% variance)"
                    )
                    st.success(f"✅ PCA applied: {n_components} components explain {exp_var_total:.1f}% variance")
                    st.rerun()
                except Exception as e:
                    st.error(f"PCA error: {e}")
    else:
        st.info("Need at least 3 numeric columns for PCA.")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION D — AUTO FEATURE SELECTION (importance-based drop)
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 D. Auto Feature Selection & Drop</div>', unsafe_allow_html=True)

    df = st.session_state.df_processed
    if target and target in df.columns:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        feat_num = [c for c in num_cols if c != target]
        valid_idx = df[feat_num + [target]].dropna().index
        X_sel = df.loc[valid_idx, feat_num]
        y_sel = df.loc[valid_idx, target]

        if len(feat_num) >= 2 and len(X_sel) > 10:
            threshold = st.slider("Drop features with importance below (%)", 0, 20, 5, key="fi_thresh") / 100

            if st.button("🔍 Analyse & Auto-Drop Low Importance Features", key="btn_auto_drop"):
                try:
                    if task == "Classification":
                        fi_m = RandomForestClassifier(n_estimators=80, random_state=42)
                    else:
                        fi_m = RandomForestRegressor(n_estimators=80, random_state=42)
                    fi_m.fit(X_sel, y_sel)
                    fi_series = pd.Series(fi_m.feature_importances_, index=feat_num).sort_values(ascending=False)
                    to_drop = fi_series[fi_series < threshold].index.tolist()

                    # Plot
                    fig, ax = plt.subplots(figsize=(9, max(3, len(fi_series)*0.38)), facecolor="#1e293b")
                    ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#334155")
                    colors = ["#f87171" if c in to_drop else "#38bdf8" for c in fi_series.index]
                    ax.barh(fi_series.index, fi_series.values, color=colors)
                    ax.axvline(threshold, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.2f}")
                    ax.set_title("Feature Importances (🔴 = will be dropped)", color="#e2e8f0")
                    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
                    ax.invert_yaxis()
                    st.pyplot(fig); plt.close()

                    if to_drop:
                        st.markdown(f'<div class="warn-box">⚠️ Will drop {len(to_drop)} low-importance features: <b>{to_drop}</b></div>', unsafe_allow_html=True)
                        if st.button(f"🗑️ Confirm Drop {len(to_drop)} Features", key="btn_confirm_drop"):
                            tmp = st.session_state.df_processed.copy()
                            tmp.drop(columns=to_drop, inplace=True)
                            st.session_state.df_processed = tmp
                            st.session_state.preprocessing_log.append(f"Auto-dropped {len(to_drop)} low-importance features: {to_drop}")
                            st.success(f"✅ Dropped {len(to_drop)} features")
                            st.rerun()
                    else:
                        st.markdown('<div class="success-box">✅ All features are above the importance threshold!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Feature importance error: {e}")

            # Correlation-based redundancy drop
            st.markdown("**Remove Highly Correlated Features (redundancy):**")
            corr_thresh = st.slider("Drop one of every pair with correlation above:", 0.70, 0.99, 0.90, 0.01, key="corr_thresh")
            if st.button("🔍 Find & Drop Correlated Features", key="btn_corr_drop"):
                corr_matrix = df[feat_num].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
                if corr_drop:
                    st.markdown(f'<div class="warn-box">⚠️ Highly correlated features to drop: <b>{corr_drop}</b></div>', unsafe_allow_html=True)
                    if st.button(f"🗑️ Confirm Drop {len(corr_drop)} Correlated Features", key="btn_confirm_corr"):
                        tmp = st.session_state.df_processed.copy()
                        tmp.drop(columns=corr_drop, inplace=True)
                        st.session_state.df_processed = tmp
                        st.session_state.preprocessing_log.append(f"Dropped correlated features (>{corr_thresh}): {corr_drop}")
                        st.success(f"✅ Dropped {len(corr_drop)} correlated features")
                        st.rerun()
                else:
                    st.markdown(f'<div class="success-box">✅ No pairs with correlation > {corr_thresh}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠️ Set a target column in the sidebar to enable importance-based selection.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION E — DROP / RENAME COLUMNS
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ E. Drop / Rename Columns</div>', unsafe_allow_html=True)

    df = st.session_state.df_processed
    all_cols = [c for c in df.columns if c != target]

    col_x, col_y = st.columns(2)
    with col_x:
        drop_cols = st.multiselect("Select columns to DROP", all_cols, key="drop_cols_fe")
        if st.button("🗑️ Drop Selected Columns", key="btn_drop_fe", disabled=len(drop_cols)==0):
            tmp = st.session_state.df_processed.copy()
            tmp.drop(columns=drop_cols, inplace=True)
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Dropped columns: {drop_cols}")
            st.success(f"✅ Dropped: {drop_cols}")
            st.rerun()

    with col_y:
        rename_col = st.selectbox("Rename column", all_cols, key="rename_col_fe")
        new_rename  = st.text_input("New name", value=rename_col, key="new_rename_fe")
        if st.button("✏️ Rename", key="btn_rename_fe"):
            tmp = st.session_state.df_processed.copy()
            tmp.rename(columns={rename_col: new_rename}, inplace=True)
            st.session_state.df_processed = tmp
            st.session_state.preprocessing_log.append(f"Renamed: {rename_col} → {new_rename}")
            st.success(f"✅ Renamed `{rename_col}` → `{new_rename}`")
            st.rerun()

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CURRENT DATASET SUMMARY
    # ─────────────────────────────────────────────
    df = st.session_state.df_processed
    st.markdown('<div class="section-header">📋 Current Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="val">{df.shape[0]:,}</div><div class="lbl">Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="val">{df.shape[1]}</div><div class="lbl">Total Columns</div></div>', unsafe_allow_html=True)
    with c3:
        new_feats = len([c for c in df.columns if c.startswith("feat_") or "_x_" in c or "_div_" in c or "_minus_" in c or "PCA_" in c or "_lag" in c or "_roll" in c or "_bin" in c])
        st.markdown(f'<div class="metric-card"><div class="val">{new_feats}</div><div class="lbl">Engineered Features</div></div>', unsafe_allow_html=True)

    st.dataframe(df.head(8), use_container_width=True)

    if st.button("↩️ Reset to Original Data", key="btn_reset_fe"):
        st.session_state.df_processed = st.session_state.df.copy()
        st.session_state.preprocessing_log = []
        st.rerun()
