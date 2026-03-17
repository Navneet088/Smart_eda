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

def render_tab_automl():
    # TAB 8 — AUTO SUGGEST & PREDICT
    # ══════════════════════════════════════════════
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