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

def render_tab_modeling():
    # TAB 7 — MODELING
    # ══════════════════════════════════════════════
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

    # ══════════════════════════════════════════════