import streamlit as st

def init_state():
    keys = [
        "df", "df_processed", "target_col", "task_type", "preprocessing_log",
        "auto_analysed", "auto_findings", "auto_stats", "auto_Xauto", "auto_yauto",
        "sug_models", "sug_scaler", "sug_features", "auto_results", "best_auto_model",
        "auto_Xtest", "auto_ytest",
        "llm_provider", "llm_api_key", "llm_chat_history", "llm_model_name",
        "llm_pending",
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

    if not st.session_state.preprocessing_log:
        st.session_state.preprocessing_log = []
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "llm_chat_history" not in st.session_state or st.session_state.llm_chat_history is None:
        st.session_state.llm_chat_history = []
