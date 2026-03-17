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

def render_tab_llm():
    # TAB 9 — AI DATA ANALYST (LLM)
    # ══════════════════════════════════════════════

    # ── init chat history
    if "llm_chat_history" not in st.session_state or st.session_state.llm_chat_history is None:
        st.session_state.llm_chat_history = []

    # ── helper: build rich data context for LLM
    def build_data_context():
        df_c = st.session_state.df_processed
        if df_c is None:
            return "No dataset loaded yet."
        target = st.session_state.target_col
        task   = st.session_state.task_type
        log    = st.session_state.preprocessing_log or []
        num_c  = df_c.select_dtypes(include=np.number).columns.tolist()
        cat_c  = df_c.select_dtypes(include="object").columns.tolist()
        desc   = df_c.describe(include="all").round(3).to_string()
        miss   = df_c.isnull().sum()
        corr_str = ""
        if len(num_c) >= 2:
            corr   = df_c[num_c].corr().round(3)
            top_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    top_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
            top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            corr_str = "\n".join([f"  {a} ↔ {b}: {r:.3f}" for a,b,r in top_pairs[:10]])
        skew_info = "\n".join([f"  {c}: skew={round(df_c[c].skew(),3)}" for c in num_c[:10]])
        ctx = f"""
    === DATASET CONTEXT ===
    Shape: {df_c.shape[0]} rows × {df_c.shape[1]} columns
    Task type: {task or 'Not set'}
    Target column: {target or 'Not set'}
    Numeric columns: {num_c}
    Categorical columns: {cat_c}
    Missing values: {miss[miss>0].to_dict() or 'None'}
    Preprocessing steps applied: {log or 'None'}

    === STATISTICAL SUMMARY ===
    {desc}

    === SKEWNESS (numeric cols) ===
    {skew_info}

    === TOP CORRELATIONS ===
    {corr_str or 'Not enough numeric columns'}

    === SAMPLE DATA (first 5 rows) ===
    {df_c.head(5).to_string()}
    """
        return ctx.strip()

    # ── helper: call LLM
    def call_llm(provider, api_key, model_name, messages, data_ctx):
        system_prompt = f"""You are an expert Data Scientist and ML Engineer acting as an interactive AI Data Analyst.
    You have full access to the user's dataset. Your job is to:
    1. Analyze the data deeply and answer questions about it
    2. Suggest the best preprocessing steps, models, and hyperparameters
    3. Explain patterns, correlations, outliers, and distributions
    4. Write Python/pandas code snippets when useful
    5. Give actionable, specific recommendations — not generic advice
    6. Be conversational, clear, and precise

    CURRENT DATASET CONTEXT:
    {data_ctx}

    Always reference specific column names, values, and statistics from the dataset in your answers.
    Format code blocks with ```python ... ```.
    """
        try:
            if provider == "Claude (Anthropic)":
                import anthropic as ac
                client = ac.Anthropic(api_key=api_key)
                claude_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=claude_msgs
                )
                return response.content[0].text

            elif provider == "GPT (OpenAI)":
                import openai as oa
                client = oa.OpenAI(api_key=api_key)
                oai_msgs = [{"role": "system", "content": system_prompt}]
                oai_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=oai_msgs,
                    max_tokens=2048,
                    temperature=0.3
                )
                return response.choices[0].message.content

            elif provider == "Gemini (Google)":
                import google.generativeai as genai_sdk
                genai_sdk.configure(api_key=api_key)
                gm = genai_sdk.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                history_gemini = []
                for m in messages[:-1]:
                    role = "user" if m["role"] == "user" else "model"
                    history_gemini.append({"role": role, "parts": [m["content"]]})
                chat = gm.start_chat(history=history_gemini)
                response = chat.send_message(messages[-1]["content"])
                return response.text

            elif provider == "Ollama (Local)":
                import requests as req
                payload = {
                    "model": model_name,
                    "messages": [{"role": "system", "content": system_prompt}] +
                                [{"role": m["role"], "content": m["content"]} for m in messages],
                    "stream": False
                }
                resp = req.post("http://localhost:11434/api/chat", json=payload, timeout=120)
                return resp.json()["message"]["content"]

            elif provider == "Groq":
                from groq import Groq
                client = Groq(api_key=api_key)
                groq_msgs = [{"role": "system", "content": system_prompt}]
                groq_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=groq_msgs,
                    max_tokens=2048,
                    temperature=0.3
                )
                return response.choices[0].message.content

            elif provider == "HuggingFace (Inference API)":
                import requests as req
                headers = {"Authorization": f"Bearer {api_key}"}
                full_prompt = system_prompt + "\n\n"
                for m in messages:
                    full_prompt += f"{'Human' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
                full_prompt += "Assistant:"
                payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 1024}}
                resp = req.post(
                    f"https://api-inference.huggingface.co/models/{model_name}",
                    headers=headers, json=payload, timeout=60
                )
                result = resp.json()
                if isinstance(result, list):
                    return result[0].get("generated_text", str(result))
                return str(result)

        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    # ─────────────────────────────────
    # UI HEADER
    # ─────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#1a0a2e);border:1px solid #7c3aed44;
                border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                    background:linear-gradient(90deg,#a78bfa,#f472b6,#38bdf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:700;margin-bottom:0.4rem;">
            🤖 AI Data Analyst
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;">
            Choose any LLM provider → paste API key → ask anything about your data.
            The AI has full context of your dataset, statistics, correlations & preprocessing steps.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────
    # PROVIDER CONFIG (collapsible)
    # ─────────────────────────────────
    with st.expander("⚙️ LLM Provider Configuration", expanded=st.session_state.llm_provider is None):

        PROVIDERS = {
            "Claude (Anthropic)": {
                "models": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
                "key_label": "Anthropic API Key",
                "key_url": "https://console.anthropic.com/",
                "needs_key": True,
                "color": "#f472b6"
            },
            "GPT (OpenAI)": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "key_label": "OpenAI API Key",
                "key_url": "https://platform.openai.com/api-keys",
                "needs_key": True,
                "color": "#4ade80"
            },
            "Gemini (Google)": {
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
                "key_label": "Google AI Studio Key",
                "key_url": "https://aistudio.google.com/",
                "needs_key": True,
                "color": "#38bdf8"
            },
            "Groq": {
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
                "key_label": "Groq API Key",
                "key_url": "https://console.groq.com/",
                "needs_key": True,
                "color": "#fbbf24"
            },
            "HuggingFace (Inference API)": {
                "models": ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta", "tiiuae/falcon-7b-instruct"],
                "key_label": "HuggingFace Token",
                "key_url": "https://huggingface.co/settings/tokens",
                "needs_key": True,
                "color": "#a78bfa"
            },
            "Ollama (Local)": {
                "models": ["llama3", "mistral", "codellama", "gemma2", "phi3", "llava"],
                "key_label": "No key needed (runs locally)",
                "key_url": "https://ollama.ai/",
                "needs_key": False,
                "color": "#34d399"
            },
        }

        # Provider pill selector
        st.markdown("**Select Provider:**")
        p_cols = st.columns(len(PROVIDERS))
        for i, (pname, pinfo) in enumerate(PROVIDERS.items()):
            with p_cols[i]:
                selected_mark = "✓ " if st.session_state.llm_provider == pname else ""
                if st.button(f"{selected_mark}{pname.split(' ')[0]}", key=f"prov_{i}",
                             use_container_width=True):
                    st.session_state.llm_provider = pname
                    st.session_state.llm_model_name = PROVIDERS[pname]["models"][0]
                    st.rerun()

        if st.session_state.llm_provider:
            pinfo = PROVIDERS[st.session_state.llm_provider]
            st.markdown(f"**Selected:** `{st.session_state.llm_provider}`")

            # Model picker
            model_choice = st.selectbox(
                "Model",
                pinfo["models"],
                index=0,
                key="llm_model_sel"
            )
            # Allow custom model name
            custom_model = st.text_input(
                "Or enter custom model name (leave blank to use above)",
                value="",
                key="llm_custom_model"
            )
            st.session_state.llm_model_name = custom_model.strip() if custom_model.strip() else model_choice

            # API Key
            if pinfo["needs_key"]:
                api_key_input = st.text_input(
                    pinfo["key_label"],
                    type="password",
                    placeholder="Paste your API key here...",
                    key="llm_api_input"
                )
                if api_key_input:
                    st.session_state.llm_api_key = api_key_input
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#64748b;">Get key: <a href="{pinfo["key_url"]}" target="_blank" style="color:#38bdf8;">{pinfo["key_url"]}</a></div>',
                    unsafe_allow_html=True
                )
            else:
                st.session_state.llm_api_key = "local"
                st.markdown('<div class="success-box">✅ Ollama runs locally — no API key needed. Make sure Ollama is running on port 11434.</div>', unsafe_allow_html=True)

            # Install hint
            install_map = {
                "Claude (Anthropic)": "pip install anthropic",
                "GPT (OpenAI)": "pip install openai",
                "Gemini (Google)": "pip install google-generativeai",
                "Groq": "pip install groq",
                "HuggingFace (Inference API)": "pip install requests",
                "Ollama (Local)": "pip install requests",
            }
            st.markdown(f'<div style="font-size:0.75rem;color:#64748b;margin-top:6px;">Install: <code>{install_map[st.session_state.llm_provider]}</code></div>', unsafe_allow_html=True)

            if st.button("💾 Save Config", key="save_llm_config"):
                st.success(f"✅ Config saved: {st.session_state.llm_provider} / {st.session_state.llm_model_name}")

    # ─────────────────────────────────
    # DATA CONTEXT PREVIEW
    # ─────────────────────────────────
    with st.expander("🗃️ Data Context sent to LLM (click to preview)"):
        ctx_preview = build_data_context()
        st.code(ctx_preview[:3000] + ("\n... [truncated]" if len(ctx_preview) > 3000 else ""), language="text")

    # ─────────────────────────────────
    # QUICK PROMPT BUTTONS
    # ─────────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Analysis Prompts</div>', unsafe_allow_html=True)
    quick_prompts = [
        ("📊 Full EDA Summary", "Give me a complete EDA summary of this dataset. Include key statistics, distributions, outliers, correlations, and any data quality issues you notice."),
        ("🧹 Preprocessing Plan", "Based on the dataset, suggest the best step-by-step preprocessing pipeline. Include which columns need scaling, encoding, outlier treatment, and why."),
        ("🤖 Best Model", "Which ML model would work best for this dataset and why? Consider dataset size, feature types, task type, and any class imbalance."),
        ("🔍 Feature Analysis", "Analyze the feature importance and correlations. Which features are most predictive? Are there any redundant or irrelevant features I should drop?"),
        ("⚠️ Data Issues", "What are the main data quality issues in this dataset? Missing values, outliers, skewness, multicollinearity — give me a prioritized action list."),
        ("💡 Insights", "What are the top 5 most interesting insights or patterns you can find in this dataset? Be specific with column names and values."),
        ("🐍 Code", "Write Python pandas code to fully preprocess this dataset — handle missing values, encode categoricals, scale numerics, and split into train/test."),
        ("📈 Model Improvement", "How can I improve my model's accuracy for this dataset? Suggest hyperparameter tuning, feature engineering, and ensembling strategies."),
    ]

    qp_cols = st.columns(4)
    for i, (label, prompt) in enumerate(quick_prompts):
        with qp_cols[i % 4]:
            if st.button(label, key=f"qp_{i}", use_container_width=True):
                st.session_state.llm_chat_history.append({"role": "user", "content": prompt})
                st.session_state.llm_pending = True
                st.rerun()

    # ─────────────────────────────────
    # CHAT INTERFACE
    # ─────────────────────────────────
    st.markdown('<div class="section-header">💬 Chat with AI Analyst</div>', unsafe_allow_html=True)

    # Render chat history
    chat_history = st.session_state.llm_chat_history or []
    for msg in chat_history:
        is_user = msg["role"] == "user"
        bg      = "#1e3a8a22" if is_user else "#14532d22"
        border  = "#3b82f655" if is_user else "#22c55e55"
        icon    = "🧑‍💻" if is_user else "🤖"
        label   = "You" if is_user else f"AI ({st.session_state.llm_model_name or 'LLM'})"
        content = msg["content"]

        # Render code blocks nicely
        parts = re.split(r"(```[\s\S]*?```)", content)
        rendered = ""
        for part in parts:
            if part.startswith("```"):
                lang_match = re.match(r"```(\w+)?\n?([\s\S]*?)```", part)
                code_content = lang_match.group(2) if lang_match else part[3:-3]
                rendered += f'<pre style="background:#0d0f1a;border:1px solid #334155;border-radius:8px;padding:0.8rem;font-size:0.78rem;color:#e2e8f0;overflow-x:auto;margin:6px 0">{code_content}</pre>'
            else:
                escaped = part.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                rendered += f'<span style="font-size:0.85rem">{escaped}</span>'

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:12px;
                    padding:1rem 1.2rem;margin:0.5rem 0;">
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                        color:{'#93c5fd' if is_user else '#4ade80'};margin-bottom:0.5rem;">
                {icon} {label}
            </div>
            <div style="color:#e2e8f0;line-height:1.6">{rendered}</div>
        </div>
        """, unsafe_allow_html=True)

    # Auto-call LLM if pending
    if st.session_state.get("llm_pending") and chat_history:
        if not st.session_state.llm_provider:
            st.markdown('<div class="warn-box">⚠️ Please select an LLM provider and enter API key above.</div>', unsafe_allow_html=True)
            st.session_state.llm_pending = False
        elif st.session_state.llm_provider != "Ollama (Local)" and not st.session_state.llm_api_key:
            st.markdown('<div class="warn-box">⚠️ Please enter your API key in the configuration above.</div>', unsafe_allow_html=True)
            st.session_state.llm_pending = False
        else:
            with st.spinner(f"🤖 {st.session_state.llm_provider} is analyzing your data..."):
                data_ctx = build_data_context()
                response = call_llm(
                    provider   = st.session_state.llm_provider,
                    api_key    = st.session_state.llm_api_key,
                    model_name = st.session_state.llm_model_name,
                    messages   = chat_history,
                    data_ctx   = data_ctx
                )
                st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
                st.session_state.llm_pending = False
                st.rerun()

    # Message input
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input(
        "Ask anything about your data... (e.g. 'Which features have highest correlation?')",
        key="llm_chat_input"
    )
    if user_input:
        st.session_state.llm_chat_history.append({"role": "user", "content": user_input})
        st.session_state.llm_pending = True
        st.rerun()

    # Chat controls
    c_a, c_b, c_c = st.columns([2, 1, 1])
    with c_b:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.llm_chat_history = []
            st.session_state.llm_pending = False
            st.rerun()
    with c_c:
        if chat_history:
            chat_text = "\n\n".join([f"{'USER' if m['role']=='user' else 'AI'}: {m['content']}" for m in chat_history])
            st.download_button("⬇️ Export Chat", chat_text.encode(), "ai_analysis.txt", "text/plain", key="dl_chat")

    # ─────────────────────────────────
    # AUTO FULL REPORT
    # ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📄 Generate Full AI Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ Ek click mein LLM poore dataset ka comprehensive report generate karega — EDA, issues, recommendations, model suggestions sab kuch.</div>', unsafe_allow_html=True)

    if st.button("📊 Generate Full AI Data Report", key="btn_full_report"):
        full_report_prompt = """Generate a comprehensive Data Analysis Report for this dataset with the following sections:

    ## 1. Executive Summary
    Brief overview of the dataset and key findings.

    ## 2. Data Quality Assessment
    - Missing values analysis
    - Outlier detection
    - Data type issues
    - Duplicate rows

    ## 3. Statistical Insights
    - Key distributions (normal/skewed)
    - Important correlations
    - Surprising patterns or anomalies

    ## 4. Feature Analysis
    - Most important features
    - Features to drop or engineer
    - Multicollinearity issues

    ## 5. Preprocessing Recommendations
    Step-by-step pipeline with reasons for each step.

    ## 6. Model Recommendations
    - Best 3 models for this task with reasons
    - Expected performance range
    - Key hyperparameters to tune

    ## 7. Action Items (Priority Order)
    Numbered list of what to do next.

    Be specific — use actual column names and statistics from the dataset."""

        st.session_state.llm_chat_history.append({"role": "user", "content": full_report_prompt})
        st.session_state.llm_pending = True
        st.rerun()

