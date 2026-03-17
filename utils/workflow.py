import streamlit as st

def get_pipeline_status():
    df       = st.session_state.df
    dfp      = st.session_state.df_processed
    target   = st.session_state.target_col
    log      = st.session_state.preprocessing_log or []
    has_data    = df is not None
    has_preproc = has_data and any(
        s for s in log
        if "[Missing]" not in s and "Auto-fill" not in s
        and "Removed duplicate" not in s and "Feature selection" not in s
    )
    has_featsel = has_data and any("Feature selection" in s for s in log)
    models_done = st.session_state.models_trained
    if has_data:
        nulls_remaining = dfp.isnull().sum().sum() if dfp is not None else 0
        miss_done = any("[Missing]" in s or "Auto-fill" in s for s in log) or nulls_remaining == 0
    else:
        miss_done = False

    return [
        {"id":"upload",  "icon":"📂", "label":"Data\nUpload",      "done": has_data,    "active": not has_data},
        {"id":"overview","icon":"📋", "label":"Overview\n& EDA",    "done": has_data,    "active": False},
        {"id":"missing", "icon":"🧹", "label":"Missing\nValues",    "done": miss_done,   "active": has_data and not miss_done},
        {"id":"preproc", "icon":"⚙️",  "label":"Preprocess\ning",   "done": has_preproc, "active": miss_done and not has_preproc},
        {"id":"featsel", "icon":"🎯", "label":"Feature\nSelection", "done": has_featsel, "active": has_preproc and not has_featsel},
        {"id":"model",   "icon":"🤖", "label":"Modeling\n& Eval",   "done": models_done, "active": has_featsel and not models_done},
        {"id":"done",    "icon":"🏆", "label":"Pipeline\nComplete", "done": models_done, "active": False},
    ]


def render_workflow_bar():
    steps = get_pipeline_status()
    nodes_html = ""
    for i, s in enumerate(steps):
        state = "done" if s["done"] else ("active" if s["active"] else "pending")
        badge = "✓ Done" if s["done"] else ("● Active" if s["active"] else "○ Pending")
        nodes_html += f"""
        <div class="wf-node {state}" style="animation-delay:{i*0.09}s">
            <div class="wf-circle">{s["icon"]}</div>
            <div class="wf-label">{s["label"].replace(chr(10),"<br>")}</div>
            <div class="wf-badge">{badge}</div>
        </div>"""
        if i < len(steps) - 1:
            conn = "done" if s["done"] else ("active" if s["active"] else "pending")
            nodes_html += f'<div class="wf-connector {conn}"></div>'

    log   = st.session_state.preprocessing_log or []
    chips = "".join(f'<span class="wf-chip done">{e}</span>' for e in log[-8:]) \
            if log else '<span class="wf-chip">No steps yet</span>'

    return f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #1e40af33;
                border-radius:16px;padding:1.2rem 1.6rem;box-shadow:0 4px 32px #0006;margin-bottom:1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">⚡ Live Pipeline Status</div>
        <div class="wf-wrap">{nodes_html}</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#475569;margin:0.4rem 0;">APPLIED STEPS →</div>
        <div class="wf-detail-strip">{chips}</div>
    </div>"""


def render_sidebar_pipeline():
    steps = get_pipeline_status()
    rows = "".join(
        f'<div class="sb-step {"done" if s["done"] else ("active" if s["active"] else "pending")}">'
        f'<div class="sb-dot"></div>{s["icon"]} {s["label"].replace(chr(10)," ")}</div>'
        for s in steps
    )
    return f'<div class="sb-pipeline">{rows}</div>'
