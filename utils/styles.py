import streamlit as st

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0d0f1a; }
.stApp { background: linear-gradient(135deg, #0d0f1a 0%, #111827 100%); color: #e2e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #38bdf8 !important; letter-spacing: -0.02em; }

.hero-title {
    font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;
}
.hero-sub { font-size: 1.1rem; color: #94a3b8; margin-top: 0.5rem; }

.metric-card {
    background: linear-gradient(145deg, #1e293b, #162032); border: 1px solid #1e40af33;
    border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center; box-shadow: 0 4px 24px #0003;
}
.metric-card .val { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

.section-header {
    font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #818cf8;
    border-left: 3px solid #818cf8; padding-left: 0.8rem; margin: 1.5rem 0 1rem;
}

.tag { display: inline-block; background: #1e3a5f; color: #7dd3fc; border-radius: 20px;
       padding: 2px 12px; font-size: 0.78rem; font-family: 'Space Mono', monospace; margin: 2px; }
.tag-cat { background: #2d1b4e; color: #c4b5fd; }
.tag-warn { background: #422006; color: #fbbf24; }

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    letter-spacing: 0.03em !important; padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important; box-shadow: 0 4px 14px #1d4ed844 !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px #1d4ed866 !important; }
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #1e293b !important; border: 1px solid #334155 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b) !important; border-right: 1px solid #1e40af33 !important; }

.stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important; color: white !important; }

.info-box { background: #0f2744; border: 1px solid #1e40af55; border-radius: 10px; padding: 0.9rem 1.2rem; color: #bae6fd; font-size: 0.9rem; margin: 0.5rem 0; }
.success-box { background: #052e16; border: 1px solid #15803d55; border-radius: 10px; padding: 0.9rem 1.2rem; color: #86efac; font-size: 0.9rem; margin: 0.5rem 0; }
.warn-box { background: #1c1407; border: 1px solid #b4530055; border-radius: 10px; padding: 0.9rem 1.2rem; color: #fcd34d; font-size: 0.9rem; margin: 0.5rem 0; }
.divider { border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0; }

/* ── WORKFLOW ── */
.wf-wrap { display:flex; align-items:stretch; gap:0; width:100%; overflow-x:auto; padding:1.4rem 0 1rem; }
.wf-node { display:flex; flex-direction:column; align-items:center; min-width:110px; flex:1; position:relative; animation:wf-fadein 0.5s ease both; }
@keyframes wf-fadein { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.wf-circle { width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.5rem;position:relative;z-index:2;border:2px solid transparent;transition:transform 0.3s ease,box-shadow 0.3s ease; }
.wf-node.done .wf-circle { background:linear-gradient(135deg,#0f4c2a,#166534);border-color:#22c55e;box-shadow:0 0 18px #22c55e55;animation:wf-pulse-done 2.5s ease infinite; }
.wf-node.active .wf-circle { background:linear-gradient(135deg,#1e3a8a,#3730a3);border-color:#60a5fa;box-shadow:0 0 22px #3b82f677;animation:wf-pulse-active 1.4s ease infinite; }
.wf-node.pending .wf-circle { background:#1e293b;border-color:#334155;opacity:0.55; }
@keyframes wf-pulse-done { 0%,100%{box-shadow:0 0 16px #22c55e44}50%{box-shadow:0 0 28px #22c55e88} }
@keyframes wf-pulse-active { 0%,100%{box-shadow:0 0 18px #3b82f655}50%{box-shadow:0 0 34px #3b82f699;transform:scale(1.06)} }
.wf-lbl { margin-top:0.55rem;font-family:'Space Mono',monospace;font-size:0.68rem;text-align:center;line-height:1.3;max-width:100px; }
.wf-node.done .wf-lbl{color:#4ade80} .wf-node.active .wf-lbl{color:#93c5fd} .wf-node.pending .wf-lbl{color:#475569}
.wf-badge { font-size:0.6rem;margin-top:0.25rem;padding:1px 8px;border-radius:20px;font-family:'DM Sans',sans-serif; }
.wf-node.done .wf-badge{background:#14532d;color:#86efac} .wf-node.active .wf-badge{background:#1e3a8a;color:#bfdbfe} .wf-node.pending .wf-badge{background:#1e293b;color:#475569}
.wf-connector { flex:1;height:2px;margin-top:28px;border-radius:1px;transition:all .4s;min-width:10px; }
.wf-connector.pending{background:#1e293b} .wf-connector.done{background:#22c55e;box-shadow:0 0 4px #22c55e55}
.wf-connector.active{background:linear-gradient(90deg,#22c55e,#3b82f6);animation:wf-flow 1.2s linear infinite;background-size:200% 100%}
@keyframes wf-flow{0%{background-position:100% 0}100%{background-position:-100% 0}}
.wf-detail-strip{display:flex;gap:0.5rem;flex-wrap:wrap;background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:0.7rem 1rem;margin:0.4rem 0 1.2rem;font-size:0.8rem;color:#64748b;}
.wf-chip{background:#1e293b;color:#94a3b8;border-radius:20px;padding:2px 10px;font-family:'Space Mono',monospace;font-size:0.7rem;border:1px solid #334155;}
.wf-chip.done{background:#14532d;color:#86efac;border-color:#166534}
.wf-chip.active{background:#1e3a8a;color:#bfdbfe;border-color:#1d4ed8;animation:wf-chip-blink 1.5s ease infinite}
@keyframes wf-chip-blink{0%,100%{opacity:1}50%{opacity:0.65}}

/* sidebar mini pipeline */
.sb-pipeline{display:flex;flex-direction:column;gap:6px;padding:0.5rem 0;}
.sb-step{display:flex;align-items:center;gap:8px;font-size:0.78rem;font-family:'DM Sans',sans-serif;padding:5px 8px;border-radius:8px;border:1px solid transparent;transition:all 0.3s;}
.sb-step.done{background:#052e16;border-color:#166534;color:#4ade80}
.sb-step.active{background:#1e3a8a22;border-color:#3b82f6;color:#93c5fd;animation:wf-chip-blink 1.5s ease infinite}
.sb-step.pending{color:#475569}
.sb-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.sb-step.done .sb-dot{background:#22c55e;box-shadow:0 0 6px #22c55e}
.sb-step.active .sb-dot{background:#3b82f6;box-shadow:0 0 6px #3b82f6}
.sb-step.pending .sb-dot{background:#334155}
</style>
""", unsafe_allow_html=True)
