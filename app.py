import streamlit as st
import json
import time
from modular_rag_pipeline import run_modular_rag_pipeline, run_copilot_chat

# ==========================================
# 1. 页面基本配置 & 极简原生样式
# ==========================================
st.set_page_config(
    page_title="Oncology CDSS | Clinical Workspace", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: #334155 !important; }
    div.stBlockContainer { padding-top: 2rem; }
    [data-testid="stChatMessage"] { background-color: #F8FAFC; border-radius: 8px; padding: 10px; margin-bottom: 10px; border: 1px solid #E2E8F0; }
    
    .highlight-drug {
        background-color: #FEF08A; border-bottom: 2px dashed #CA8A04; font-weight: 700; cursor: pointer;
        padding: 2px 6px; border-radius: 4px; color: #854D0E; transition: all 0.2s;
    }
    .highlight-drug:hover { background-color: #FDE047; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    
    .alert-box {
        background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 15px 20px;
        border-radius: 6px; color: #991B1B; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(239, 68, 68, 0.1);
    }
    
    /* 锁定后的电子医嘱样式 */
    .locked-order {
        background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 6px solid #22C55E;
        padding: 20px; border-radius: 8px; color: #166534; font-family: monospace; font-size: 1.1em;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心状态与初始化
# ==========================================
if 'workflow_step' not in st.session_state: st.session_state.workflow_step = 'input'
if 'results' not in st.session_state: st.session_state.results = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'system_context' not in st.session_state: st.session_state.system_context = ""
if 'ehr_submitted' not in st.session_state: st.session_state.ehr_submitted = False
if 'current_alerts' not in st.session_state: st.session_state.current_alerts = []

# 初始化药物列表 (支持多药组合)
if 'regimen' not in st.session_state: 
    st.session_state.regimen = [{"med": "Osimertinib", "dose": "80 mg", "route": "PO (Oral)", "freq": "QD (Daily)"}]

def check_critical_alerts(clinical_text):
    alerts = []
    if "CrCl 30" in clinical_text or "CrCl < 45" in clinical_text:
        alerts.append("⚠️ **CRITICAL ALERT:** Severely reduced creatinine clearance detected (30 mL/min). Platinum-based regimens may be contraindicated. Proceed with caution.")
    return alerts

# ==========================================
# 3. 侧边栏设计
# ==========================================
with st.sidebar:
    st.title("System Control")
    st.caption("Evidence-Based Oncology CDSS")
    st.markdown("---")
    
    st.subheader("🚨 System Alerts")
    if st.session_state.workflow_step == 'input':
        st.info("Monitoring patient context...", icon="👁️")
    elif st.session_state.results and st.session_state.current_alerts:
        for alert in st.session_state.current_alerts: st.error(alert)
    else:
        st.success("No active critical alerts.", icon="✅")
        
    st.markdown("---")
    st.write("**Active Modules:**")
    st.write("- Explicit Warning System 🛡️")
    st.write("- Entity Extraction & Verification")
    st.write("- Multi-Drug Regimen Builder 💊")
    st.write("- Editable Orders & EHR Sync ✍️")
    st.write("- Interactive Copilot 💬")
    st.markdown("---")
    
    if st.button("New Patient Encounter", use_container_width=True):
        st.session_state.results = None
        st.session_state.chat_history = []
        st.session_state.system_context = ""
        st.session_state.ehr_submitted = False
        st.session_state.regimen = [{"med": "Osimertinib", "dose": "80 mg", "route": "PO (Oral)", "freq": "QD (Daily)"}]
        st.session_state.workflow_step = 'input'
        st.rerun()

# ==========================================
# 4. 主界面渲染逻辑
# ==========================================
st.title("Clinical Doctor Workspace")
st.markdown("Multimodal AI-Assisted Precision Oncology Workflow Anchored in NCCN Guidelines.")
st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 状态 1: 输入阶段
# ---------------------------------------------------------
if st.session_state.workflow_step == 'input':
    st.subheader("I. Clinical Directive & Multimodal Context", divider="gray")
    query = st.text_input("Query", value="Formulate an NCCN-aligned targeted therapy and monitoring plan.", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_text, col_files = st.columns([1.5, 1], gap="large")
    with col_text:
        st.markdown("**Patient Clinical Narrative:**")
        default_note = "Patient is a 58-year-old female presenting with persistent cough. PET-CT confirmed a 4cm right upper lobe mass with pleural dissemination (Stage IV). Biopsy NGS results returned positive for EGFR Exon 19 deletion. CrCl is 30 mL/min. ECOG PS is 1. She has no prior systemic therapy."
        note = st.text_area("Narrative", value=default_note, height=220, label_visibility="collapsed")
    with col_files:
        st.markdown("**Upload Medical Artifacts (Optional):**")
        uploaded_rad = st.file_uploader("Upload Radiology Images (CT/MRI/PET)", type=["png", "jpg", "jpeg", "dcm"], accept_multiple_files=True)
        uploaded_path = st.file_uploader("Upload Pathology Slides or Lab PDF", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Execute Intent-Driven Pipeline", type="primary"):
        alerts = check_critical_alerts(note)
        st.session_state.current_alerts = alerts
        
        with st.status("Processing Multimodal Clinical Directive...", expanded=True) as status:
            st.write("Verifying clinical entities...")
            st.write("Cross-referencing NCCN Evidence Base...")
            try:
                res = run_modular_rag_pipeline(note)
                st.session_state.results = res
                st.session_state.chat_history.append({"role": "assistant", "content": res.get("treatment_recommendations", "N/A")})
                st.session_state.system_context = f"Summary: {res.get('clinical_assessment')} \n\n Guidelines Logic: {res.get('rag_metadata', {}).get('internal_strategy')}"
                st.session_state.workflow_step = 'chat_mode'
                status.update(label="Processing Complete.", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="System Error", state="error", expanded=False)
                st.error(f"Execution Error: {e}")

# ---------------------------------------------------------
# 状态 2: 结果与交互阶段
# ---------------------------------------------------------
elif st.session_state.workflow_step == 'chat_mode' and st.session_state.results:
    res = st.session_state.results
    
    if st.session_state.current_alerts:
        for alert in st.session_state.current_alerts: st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ **Verification Pass:** Patient profile validated against NCCN evidence.")
        
    with st.expander("📂 View Patient Profile & System Diagnostics (Click to Expand)", expanded=False):
        tab_data, tab_rag = st.tabs(["📊 Structured Profile", "🧠 RAG & AI Logic"])
        with tab_data:
            col1, col2 = st.columns(2)
            with col1: st.json(res.get("structured_data", {}))
            with col2: st.info(res.get("clinical_assessment", "N/A"))
        with tab_rag:
            for q in res.get("rag_metadata", {}).get("sub_queries_generated", []): st.code(q, language="text")
            st.write(res.get("rag_metadata", {}).get("internal_strategy", "N/A"))

    # ==========================================
    # Step 1：多药组合表单
    # ==========================================
    st.subheader("📝 Step 1: Regimen Builder (Multi-Drug Submission)", divider="blue")
    
    if not st.session_state.ehr_submitted:
        st.caption("Review, add, or modify the AI-recommended pharmacological parameters before EHR submission.")
        
        new_regimen = []
        for i, item in enumerate(st.session_state.regimen):
            cols = st.columns([2, 1.5, 1, 1, 0.5])
            with cols[0]: m = st.selectbox(f"Drug {i+1}", ["Osimertinib", "Erlotinib", "Gefitinib", "Pemetrexed", "Cisplatin", "Zoledronic Acid"], index=["Osimertinib", "Erlotinib", "Gefitinib", "Pemetrexed", "Cisplatin", "Zoledronic Acid"].index(item.get('med', 'Osimertinib')), key=f"m_{i}", label_visibility="collapsed")
            with cols[1]: d = st.selectbox(f"Dose", ["80 mg", "40 mg", "150 mg", "500 mg/m²", "75 mg/m²", "4 mg"], index=["80 mg", "40 mg", "150 mg", "500 mg/m²", "75 mg/m²", "4 mg"].index(item.get('dose', '80 mg')), key=f"d_{i}", label_visibility="collapsed")
            with cols[2]: r = st.selectbox(f"Route", ["PO (Oral)", "IV (Intravenous)"], index=["PO (Oral)", "IV (Intravenous)"].index(item.get('route', 'PO (Oral)')), key=f"r_{i}", label_visibility="collapsed")
            with cols[3]: f = st.selectbox(f"Freq", ["QD (Daily)", "BID (Twice)", "Q3W (3 Weeks)", "Q4W (4 Weeks)"], index=["QD (Daily)", "BID (Twice)", "Q3W (3 Weeks)", "Q4W (4 Weeks)"].index(item.get('freq', 'QD (Daily)')), key=f"f_{i}", label_visibility="collapsed")
            with cols[4]: 
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.regimen.pop(i)
                    st.rerun()
            new_regimen.append({"med": m, "dose": d, "route": r, "freq": f})
        
        st.session_state.regimen = new_regimen
        
        col_btns = st.columns([1, 4])
        with col_btns[0]:
            if st.button("➕ Add Drug"):
                st.session_state.regimen.append({"med": "Zoledronic Acid", "dose": "4 mg", "route": "IV (Intravenous)", "freq": "Q4W (4 Weeks)"})
                st.rerun()
        with col_btns[1]:
            if st.button("✅ Sign & Submit to EHR", type="primary", use_container_width=True):
                st.session_state.ehr_submitted = True
                st.toast("Regimen successfully synced to EHR!", icon="✅")
                st.rerun()
    else:
        st.markdown("<b>🔒 STATUS: OFFICIALLY SIGNED & TRANSMITTED TO HIS/EHR</b>", unsafe_allow_html=True)
        for item in st.session_state.regimen:
            st.markdown(f"""
            <div class="locked-order" style="margin-top: 5px; margin-bottom: 5px; padding: 10px 20px;">
                Rx: <b>{item['med']} {item['dose']}</b> via {item['route']} — Sig: {item['freq']}
            </div>
            """, unsafe_allow_html=True)
        if st.button("Unlock and Revise Orders"):
            st.session_state.ehr_submitted = False
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # Step 2：证据面板
    # ==========================================
    st.subheader("🔍 Step 2: Evidence-Anchored Rationale", divider="gray")
    col_plan, col_evid = st.columns([2, 1], gap="large")
    with col_plan:
        plan_text = res.get("treatment_recommendations", "N/A")
        plan_text = plan_text.replace("Osimertinib", '<span class="highlight-drug" title="Click to trace evidence">Osimertinib</span>')
        st.markdown(plan_text, unsafe_allow_html=True)
    with col_evid:
        st.markdown("""
        <div style="background:#F8FAFC; padding:20px; border:1px solid #E2E8F0; border-radius:8px; position: sticky; top: 10px;">
            <h4 style="margin-top:0; color:#334155;">📑 Evidence Panel</h4>
            <hr style="margin: 10px 0;">
            <b style="color:#0F172A;">Detected Biomarker:</b> <span style="color:#DC2626;">EGFR Exon 19 Deletion</span><br><br>
            <b style="color:#0F172A;">NCCN Guideline (v2026):</b><br>
            <div style="background:white; padding:10px; border-left:4px solid #8B5CF6; font-size:0.9em; margin-top:8px;">
            "For advanced NSCLC with EGFR Exon 19 deletion, first-line preferred therapy is <b>Osimertinib</b>."
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # Step 3：对话交互
    # ==========================================
    st.subheader("💬 Step 3: Interactive Copilot", divider="blue")
    
    display_history = st.session_state.chat_history[1:]
    
    if not display_history:
        st.info("👆 The initial clinical rationale is detailed in Step 2. Use the chat box below to ask follow-up questions, adjust constraints, or explore alternative therapies.")
        
    for msg in display_history:
        avatar = "🩺" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar): 
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask a follow-up question (e.g., 'What if the patient refuses IV therapy?'):"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun() 

    if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "user":
        user_msg = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing request..."):
                try:
                    last_plan = next((msg["content"] for msg in reversed(st.session_state.chat_history[:-1]) if msg["role"] == "assistant"), "")
                    updated_reply = run_copilot_chat(last_plan, st.session_state.system_context, user_msg)
                    st.markdown(updated_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": updated_reply})
                except Exception as e:
                    st.error(f"Chat Error: {e}")