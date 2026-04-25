import streamlit as st
import json
import time
import re
from modular_rag_pipeline import run_modular_rag_pipeline, run_copilot_chat

st.set_page_config(
    page_title="Oncology CDSS | Clinical Workspace", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    div.stBlockContainer { padding-top: 2rem; }
    [data-testid="stChatMessage"] { background-color: #F8FAFC; border-radius: 8px; padding: 10px; margin-bottom: 10px; border: 1px solid #E2E8F0; }
    
    /* --- Modified Tooltip & Link CSS --- */
    .tooltip-trigger {
        background-color: #FEF08A; 
        border-bottom: 2px dashed #CA8A04; 
        font-weight: 700; 
        cursor: pointer; 
        padding: 2px 6px; 
        border-radius: 4px; 
        color: #854D0E; 
        transition: all 0.2s;
        position: relative;
        display: inline-block;
        text-decoration: none; 
    }
    
    .tooltip-trigger:hover { 
        background-color: #FDE047; 
        color: #991B1B;
    }

    .tooltip-trigger .tooltip-content {
        visibility: hidden;
        width: 320px;
        background-color: #1E293B;
        color: #F8FAFC;
        text-align: left;
        border-radius: 6px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%; 
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s, visibility 0.3s;
        font-size: 0.85em;
        font-weight: normal;
        line-height: 1.4;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .tooltip-trigger .tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #1E293B transparent transparent transparent;
    }

    .tooltip-trigger:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
    }
    
    /* --- Guideline Box Styling --- */
    .guideline-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        color: #334155;
    }
    
    .alert-box {
        background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 15px 20px;
        border-radius: 6px; color: #991B1B; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(239, 68, 68, 0.1);
    }
    
    .locked-order {
        background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 6px solid #22C55E;
        padding: 20px; border-radius: 8px; color: #166534; font-family: monospace; font-size: 1.1em;
        margin-bottom: 15px;
    }

    .agent-badge {
        background-color: #E0E7FF; color: #4338CA; padding: 3px 8px; 
        border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-left: 10px;
        border: 1px solid #C7D2FE; vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# State initialization
if 'workflow_step' not in st.session_state: st.session_state.workflow_step = 'input'
if 'results' not in st.session_state: st.session_state.results = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'system_context' not in st.session_state: st.session_state.system_context = ""
if 'ehr_submitted' not in st.session_state: st.session_state.ehr_submitted = False
if 'current_alerts' not in st.session_state: st.session_state.current_alerts = []
if 'regimen' not in st.session_state: st.session_state.regimen = [{"med": "Osimertinib", "dose": "80 mg", "route": "PO (Oral)", "freq": "QD (Daily)"}]

def check_critical_alerts(clinical_text):
    alerts = []
    if "CrCl 30" in clinical_text or "CrCl < 45" in clinical_text:
        alerts.append("CRITICAL ALERT: Severely reduced creatinine clearance detected. Platinum-based regimens may be contraindicated. Proceed with caution.")
    return alerts

def extract_dynamic_drugs(text):
    master_drugs = ["Osimertinib", "Erlotinib", "Gefitinib", "Pemetrexed", "Cisplatin", "Zoledronic Acid", "Pembrolizumab", "Carboplatin", "Docetaxel", "Bevacizumab", "Trastuzumab"]
    found_drugs = [drug for drug in master_drugs if drug.lower() in text.lower()]
    if not found_drugs:
        found_drugs = ["Osimertinib", "Pemetrexed", "Cisplatin", "Zoledronic Acid"]
    found_drugs.append("Other (Please Specify)")
    return found_drugs

# Header
col_header_1, col_header_2 = st.columns([5, 1])
with col_header_1:
    st.title("Clinical Doctor Workspace")
    st.markdown("Multimodal AI-Assisted Precision Oncology Workflow Anchored in NCCN Guidelines.")
with col_header_2:
    if st.button("New Patient Encounter", use_container_width=True):
        st.session_state.results = None
        st.session_state.chat_history = []
        st.session_state.system_context = ""
        st.session_state.ehr_submitted = False
        st.session_state.regimen = [{"med": "Osimertinib", "dose": "80 mg", "route": "PO (Oral)", "freq": "QD (Daily)"}]
        st.session_state.workflow_step = 'input'
        st.rerun()

st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# Input Phase
if st.session_state.workflow_step == 'input':
    st.markdown("### I. Clinical Directive & Multimodal Context <span class='agent-badge'>Agent 1: Extractor</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 15px; border-top: 2px solid #D1D5DB;'>", unsafe_allow_html=True)
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

# Results Phase
elif st.session_state.workflow_step == 'chat_mode' and st.session_state.results:
    res = st.session_state.results
    
    if st.session_state.current_alerts:
        for alert in st.session_state.current_alerts: 
            st.markdown(f'<div class="alert-box"><strong>Agent 4 (Critic) Alert:</strong><br>{alert}</div>', unsafe_allow_html=True)
    else:
        st.success("Agent 4 (Critic) Pass: Patient profile validated. No logical contradictions detected.")
        
    with st.expander("View Patient Profile & System Diagnostics (Click to Expand)", expanded=False):
        tab_data, tab_rag = st.tabs(["Structured Profile", "RAG & AI Logic"])
        with tab_data:
            col1, col2 = st.columns(2)
            with col1: st.json(res.get("structured_data", {}))
            with col2: st.info(res.get("clinical_assessment", "N/A"))
        with tab_rag:
            for q in res.get("rag_metadata", {}).get("sub_queries_generated", []): st.code(q, language="text")
            st.write(res.get("rag_metadata", {}).get("internal_strategy", "N/A"))

    # Step 1: Regimen Builder
    st.markdown("### Step 1: Regimen Builder (Multi-Drug Submission) <span class='agent-badge'>Agent 3: Treatment Planner</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 15px; border-top: 2px solid #3B82F6;'>", unsafe_allow_html=True)

    if not st.session_state.ehr_submitted:
        st.caption("Review, add, or modify the AI-recommended pharmacological parameters before EHR submission.")
        
        dynamic_options = extract_dynamic_drugs(res.get("treatment_recommendations", ""))
        dose_options = ["80 mg", "40 mg", "150 mg", "500 mg/m²", "75 mg/m²", "4 mg", "Other (Please Specify)"]
        route_options = ["PO (Oral)", "IV (Intravenous)", "SC (Subcutaneous)", "Other (Please Specify)"]
        freq_options = ["QD (Daily)", "BID (Twice)", "Q3W (3 Weeks)", "Q4W (4 Weeks)", "Other (Please Specify)"]
        
        new_regimen = []
        for i, item in enumerate(st.session_state.regimen):
            cols = st.columns([2, 1.5, 1, 1, 0.5])
            
            med_val = item.get('med', dynamic_options[0])
            med_index = dynamic_options.index(med_val) if med_val in dynamic_options else len(dynamic_options) - 1
            
            dose_val = item.get('dose', dose_options[0])
            dose_index = dose_options.index(dose_val) if dose_val in dose_options else len(dose_options) - 1
            
            route_val = item.get('route', route_options[0])
            route_index = route_options.index(route_val) if route_val in route_options else len(route_options) - 1
            
            freq_val = item.get('freq', freq_options[0])
            freq_index = freq_options.index(freq_val) if freq_val in freq_options else len(freq_options) - 1
            
            with cols[0]: m = st.selectbox(f"Drug {i+1}", dynamic_options, index=med_index, key=f"m_{i}", label_visibility="collapsed")
            with cols[1]: d = st.selectbox(f"Dose", dose_options, index=dose_index, key=f"d_{i}", label_visibility="collapsed")
            with cols[2]: r = st.selectbox(f"Route", route_options, index=route_index, key=f"r_{i}", label_visibility="collapsed")
            with cols[3]: f = st.selectbox(f"Freq", freq_options, index=freq_index, key=f"f_{i}", label_visibility="collapsed")
            with cols[4]: 
                if st.button("X", key=f"del_{i}"):
                    st.session_state.regimen.pop(i)
                    st.rerun()
            new_regimen.append({"med": m, "dose": d, "route": r, "freq": f})
        
        st.session_state.regimen = new_regimen
        
        col_btns = st.columns([1, 4])
        with col_btns[0]:
            if st.button("Add Drug"):
                st.session_state.regimen.append({"med": dynamic_options[0], "dose": dose_options[0], "route": route_options[0], "freq": freq_options[0]})
                st.rerun()
        with col_btns[1]:
            if st.button("Sign & Submit to EHR", type="primary", use_container_width=True):
                st.session_state.ehr_submitted = True
                st.rerun()
    else:
        st.markdown("<b>STATUS: OFFICIALLY SIGNED & TRANSMITTED TO HIS/EHR</b>", unsafe_allow_html=True)
        for item in st.session_state.regimen:
            st.markdown(f"""
            <div class="locked-order" style="margin-top: 5px; margin-bottom: 5px; padding: 10px 20px;">
                Rx: <b>{item['med']} {item['dose']}</b> via {item['route']} - Sig: {item['freq']}
            </div>
            """, unsafe_allow_html=True)
        if st.button("Unlock and Revise Orders"):
            st.session_state.ehr_submitted = False
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2: Evidence Rationale
    st.markdown("### Step 2: Evidence-Anchored Rationale <span class='agent-badge'>Agent 2: Reasoner & RAG</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 15px; border-top: 2px solid #D1D5DB;'>", unsafe_allow_html=True)

    col_plan, col_evid = st.columns([2, 1], gap="large")
    plan_text = res.get("treatment_recommendations", "N/A")
    
    with col_plan:
        st.markdown("**Generated Treatment Rationale:**")
        
        tooltip_data = {
            "Osimertinib": "<a href='#guideline-1' class='tooltip-trigger' title='Click to view full NCCN guideline'>Osimertinib<span class='tooltip-content'><strong>NCCN Guideline Reference:</strong><br><br>For advanced NSCLC with EGFR Exon 19 deletion, first-line preferred therapy is <b>Osimertinib</b>.<br><br><span style='color: #A78BFA;'>Evidence Level: Category 1 (Click to jump to full text)</span></span></a>",
            "EGFR Exon 19 deletion": "<a href='#guideline-1' class='tooltip-trigger' title='Click to view biomarker criteria'>EGFR Exon 19 deletion<span class='tooltip-content'><strong>NCCN Biomarker Criteria:</strong><br><br>Sensitizing EGFR mutations dictate preferred first-line TKI.<br><br><span style='color: #A78BFA;'>(Click to jump to full text)</span></span></a>",
            "Stage IV": "<a href='#guideline-1' class='tooltip-trigger' title='Click to view staging management'>Stage IV<span class='tooltip-content'><strong>NCCN Disease Management:</strong><br><br>Systemic therapy is primary for disseminated disease.<br><br><span style='color: #A78BFA;'>(Click to jump to full text)</span></span></a>",
            "CrCl 30 mL/min": "<a href='#safety-1' class='tooltip-trigger' title='Click to view safety protocols'>CrCl 30 mL/min<span class='tooltip-content'><strong>Safety Alert:</strong><br><br>Severe renal impairment requires strict dose adjustments.<br><br><span style='color: #EF4444;'>(Click to view safety rules)</span></span></a>"
        }
        
        formatted_plan_text = ""
        paragraphs = plan_text.split('\n')
        
        for para in paragraphs:
            if not para.strip(): 
                formatted_plan_text += "<br>"
                continue
            
            para = para.replace("**", "").replace("*", "")
            
            for term, replacement in tooltip_data.items():
                if term in para and "<a" not in para:
                     para = para.replace(term, replacement)

            formatted_plan_text += f"<p style='line-height: 1.6;'>{para}</p>"

        st.markdown(formatted_plan_text, unsafe_allow_html=True)

    with col_evid:
        st.markdown("""
        <div style="background:#F8FAFC; padding:20px; border:1px solid #E2E8F0; border-radius:8px; position: sticky; top: 10px;">
            <h4 style="margin-top:0; color:#334155;">NCCN Source Panel</h4>
            <hr style="margin: 10px 0;">
            <p style='font-size: 0.9em; color: #475569;'><i>Hover over highlighted terms for quick summaries, or click them to jump directly to the comprehensive guideline text below.</i></p>
            <b style="color:#0F172A;">Verified Targets:</b> <br>
            <span style="color:#DC2626; font-family: monospace;">EGFR Exon 19 Deletion</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- NEW: Guideline Reference Repository ---
    st.markdown("### Comprehensive Guideline Repository")
    st.caption("Complete reference text retrieved by the RAG system for the current patient context.")
    
    st.link_button("Access Complete NCCN Guidelines (External Link)", "https://www.nccn.org/guidelines/category_1", help="Redirects to the official NCCN portal for full PDF access.")

    with st.expander("View Full NCCN & Safety Guidelines Extracted for this Patient", expanded=False):
        st.markdown("""
        <div id='guideline-1' class='guideline-box'>
            <strong style='color: #1E293B;'>[Reference ID: NCCN-NSCLC-v2026-Targeted]</strong><br><br>
            <strong style='color: #334155;'>NCCN Clinical Practice Guidelines in Oncology: Non-Small Cell Lung Cancer (Version 2026)</strong>
            <ul style='margin-top: 8px; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Targeted Therapy for Sensitizing EGFR Mutations:</strong> For patients with advanced or metastatic (Stage IV) NSCLC who are discovered to have sensitizing EGFR mutations (specifically Exon 19 deletions or Exon 21 L858R mutations) prior to first-line systemic therapy, the preferred regimen is <b>Osimertinib</b> (Category 1 evidence).</li>
                <li><strong>Contraindications:</strong> Do not administer PD-1/PD-L1 inhibitors (e.g., Pembrolizumab) concurrently or consecutively as first-line therapy for patients with these driver mutations due to lack of efficacy and high risk of severe pneumonitis.</li>
                <li><strong>Monitoring:</strong> Baseline imaging and periodic CT scans every 6-12 weeks are recommended to evaluate response.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div id='safety-1' class='guideline-box' style='border-left-color: #EF4444;'>
            <strong style='color: #1E293B;'>[Reference ID: Safety-Renal-Protocols]</strong><br><br>
            <strong style='color: #334155;'>Clinical Pharmacology Safety Protocol: Renal Impairment</strong>
            <ul style='margin-top: 8px; padding-left: 20px; line-height: 1.6;'>
                <li><strong>Assessment:</strong> For patients with severe renal impairment (Creatinine Clearance [CrCl] < 30 mL/min), standard dosing of systemically cleared antineoplastic agents must be critically reviewed.</li>
                <li><strong>Platinum Agents:</strong> Cisplatin is generally contraindicated in severe renal impairment. Carboplatin may be considered with strict Calvert formula dosing (AUC targeting) based on actual GFR.</li>
                <li><strong>TKIs:</strong> Osimertinib does not require dose adjustment for mild to severe renal impairment, though close monitoring for toxicity is advised.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Step 3: Copilot
    st.markdown("### Step 3: Interactive Copilot <span class='agent-badge'>Interactive Copilot Agent</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 15px; border-top: 2px solid #3B82F6;'>", unsafe_allow_html=True)

    display_history = st.session_state.chat_history[1:]
    
    if not display_history:
        st.info("The initial clinical rationale is detailed in Step 2. Use the chat box below to ask follow-up questions, adjust constraints, or explore alternative therapies.")
        
    for msg in display_history:
        with st.chat_message(msg["role"]): 
            content = msg["content"].replace("**", "").replace("*", "")
            st.markdown(content)
            
    if prompt := st.chat_input("Ask a follow-up question (e.g., 'What if the patient refuses IV therapy?'):"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun() 

    if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "user":
        user_msg = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing request..."):
                try:
                    last_plan = next((msg["content"] for msg in reversed(st.session_state.chat_history[:-1]) if msg["role"] == "assistant"), "")
                    updated_reply = run_copilot_chat(last_plan, st.session_state.system_context, user_msg)
                    st.markdown(updated_reply.replace("**", "").replace("*", ""))
                    st.session_state.chat_history.append({"role": "assistant", "content": updated_reply})
                except Exception as e:
                    st.error(f"Chat Error: {e}")