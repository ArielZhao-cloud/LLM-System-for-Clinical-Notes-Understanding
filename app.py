import streamlit as st
import json
import uuid
import traceback
from langgraph.graph import StateGraph, START, END

# Directly import base node functions from the backend, taking full control of the flow
from multi_agent_pipeline import (
    ClinicalNoteState,
    extraction_node,
    reasoning_node,
    critic_node,
    decision_node,
    routing_logic,
    optimized_extractor,
    USE_RAG
)

# ==========================================
# 1. 页面配置与原生 CSS 优化 (修复对比度问题)
# ==========================================
st.set_page_config(page_title="Clinical AI Copilot", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Segoe UI', Roboto, sans-serif; }
    header {visibility: hidden;}
    .stHeading h1 { color: #1e3a8a; font-weight: 800; }

    /* 修复 1：JSON 编辑器极客风格，并强制修复 disabled 状态下的对比度 */
    .stTextArea textarea {
        font-family: 'Courier New', Courier, monospace;
        background-color: #0f172a !important; 
        color: #10b981 !important;
        border-radius: 6px; font-size: 14px; line-height: 1.5;
    }
    .stTextArea textarea:disabled {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important; /* 强制覆盖苹果/谷歌浏览器的禁用态变灰 */
        cursor: not-allowed;
    }

    /* 侧边栏定制 */
    [data-testid="stSidebar"] { background-color: #1e3a8a; color: white; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white; }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }

    /* 修复 2：强制侧边栏按钮为白底深蓝字，解决白底白字看不清的问题 */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        font-weight: bold;
        border: 1px solid #cbd5e1;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Core Architecture Refactor: Bulletproof Dual-Graph System
# ==========================================
@st.cache_resource
def build_graphs():
    g1 = StateGraph(ClinicalNoteState)
    g1.add_node("Extractor", extraction_node)
    g1.add_node("Critic", critic_node)
    g1.add_edge(START, "Extractor")
    g1.add_edge("Extractor", "Critic")
    g1.add_conditional_edges("Critic", routing_logic, {"revise": "Extractor", "approved": END})

    g2 = StateGraph(ClinicalNoteState)
    g2.add_node("Reasoner", reasoning_node)
    g2.add_node("DecisionMaker", decision_node)
    g2.add_edge(START, "Reasoner")
    g2.add_edge("Reasoner", "DecisionMaker")
    g2.add_edge("DecisionMaker", END)

    return g1.compile(), g2.compile()


extraction_graph, decision_graph = build_graphs()

# ==========================================
# 3. State Initialization
# ==========================================
if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 'input'

if 'current_note' not in st.session_state:
    st.session_state.current_note = "Patient is a 65yo M presenting with acute ST-segment elevation myocardial infarction (STEMI). PCI was performed immediately. Currently prescribed Aspirin and Clopidogrel. No other medications noted."

if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {}

# ==========================================
# 4. Sidebar: System Telemetry
# ==========================================
with st.sidebar:
    st.markdown("## Clinical Copilot")
    st.markdown("### System Telemetry")
    st.markdown("---")

    if optimized_extractor:
        st.markdown("[OK] **Extractor:** Optimized")
    else:
        st.markdown("[WARN] **Extractor:** Baseline")

    if USE_RAG:
        st.markdown("[OK] **Vector DB:** Connected")
    else:
        st.markdown("[ERR] **Vector DB:** Disconnected")

    st.markdown("---")
    st.markdown("### Process Progress")
    if st.session_state.workflow_step == 'input':
        st.write("[Current] -> [ ] -> [ ]")
    elif st.session_state.workflow_step == 'reviewing':
        st.write("[Done] -> [Current] -> [ ]")
    else:
        st.write("[Done] -> [Done] -> [Current]")

    st.markdown("---")
    if st.button("Restart System Session", use_container_width=True):
        st.session_state.workflow_step = 'input'
        st.session_state.agent_state = {}
        st.rerun()

# ==========================================
# 5. Main UI: Dashboard
# ==========================================
st.title("Clinical AI Copilot (Human-in-the-Loop)")
st.markdown("A sophisticated CDSS decoupled pipeline featuring rigorous auditing and explicit physician sign-off.")
st.markdown("---")

# ---------------------------------------------------------
# [Step 1/3] Clinical Note Input
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown("### [1/3] Patient Clinical Note Input")
    is_step1 = (st.session_state.workflow_step == 'input')

    clinical_note = st.text_area("Raw clinical text:", value=st.session_state.current_note, height=150,
                                 disabled=not is_step1)

    if is_step1:
        if st.button("Step 1: AI Extraction & Audit", type="primary"):
            if clinical_note.strip():
                st.session_state.current_note = clinical_note

                initial_state = {
                    "original_text": clinical_note,
                    "revision_count": 0,
                    "status": "processing",
                    "critic_feedback": "None"
                }

                with st.spinner('AI Phase 1: Extracting and auditing entities...'):
                    try:
                        final_state_1 = extraction_graph.invoke(initial_state)
                        st.session_state.agent_state = final_state_1
                        st.session_state.workflow_step = 'reviewing'
                        st.rerun()
                    except Exception as e:
                        st.error("❌ Extraction Graph Failed!")
                        st.code(traceback.format_exc())
            else:
                st.error("Please enter a note.")
    else:
        if st.button("<- Re-enter Input Note", key="back_to_1"):
            st.session_state.workflow_step = 'input'
            st.rerun()

# ---------------------------------------------------------
# [Step 2/3] Physician Interception & Review
# ---------------------------------------------------------
if st.session_state.workflow_step in ['reviewing', 'done']:
    with st.container(border=True):
        st.markdown("### [2/3] Action Required: Physician Sign-off")
        is_step2 = (st.session_state.workflow_step == 'reviewing')

        if is_step2:
            st.markdown("""
            <div style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                <strong style="color: #d97706;">Attention:</strong> The AI has completed entity extraction and internal audit. Please review and edit the structured JSON data below to ensure absolute accuracy before downstream generation.
            </div>
            """, unsafe_allow_html=True)

        extracted_json_str = st.session_state.agent_state.get("extracted_entities", "{}")
        edited_json_str = st.text_area("Structured Clinical Data (Editable JSON):", value=extracted_json_str,
                                       height=350, disabled=not is_step2)

        if is_step2:
            if st.button("Step 2: Sign-off & Generate Plan", type="primary"):
                with st.spinner('AI Phase 2: Generating SOAP note and consulting guidelines...'):
                    try:
                        st.session_state.agent_state["extracted_entities"] = edited_json_str
                        final_state_2 = decision_graph.invoke(st.session_state.agent_state)
                        st.session_state.agent_state.update(final_state_2)
                        st.session_state.workflow_step = 'done'
                        st.rerun()
                    except Exception as e:
                        st.error("❌ Decision Graph Failed!")
                        st.code(traceback.format_exc())
        else:
            if st.button("<- Unlock & Re-edit JSON Data", key="back_to_2"):
                st.session_state.workflow_step = 'reviewing'
                st.rerun()

# ---------------------------------------------------------
# [Step 3/3] Final Display (with Rich Visual Clinical Dashboard)
# ---------------------------------------------------------
if st.session_state.workflow_step == 'done':
    with st.container(border=True):
        st.markdown("### [3/3] Final Clinical Outputs")

        col_out1, col_out2 = st.columns(2)

        with col_out1:
            with st.container(border=True):
                st.markdown("#### Phase A: SOAP Summary")
                st.markdown("*Objective SOAP note based strictly on human-approved data.*")

                summary_output = st.session_state.agent_state.get('current_summary', '')
                if summary_output:
                    st.write(summary_output)
                else:
                    st.error("Summary is empty. Check terminal for specific node errors.")

        with col_out2:
            with st.container(border=True):
                st.markdown("#### Phase B: Decision Support")
                st.markdown("*Guideline-grounded insights powered by RAG.*")

                plan_output = st.session_state.agent_state.get('final_treatment_plan', '')

                # ==========================================
                # New: Smart "Visual Dashboard" for Clinical Insights
                # Traffic light indicators to instantly grab the physician's attention
                # ==========================================
                if plan_output:
                    if "Missing Recommendations" in plan_output or "missing" in plan_output.lower():
                        st.markdown("""
                        <div style="background-color: #fee2e2; border-left: 5px solid #ef4444; padding: 12px; border-radius: 6px; margin-bottom: 20px;">
                            <h4 style="color: #b91c1c; margin-top: 0; margin-bottom: 5px;">🚨 Guideline Variance Detected</h4>
                            <p style="color: #7f1d1d; margin-bottom: 0;"><strong>Action Needed:</strong> The current treatment plan is missing critical medications per AHA/ACC protocols. Please review the detailed deviations below.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #dcfce7; border-left: 5px solid #22c55e; padding: 12px; border-radius: 6px; margin-bottom: 20px;">
                            <h4 style="color: #15803d; margin-top: 0; margin-bottom: 5px;">✅ Fully Compliant</h4>
                            <p style="color: #166534; margin-bottom: 0;">The current treatment aligns perfectly with retrieved medical guidelines.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Specific decision report
                    st.write(plan_output)
                else:
                    st.error("Treatment plan is empty. Check terminal for specific node errors.")

                with st.expander("View Retrieved Guideline Context"):
                    st.info(st.session_state.agent_state.get('retrieved_guidelines', 'No context retrieved.'))

                    # bala
                    