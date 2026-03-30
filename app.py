import streamlit as st
import json
from multi_agent_pipeline import run_full_agent_pipeline

# ==========================================
# 1. 页面基本配置 & 样式注入
# ==========================================
st.set_page_config(page_title="OncoAgent Pro | Multimodal Edition", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #f4f7f6; }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] { background-color: #1e293b; color: white; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div { color: #f8fafc; }
    
    /* 卡片模拟 */
    div.stBlockContainer { padding-top: 2rem; }
    .report-card { 
        background-color: white; 
        padding: 25px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        margin-bottom: 25px;
    }
    
    /* 各种卡片的顶部彩色边框 */
    .card-input { border-top: 5px solid #64748b; }
    .card-tools { border-top: 5px solid #8b5cf6; } 
    .card-json { border-top: 5px solid #3b82f6; }
    .card-assessment { border-top: 5px solid #10b981; }
    .card-plan { border-top: 5px solid #ef4444; }
    
    /* 标题样式 */
    .section-title { 
        margin: 0 0 15px 0; 
        color: #1e293b; 
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* 治疗方案专属高亮框 */
    .plan-box { 
        background-color: #fef2f2; 
        border-left: 5px solid #ef4444; 
        padding: 15px; 
        border-radius: 8px;
        color: #7f1d1d;
        font-weight: 500;
    }

    /* 工具返回结果框 */
    .tool-box {
        background-color: #f3e8ff;
        border-left: 5px solid #8b5cf6;
        padding: 15px;
        border-radius: 8px;
        color: #4c1d95;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 初始化 Session State
# ==========================================
if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 'input'
if 'results' not in st.session_state:
    st.session_state.results = None

# ==========================================
# 3. 侧边栏设计
# ==========================================
with st.sidebar:
    st.title("OncoAgent Pro")
    st.caption("Nature Cancer Benchmark Edition")
    st.markdown("---")
    st.write("**Engine:** DeepSeek-V3")
    st.write("**Architecture:** 4-Agent Framework")
    st.write("**Loaded Tools:**")
    st.write("- MedSAM (Radiology)")
    st.write("- STAMP (Pathology AI)")
    st.write("- OncoKB (Genomics)")
    st.markdown("---")
    if st.button("Analyze New Patient", use_container_width=True):
        st.session_state.workflow_step = 'input'
        st.session_state.results = None
        st.rerun()

# ==========================================
# 4. 主页面逻辑
# ==========================================
st.title("Autonomous Multimodal Oncology Agent")
st.markdown("Integrating LLM reasoning with Precision Oncology Tools (MedSAM, Pathology AI, OncoKB).")

# --- STEP 1: INPUT ---
if st.session_state.workflow_step == 'input':
    with st.container():
        st.markdown('<div class="report-card card-input">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">1. Clinical Oncology Record</div>', unsafe_allow_html=True)
        
        default_note = "Patient is a 55-year-old male. Recent MRI scan of the liver shows suspected metastasis. A needle biopsy of the liver lesion was performed yesterday. Currently seeking treatment options."
        note = st.text_area("Paste pathology reports or clinical notes:", value=default_note, height=200)
        
        if st.button("Start Multimodal Analysis", type="primary", use_container_width=True):
            with st.status("Initiating Autonomous Agent Pipeline...", expanded=True) as status:
                st.write("Scanning text for imaging and biopsy triggers...")
                st.write("Invoking MedSAM for tumor volume calculation...")
                st.write("Running Vision Transformer on H&E slides...")
                st.write("Querying OncoKB for targeted therapies...")
                st.write("Agents 1-4 are synthesizing final report...")
                
                try:
                    res = run_full_agent_pipeline(note)
                    st.session_state.results = res
                    st.session_state.workflow_step = 'done'
                    status.update(label="Analysis Complete! All agents successfully synchronized.", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="Analysis Failed", state="error", expanded=False)
                    st.error(f"Pipeline Error: {e}")
                    
        st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: DASHBOARD OUTPUT ---
if st.session_state.workflow_step == 'done' and st.session_state.results:
    res = st.session_state.results
    
    if res.get("correction_triggered"):
        st.warning("**Self-Correction Triggered:** Critic Agent detected an omission and forced the Extractor to re-evaluate the tool results.")
    else:
        st.success("**Data Verified:** Critic Agent confirmed complete extraction from text and multimodal tools.")

    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown('<div class="report-card card-tools">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">2. Multimodal Tool Findings</div>', unsafe_allow_html=True)
        tool_data = res.get("tool_data_used", "No tool data generated.")
        st.markdown(f'<div class="tool-box">{tool_data}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card card-json">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">3. Structured Data (Verified)</div>', unsafe_allow_html=True)
        st.json(res['final_json'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="report-card card-assessment">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">4. Clinical Assessment (Reasoner)</div>', unsafe_allow_html=True)
        summary_html = res['summary'].replace('\n', '<br>')
        st.markdown(f"<div style='line-height: 1.6; color: #334155;'>{summary_html}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card card-plan">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">5. Treatment Plan (Planner)</div>', unsafe_allow_html=True)
        plan_html = res["treatment_plan"].replace('\n', '<br>')
        st.markdown(f'<div class="plan-box">{plan_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br><p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Generated by Autonomous Multi-Agent Framework • Emulating Nature Cancer 2025 Methodology</p>", unsafe_allow_html=True)