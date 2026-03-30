import os, dspy, json, re
from typing import Dict, Any
from dotenv import load_dotenv

# 1. Environment and Model Initialization
load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")

llm = dspy.OpenAI(
    model='deepseek-chat', 
    api_key=ds_key, 
    api_base='https://api.deepseek.com', 
    max_tokens=2500,
    model_type='chat'
)
dspy.settings.configure(lm=llm, cache=False, max_retries=2)

# ==========================================
# 2. Simulated Tool Layer (Precision Oncology Tools) 
# ==========================================
class PrecisionOncologyTools:
    """Context-aware simulated multimodal tools"""
    
    @staticmethod
    def run_medsam(text: str) -> str:
        """Simulate MedSAM image segmentation tool"""
        text_lower = text.lower()
        
        # Intercept non-cancerous fractures/benign lesions
        if "compression fracture" in text_lower and "metastatic" not in text_lower:
            return "[MedSAM Tool] Detected bone fracture. No measurable solid malignant lesions found for RECIST evaluation."
            
        if "mri" in text_lower or "ct" in text_lower or "scan" in text_lower:
            # Determine status based on keywords
            if "progress" in text_lower or "enlarged" in text_lower or "new" in text_lower:
                return "[MedSAM Tool] Segmented reference lesions. Calculated area increased by 45%. Status confirms Progressive Disease (PD)."
            elif "decrease" in text_lower or "response" in text_lower:
                return "[MedSAM Tool] Segmented reference lesions. Area decreased by 30%. Status: Partial Response (PR)."
            else:
                return "[MedSAM Tool] Segmented reference lesions. Area changed by +5%. Status: Stable Disease (SD)."
                
        return "[MedSAM Tool] No relevant imaging data found for segmentation."

    @staticmethod
    def run_pathology_ai(text: str) -> str:
        """模拟 STAMP 视觉转换器，极严谨的上下文匹配"""
        text_lower = text.lower()
        
        # 修复 Case 3: 不强行声明 HER2 状态，避免与原文冲突
        if "breast cancer" in text_lower or "ductal carcinoma" in text_lower:
            return "[Pathology AI] Feature extraction from H&E slide confirms high ER/PR expression. Phenotype consistent with luminal breast cancer."
        
        # 修复 Case 4: 必须是肺癌才能触发 EGFR，光有"lung"不行
        elif "lung cancer" in text_lower or "nsclc" in text_lower:
            return "[Pathology AI] Feature extraction from H&E slide: EGFR L858R Mutation detected. PD-L1 TPS 50%."
            
        elif "colorectal" in text_lower or "rectal" in text_lower or "cholangio" in text_lower or "esophageal" in text_lower:
            return "[Pathology AI] Feature extraction from H&E slide: MSI-High (Probability 0.95), BRAF V600E Mutant, KRAS Wild-type."
            
        elif "myeloma" in text_lower or "lymphoma" in text_lower:
            return "[Pathology AI] Features consistent with hematologic malignancy. No solid tumor actionable mutations found."
            
        else:
            return "[Pathology AI] Tissue insufficient or no actionable solid tumor biomarkers identified."
            
    @staticmethod
    def run_oncokb(biomarker_text: str) -> str:
        """Simulate OncoKB database search"""
        findings = []
        if "BRAF V600E" in biomarker_text:
            findings.append("Dabrafenib + Trametinib (Level 1 evidence)")
        if "MSI-High" in biomarker_text:
            findings.append("Pembrolizumab (Level 1 evidence for MSI-H solid tumors)")
        if "EGFR L858R" in biomarker_text:
            findings.append("Osimertinib (Level 1 evidence for EGFR-mutant NSCLC)")
        if "ER Positive" in biomarker_text:
            findings.append("Endocrine therapy (e.g., Fulvestrant, Aromatase Inhibitors) +/- CDK4/6 inhibitors (Level 1 evidence)")
        
        if findings:
            return f"[OncoKB Search] Actionable targets identified: {', '.join(findings)}."
        return "[OncoKB Search] No actionable genomic targets found for current indication."

    @classmethod
    def execute_all_tools(cls, text: str) -> str:
        """Execute all tools and aggregate report"""
        medsam_res = cls.run_medsam(text)
        pathology_res = cls.run_pathology_ai(text)
        oncokb_res = cls.run_oncokb(pathology_res)
        
        return f"=== MULTIMODAL TOOL RESULTS ===\n1. {medsam_res}\n2. {pathology_res}\n3. {oncokb_res}\n==============================="


# ==========================================
# 3. Agent Signatures Definition
# ==========================================
class OncologyExtractor(dspy.Signature):
    """
    Extraction Task: Convert raw notes AND multimodal tool results into structured JSON.
    CRITICAL: You MUST include findings from the MULTIMODAL TOOL RESULTS (e.g., new mutations, MSI status) into the 'biomarkers' and 'diagnosis' JSON fields.
    """
    clinical_note = dspy.InputField(desc="Raw clinical text.")
    tool_results = dspy.InputField(desc="Data from MedSAM, Pathology AI, and OncoKB.")
    previous_feedback = dspy.InputField(desc="Feedback from Critic. 'None' if first run.")
    extracted_json = dspy.OutputField(desc="Valid JSON with keys: diagnosis, biomarkers, treatments.")

class OncologyCritic(dspy.Signature):
    """
    Audit Task: Ensure the extracted JSON captures ALL critical information from BOTH the clinical note AND the tool results.
    Return 'Fail' if critical biomarkers from Pathology AI or OncoKB are missing in the JSON.
    """
    original_note = dspy.InputField()
    tool_results = dspy.InputField()
    extracted_json = dspy.InputField()
    audit_result = dspy.OutputField(desc="'Pass' or 'Fail'")
    feedback = dspy.OutputField(desc="Specific missing entities if Fail.")

class ClinicalReasoner(dspy.Signature):
    """Synthesize data into a professional oncological assessment, explaining how biomarkers link to the diagnosis."""
    extracted_data = dspy.InputField(desc="Structured JSON including tool findings.")
    clinical_summary = dspy.OutputField(desc="Professional narrative assessment.")

class TreatmentPlanner(dspy.Signature):
    """Develop a treatment plan based on the assessment, incorporating OncoKB targeted therapies where applicable."""
    clinical_summary = dspy.InputField()
    guideline_context = dspy.InputField()
    treatment_plan = dspy.OutputField()

# ==========================================
# 4. Helper Functions & Pipeline
# ==========================================
def clean_and_parse_json(raw_output: str) -> Dict[str, Any]:
    try:
        match = re.search(r'(\{.*\})', str(raw_output), re.DOTALL)
        text = match.group(1) if match else str(raw_output)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return json.loads(text)
    except Exception:
        return {"diagnosis": [], "biomarkers": [], "treatments": []}

def run_full_agent_pipeline(note_text: str):
    # Execute tools autonomously to gather incremental data
    tool_findings = PrecisionOncologyTools.execute_all_tools(note_text)

    extractor = dspy.Predict(OncologyExtractor)
    critic = dspy.Predict(OncologyCritic)
    reasoner = dspy.Predict(ClinicalReasoner)
    planner = dspy.Predict(TreatmentPlanner)

    # 1. Joint Extraction (Text + Tools)
    res_1 = extractor(clinical_note=note_text, tool_results=tool_findings, previous_feedback="None")
    current_json_str = res_1.extracted_json
    
    # 2. Audit Phase
    audit = critic(original_note=note_text, tool_results=tool_findings, extracted_json=current_json_str)
    correction_triggered = False
    
    if "Fail" in str(audit.audit_result):
        correction_triggered = True
        res_refined = extractor(clinical_note=note_text, tool_results=tool_findings, previous_feedback=audit.feedback)
        current_json_str = res_refined.extracted_json

    # 3. Clinical Reasoning & Planning
    final_data = clean_and_parse_json(current_json_str)
    res_summary = reasoner(extracted_data=json.dumps(final_data))
    res_plan = planner(
        clinical_summary=res_summary.clinical_summary, 
        guideline_context="NCCN Oncology Guidelines 2026"
    )
    
    return {
        "final_json": final_data,
        "summary": res_summary.clinical_summary,
        "treatment_plan": res_plan.treatment_plan,
        "correction_triggered": correction_triggered,
        "tool_data_used": tool_findings
    }

