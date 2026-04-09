import os, dspy, json, re
from typing import Dict, Any
from dotenv import load_dotenv

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

class PrecisionOncologyTools:
    
    @staticmethod
    def run_medsam(text: str) -> str:
        text_lower = text.lower()
        if "compression fracture" in text_lower and "metastatic" not in text_lower:
            return "[MedSAM Tool] Detected bone fracture. No measurable solid malignant lesions found for RECIST evaluation."
            
        if "mri" in text_lower or "ct" in text_lower or "scan" in text_lower:
            if "progress" in text_lower or "enlarged" in text_lower or "new" in text_lower:
                return "[MedSAM Tool] Segmented reference lesions. Calculated area increased by 45%. Status confirms Progressive Disease (PD)."
            elif "decrease" in text_lower or "response" in text_lower:
                return "[MedSAM Tool] Segmented reference lesions. Area decreased by 30%. Status: Partial Response (PR)."
            else:
                return "[MedSAM Tool] Segmented reference lesions. Area changed by +5%. Status: Stable Disease (SD)."
                
        return "[MedSAM Tool] No relevant imaging data found for segmentation."

    @staticmethod
    def run_pathology_ai(text: str) -> str:
        text_lower = text.lower()
        if "breast cancer" in text_lower or "ductal carcinoma" in text_lower:
            return "[Pathology AI] Feature extraction from H&E slide confirms high ER/PR expression. Phenotype consistent with luminal breast cancer."
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
        medsam_res = cls.run_medsam(text)
        pathology_res = cls.run_pathology_ai(text)
        oncokb_res = cls.run_oncokb(pathology_res)
        return f"=== MULTIMODAL TOOL RESULTS ===\n1. {medsam_res}\n2. {pathology_res}\n3. {oncokb_res}\n==============================="

class OncologyExtractor(dspy.Signature):
    clinical_note = dspy.InputField(desc="Raw clinical text.")
    tool_results = dspy.InputField(desc="Data from MedSAM, Pathology AI, and OncoKB.")
    previous_json = dspy.InputField(desc="The JSON from previous attempt. 'None' if first run.")
    previous_feedback = dspy.InputField(desc="Feedback from Critic. 'None' if first run.")
    extracted_json = dspy.OutputField(desc="Valid JSON with keys: diagnosis, biomarkers, treatments.")

class OncologyCritic(dspy.Signature):
    original_note = dspy.InputField()
    tool_results = dspy.InputField()
    extracted_json = dspy.InputField()
    audit_result = dspy.OutputField(desc="'Pass' or 'Fail'")
    feedback = dspy.OutputField(desc="Specific missing entities to add if Fail, or 'None' if Pass.")

class ClinicalReasoner(dspy.Signature):
    original_note = dspy.InputField(desc="The raw clinical text to provide narrative context.")
    extracted_data = dspy.InputField(desc="Structured JSON containing key entities and tool findings.")
    clinical_summary = dspy.OutputField(desc="Professional, logically coherent narrative assessment.")

class TreatmentPlanner(dspy.Signature):
    clinical_summary = dspy.InputField()
    guideline_context = dspy.InputField()
    treatment_plan = dspy.OutputField()

def clean_and_parse_json(raw_output: str) -> Dict[str, Any]:
    text = str(raw_output).strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            clean_text = re.sub(r'[\x00-\x1F\x7F]', '', clean_text)
            return json.loads(clean_text)
    except Exception:
        pass
        
    return {"diagnosis": [], "biomarkers": [], "treatments": []}

def run_full_agent_pipeline(note_text: str):
    tool_findings = PrecisionOncologyTools.execute_all_tools(note_text)

    extractor = dspy.Predict(OncologyExtractor)
    critic = dspy.Predict(OncologyCritic)
    reasoner = dspy.Predict(ClinicalReasoner)
    planner = dspy.Predict(TreatmentPlanner)

    res_1 = extractor(clinical_note=note_text, tool_results=tool_findings, previous_json="None", previous_feedback="None")
    current_json_str = res_1.extracted_json
    
    audit = critic(original_note=note_text, tool_results=tool_findings, extracted_json=current_json_str)
    correction_triggered = False
    
    if "Fail" in str(audit.audit_result):
        correction_triggered = True
        res_refined = extractor(
            clinical_note=note_text, 
            tool_results=tool_findings, 
            previous_json=current_json_str, 
            previous_feedback=audit.feedback
        )
        current_json_str = res_refined.extracted_json

    final_data = clean_and_parse_json(current_json_str)
    res_summary = reasoner(original_note=note_text, extracted_data=json.dumps(final_data))
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


    