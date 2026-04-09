import os, json, re, dspy
from typing import Dict, Any
from multi_agent_pipeline import (
    PrecisionOncologyTools, OncologyExtractor, OncologyCritic, 
    ClinicalReasoner, clean_and_parse_json, llm
)

dspy.settings.configure(lm=llm, cache=False, max_retries=2)

class GuidelineKnowledgeBase:
    """A simulated local knowledge base for oncology guidelines."""
    
    GUIDELINES = {
        "egfr": "NCCN NSCLC v2026: For patients with EGFR Exon 19 del or L858R mutations, first-line therapy is Osimertinib. If progression occurs, assess for T790M or MET amplification. Do not use immunotherapy (PD-1/PD-L1) as first-line even if PD-L1 is high.",
        "braf": "NCCN Melanoma/Colorectal v2026: BRAF V600E mutation detected. Recommended combination targeted therapy: Dabrafenib + Trametinib or Encorafenib + Cetuximab. Monotherapy with BRAF inhibitors is not recommended due to rapid resistance.",
        "msi": "NCCN Solid Tumors v2026: For dMMR/MSI-H tumors, immune checkpoint inhibitors (e.g., Pembrolizumab, Nivolumab) are strongly recommended across tumor types. Chemotherapy efficacy is generally lower in this subgroup.",
        "er_pr": "NCCN Breast v2026: HR-positive (ER/PR > 1%), HER2-negative early breast cancer. Recommend 5-10 years of adjuvant endocrine therapy (Tamoxifen or Aromatase Inhibitor). Consider adding CDK4/6 inhibitor (Abemaciclib/Ribociclib) if node-positive or high risk.",
        "default": "NCCN General Guidelines v2026: A multidisciplinary tumor board discussion is required. In the absence of actionable driver mutations, consider standard platinum-based doublet chemotherapy or enrollment in a clinical trial."
    }

    @classmethod
    def retrieve(cls, query_text: str) -> str:
        text = query_text.lower()
        retrieved_contexts = []
        
        if "egfr" in text: retrieved_contexts.append(cls.GUIDELINES["egfr"])
        if "braf" in text: retrieved_contexts.append(cls.GUIDELINES["braf"])
        if "msi" in text or "dmmr" in text: retrieved_contexts.append(cls.GUIDELINES["msi"])
        if "er " in text or "pr " in text or "luminal" in text: retrieved_contexts.append(cls.GUIDELINES["er_pr"])
        
        if not retrieved_contexts:
            retrieved_contexts.append(cls.GUIDELINES["default"])
            
        return "\n".join(retrieved_contexts)

class SearchAgent(dspy.Signature):
    """
    Generate a precise search query for oncology guidelines.
    Based on the clinical assessment and structured data, create a 5-10 word query.
    Example: 'First-line treatment for Stage IV NSCLC with EGFR L858R'
    """
    clinical_summary = dspy.InputField()
    extracted_data = dspy.InputField()
    search_query = dspy.OutputField(desc="A specific clinical search query.")

class ContextAwarePlanner(dspy.Signature):
    """
    Develop a treatment plan by synthesizing patient data and clinical guidelines.
    CRITICAL RULE: The 'retrieved_guidelines' are for REFERENCE only. 
    If a guideline contradicts the patient's specific history (e.g., prior drug failure, allergies), 
    you MUST prioritize the patient's safety and historical records over the guideline.
    Explain your reasoning if you deviate from the general guideline.
    """
    clinical_summary = dspy.InputField()
    retrieved_guidelines = dspy.InputField()
    treatment_plan = dspy.OutputField()

def run_advanced_rag_pipeline(note_text: str):
    tool_findings = PrecisionOncologyTools.execute_all_tools(note_text)
    
    extractor = dspy.Predict(OncologyExtractor)
    critic = dspy.Predict(OncologyCritic)
    reasoner = dspy.Predict(ClinicalReasoner)
    search_agent = dspy.Predict(SearchAgent)
    planner = dspy.Predict(ContextAwarePlanner)
    
    # 1. Extraction & Audit
    res_ext = extractor(clinical_note=note_text, tool_results=tool_findings, previous_json="None", previous_feedback="None")
    current_json_str = res_ext.extracted_json
    
    audit = critic(original_note=note_text, tool_results=tool_findings, extracted_json=current_json_str)
    
    if audit and "Fail" in str(audit.audit_result):
        res_ref = extractor(clinical_note=note_text, tool_results=tool_findings, previous_json=current_json_str, previous_feedback=audit.feedback)
        if res_ref:
            current_json_str = res_ref.extracted_json

    # 2. Reasoning
    final_data = clean_and_parse_json(current_json_str)
    res_reason = reasoner(original_note=note_text, extracted_data=json.dumps(final_data))
    summary_text = res_reason.clinical_summary if res_reason else "N/A"

    # 3. Dynamic Query & Retrieval
    query_res = search_agent(clinical_summary=summary_text, extracted_data=json.dumps(final_data))
    query = query_res.search_query if query_res else "oncology treatment guidelines"
    retrieved_docs = GuidelineKnowledgeBase.retrieve(query + " " + summary_text)
    
    # 4. Context-Aware Planning
    res_plan = planner(clinical_summary=summary_text, retrieved_guidelines=retrieved_docs)
    
    return {
        "structured_data": final_data,
        "clinical_assessment": summary_text,
        "treatment_recommendations": res_plan.treatment_plan if res_plan else "N/A",
        "retrieved_query": query,
        "retrieved_docs": retrieved_docs
    }