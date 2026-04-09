import os
import json
import re
import dspy
from typing import List, Dict, Any
from multi_agent_pipeline import (
    PrecisionOncologyTools, OncologyExtractor, OncologyCritic, 
    ClinicalReasoner, clean_and_parse_json, llm
)

dspy.settings.configure(lm=llm, cache=False, max_retries=2)

class GuidelineKnowledgeBase:
    """Simulated vector database for oncology guidelines."""
    GUIDELINES = {
        "egfr": "NCCN NSCLC v2026: For patients with EGFR Exon 19 del or L858R mutations, first-line therapy is Osimertinib. If progression occurs, assess for T790M or MET amplification. Do not use immunotherapy (PD-1/PD-L1) as first-line even if PD-L1 is high.",
        "braf": "NCCN Melanoma/Colorectal v2026: BRAF V600E mutation detected. Recommended combination targeted therapy: Dabrafenib + Trametinib or Encorafenib + Cetuximab. Monotherapy with BRAF inhibitors is not recommended due to rapid resistance.",
        "msi": "NCCN Solid Tumors v2026: For dMMR/MSI-H tumors, immune checkpoint inhibitors (e.g., Pembrolizumab, Nivolumab) are strongly recommended across tumor types. Chemotherapy efficacy is generally lower in this subgroup.",
        "er_pr": "NCCN Breast v2026: HR-positive (ER/PR > 1%), HER2-negative early breast cancer. Recommend 5-10 years of adjuvant endocrine therapy (Tamoxifen or Aromatase Inhibitor). Consider adding CDK4/6 inhibitor (Abemaciclib/Ribociclib) if node-positive or high risk.",
        "default": "NCCN General Guidelines v2026: A multidisciplinary tumor board discussion is required. In the absence of actionable driver mutations, consider standard platinum-based doublet chemotherapy or enrollment in a clinical trial."
    }

    @classmethod
    def raw_retrieve(cls, query: str) -> List[str]:
        """Perform initial high-recall coarse retrieval."""
        text = query.lower()
        results = []
        if "egfr" in text: results.append(cls.GUIDELINES["egfr"])
        if "braf" in text: results.append(cls.GUIDELINES["braf"])
        if "msi" in text or "dmmr" in text: results.append(cls.GUIDELINES["msi"])
        if "er " in text or "pr " in text or "luminal" in text: results.append(cls.GUIDELINES["er_pr"])
        if not results: results.append(cls.GUIDELINES["default"])
        return results

class SubQueryGenerator(dspy.Signature):
    """
    Decompose the complex patient case into 2-3 specific, granular search queries.
    Focus on disease progression, specific mutations, and prior treatments.
    """
    clinical_summary = dspy.InputField()
    extracted_data = dspy.InputField()
    sub_queries = dspy.OutputField(desc="2-3 specific search queries separated by a semicolon (;)")

class RelevanceReranker(dspy.Signature):
    """
    Critically evaluate if the retrieved guideline is highly relevant to the SPECIFIC patient context.
    Assign a score of 0 if the guideline contradicts the patient's actual condition.
    """
    patient_context = dspy.InputField()
    retrieved_guideline = dspy.InputField()
    relevance_score = dspy.OutputField(desc="ONLY a single integer from 0 to 10 (10 being perfectly applicable)")

class ContextualTreatmentPlanner(dspy.Signature):
    """
    Develop a comprehensive and actionable treatment plan.
    Synthesize the clinical_summary with the retrieved guidelines. 
    Each guideline has a [Relevance Score: X/10]. 
    - Strongly incorporate guidelines with scores >= 7.
    - For guidelines with scores 4-6, extract specific drug names or trials safely.
    - If all scores are low, rely on your internal medical knowledge to provide specific, actionable recommendations.
    """
    clinical_summary = dspy.InputField()
    scored_guidelines = dspy.InputField(desc="Retrieved guidelines with explicit relevance scores.")
    step_by_step_strategy = dspy.OutputField(desc="A logical blueprint on how to apply the knowledge.")
    treatment_plan = dspy.OutputField(desc="The final detailed clinical recommendation. Must be highly actionable.")

def run_modular_rag_pipeline(note_text: str):
    # Phase 1: Information Extraction and Verification
    tool_findings = PrecisionOncologyTools.execute_all_tools(note_text)
    extractor = dspy.Predict(OncologyExtractor)
    critic = dspy.Predict(OncologyCritic)
    reasoner = dspy.Predict(ClinicalReasoner)
    
    res_ext = extractor(clinical_note=note_text, tool_results=tool_findings, previous_json="None", previous_feedback="None")
    current_json = res_ext.extracted_json
    
    audit = critic(original_note=note_text, tool_results=tool_findings, extracted_json=current_json)
    if audit and "Fail" in str(audit.audit_result):
        res_ref = extractor(clinical_note=note_text, tool_results=tool_findings, previous_json=current_json, previous_feedback=audit.feedback)
        if res_ref: current_json = res_ref.extracted_json

    final_data = clean_and_parse_json(current_json)
    res_reason = reasoner(original_note=note_text, extracted_data=json.dumps(final_data))
    summary_text = res_reason.clinical_summary if res_reason else "N/A"

    # Phase 2: Multi-hop Query Decomposition
    query_gen = dspy.Predict(SubQueryGenerator)
    q_res = query_gen(clinical_summary=summary_text, extracted_data=json.dumps(final_data))
    queries = str(q_res.sub_queries).split(';') if q_res else [summary_text]
    
    raw_docs = set()
    for q in queries:
        raw_docs.update(GuidelineKnowledgeBase.raw_retrieve(q.strip()))
    
    # Phase 3: Semantic Reranking with Soft Injection
    reranker = dspy.Predict(RelevanceReranker)
    scored_docs = []
    
    for doc in raw_docs:
        try:
            score_res = reranker(patient_context=summary_text, retrieved_guideline=doc)
            extracted_nums = re.findall(r'\b(10|[0-9])\b', str(score_res.relevance_score))
            if extracted_nums:
                score = int(extracted_nums[0])
                if score >= 4:
                    scored_docs.append({"score": score, "text": f"[Relevance Score: {score}/10] {doc}"})
        except Exception:
            continue
            
    scored_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
    
    if scored_docs:
        final_context = "\n".join([d["text"] for d in scored_docs])
    else:
        final_context = "[Notice] No highly relevant external guidelines found. Rely on latest internal NCCN oncology consensus to generate specific drug recommendations."

    # Phase 4: Adaptive Strategy-First Generation
    planner = dspy.Predict(ContextualTreatmentPlanner)
    res_plan = planner(clinical_summary=summary_text, scored_guidelines=final_context)
    
    return {
        "structured_data": final_data,
        "clinical_assessment": summary_text,
        "treatment_recommendations": res_plan.treatment_plan if res_plan else "N/A",
        "rag_metadata": {
            "sub_queries_generated": queries,
            "internal_strategy": res_plan.step_by_step_strategy if res_plan else "N/A"
        }
    }


    