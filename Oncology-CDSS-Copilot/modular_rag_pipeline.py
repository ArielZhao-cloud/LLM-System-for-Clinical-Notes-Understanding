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
    """
    Expert-Curated Structured Knowledge Base for Oncology.
    High signal-to-noise ratio to prevent hallucination while maximizing completeness.
    """
    KNOWLEDGE_GRAPH = {
        # --- NSCLC (Lung) ---
        "nsclc_egfr_sens": (
            ["egfr", "l858r", "exon 19", "19del"], 
            "NCCN NSCLC v2026: For advanced NSCLC with EGFR Exon 19 deletion or L858R, first-line preferred therapy is Osimertinib. Do not use PD-1/PD-L1 inhibitors as first-line."
        ),
        "nsclc_egfr_t790m": (
            ["t790m", "progression on tki", "tki resistance"], 
            "NCCN NSCLC v2026: For NSCLC progressing on 1st/2nd-gen EGFR TKI, test for T790M. If positive, use Osimertinib. If negative, switch to platinum doublet chemotherapy."
        ),
        "nsclc_egfr_ex20": (
            ["exon 20 insertion", "ex20ins", "exon 20"], 
            "NCCN NSCLC v2026: For EGFR Exon 20 insertion mutations, preferred targeted therapies include Amivantamab or Mobocertinib after platinum-based chemotherapy."
        ),
        "nsclc_alk": (
            ["alk", "alk fusion", "alk positive"], 
            "NCCN NSCLC v2026: For ALK rearrangement positive NSCLC, preferred first-line therapies are Alectinib, Brigatinib, or Lorlatinib."
        ),
        "nsclc_ros1": (
            ["ros1"], 
            "NCCN NSCLC v2026: For ROS1 rearrangement positive NSCLC, first-line targeted therapy is Entrectinib or Crizotinib."
        ),
        "nsclc_kras_g12c": (
            ["kras", "g12c"], 
            "NCCN NSCLC v2026: For KRAS G12C mutation, Sotorasib or Adagrasib are indicated for patients progressing after at least one prior systemic therapy."
        ),
        "nsclc_pdl1_high": (
            ["pd-l1 > 50%", "pdl1 > 50%", "pd-l1 high"], 
            "NCCN NSCLC v2026: For PD-L1 >= 50% and negative for actionable mutations (EGFR/ALK/ROS1), first-line Pembrolizumab, Cemiplimab, or Atezolizumab monotherapy is preferred."
        ),
        
        # --- Breast Cancer ---
        "breast_hr_pos_her2_neg": (
            ["er+", "pr+", "hr+", "hr positive", "luminal", "her2-", "her2 negative"], 
            "NCCN Breast v2026: For HR-positive/HER2-negative metastatic disease, first-line preferred is a CDK4/6 inhibitor (Palbociclib, Ribociclib, or Abemaciclib) + Endocrine therapy (Letrozole, Anastrozole, or Fulvestrant)."
        ),
        "breast_her2_pos": (
            ["her2+", "her2 positive", "her2 amplified"], 
            "NCCN Breast v2026: For HER2-positive metastatic breast cancer, first-line therapy is Trastuzumab + Pertuzumab + a Taxane (Docetaxel/Paclitaxel). Second-line preferred is T-DXd (Enhertu)."
        ),
        "breast_tnbc_brca": (
            ["tnbc", "triple negative", "brca1", "brca2"], 
            "NCCN Breast v2026: For Triple-Negative Breast Cancer (TNBC) with germline BRCA1/2 mutation, PARP inhibitors (Olaparib or Talazoparib) are preferred options."
        ),

        # --- Colorectal Cancer (CRC) ---
        "crc_msi_high": (
            ["msi-h", "dmmr", "microsatellite instability", "msi"], 
            "NCCN Colorectal v2026: For dMMR/MSI-H advanced colorectal cancer, Pembrolizumab or Nivolumab (+/- Ipilimumab) are strongly recommended as first-line therapy over chemotherapy."
        ),
        "crc_braf": (
            ["braf v600e", "braf+"], 
            "NCCN Colorectal v2026: For BRAF V600E mutated CRC, first-line remains chemo+biologic. For second-line and beyond, targeted doublet Encorafenib + Cetuximab/Panitumumab is recommended. NEVER use BRAF monotherapy."
        ),
        "crc_ras_wt_left": (
            ["kras wild", "nras wild", "ras wt", "left-sided"], 
            "NCCN Colorectal v2026: For RAS wild-type left-sided colon cancer, first-line chemotherapy (FOLFOX/FOLFIRI) combined with an anti-EGFR antibody (Cetuximab or Panitumumab) is preferred."
        ),

        # --- Melanoma ---
        "melanoma_braf": (
            ["melanoma", "braf v600", "braf"], 
            "NCCN Melanoma v2026: For BRAF V600 mutated advanced melanoma, combination BRAF/MEK inhibition (Dabrafenib+Trametinib, Vemurafenib+Cobimetinib, or Encorafenib+Binimetinib) or Immunotherapy (anti-PD1 +/- anti-CTLA4) are first-line options."
        ),

        # --- Prostate / Ovarian / Pancreatic ---
        "prostate_mcrpc_brca": (
            ["mcrpc", "castration-resistant", "prostate", "brca"], 
            "NCCN Prostate v2026: For metastatic castration-resistant prostate cancer (mCRPC) with HRR/BRCA mutations progressing on AR-directed therapy, PARP inhibitors (Olaparib, Rucaparib) are indicated."
        ),
        "ovarian_brca_maint": (
            ["ovarian", "brca", "maintenance"], 
            "NCCN Ovarian v2026: For BRCA-mutated advanced ovarian cancer responding to platinum-based chemotherapy, PARP inhibitor maintenance therapy (Olaparib, Niraparib) is strongly recommended."
        ),

        # --- Supportive Care ---
        "support_bone_mets": (
            ["bone met", "bone metastasis", "skeletal"], 
            "NCCN Supportive Care v2026: For patients with solid tumor bone metastases, initiate bone-modifying agents (Zoledronic acid or Denosumab) to prevent skeletal-related events. Monitor calcium levels and recommend dental exam."
        ),
        "support_neutropenia": (
            ["neutropenia", "febrile", "anc <"], 
            "NCCN Supportive Care v2026: For high risk of febrile neutropenia (>20%) from chemotherapy, prophylactic G-CSF (e.g., Filgrastim, Pegfilgrastim) is recommended."
        ),
        
        # --- Fallback ---
        "default": (
            ["default"], 
            "NCCN General Oncology Consensus v2026: A multidisciplinary tumor board discussion is required. Ensure NGS comprehensive genomic profiling is complete. In the absence of specific actionable targets, consider standard platinum-based doublet chemotherapy, evaluate performance status, and prioritize palliative symptom management and clinical trial enrollment."
        )
    }

    @classmethod
    def raw_retrieve(cls, query: str) -> List[str]:
        """Smart keyword-matching retrieval."""
        text = query.lower()
        results = set()
        
        for key, (keywords, guideline_text) in cls.KNOWLEDGE_GRAPH.items():
            if key == "default":
                continue
            for kw in keywords:
                if kw in text:
                    results.add(guideline_text)
                    break 
                    
        if not results:
            results.add(cls.KNOWLEDGE_GRAPH["default"][1])
            
        return list(results)

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
    Develop a highly COMPREHENSIVE, detailed, and flowing oncology treatment plan.
    Synthesize the patient's clinical_summary with the retrieved medical guidelines.
    
    CRITICAL INSTRUCTIONS FOR SCORING HIGH:
    1. TARGETED ACCURACY: You MUST use the exact targeted drugs, regimens, and mutations mentioned in the 'scored_guidelines'. Treat them as absolute clinical truth.
    2. MAXIMUM COMPLETENESS: Do NOT just output a brief checklist. You must write a rich, multi-paragraph clinical narrative. Expand extensively on the guidelines using your internal medical knowledge to cover:
       - Detailed rationale for the chosen primary therapy
       - Comprehensive side-effect and toxicity management strategies
       - Long-term follow-up and monitoring schedules
       - Palliative, nutritional, and supportive care details
    3. TONE: Write like an expert attending oncologist delivering a thorough, actionable, and holistic patient care strategy.
    """
    clinical_summary = dspy.InputField()
    scored_guidelines = dspy.InputField(desc="Retrieved guidelines with explicit relevance scores.")
    step_by_step_strategy = dspy.OutputField(desc="A logical blueprint on how to apply the knowledge.")
    treatment_plan = dspy.OutputField(desc="The final detailed clinical recommendation. MUST be highly comprehensive, rich in detail, and deeply actionable.")

class CopilotChatUpdater(dspy.Signature):
    """
    Act as an Interactive Clinical Copilot.
    Update or modify an existing clinical treatment plan based on the doctor's new instruction.
    
    CRITICAL RULES:
    1. Maintain the rigorous, structured format (Pharmacology, Toxicity, Supportive Care, Next Steps).
    2. Answer the doctor's specific question or apply the requested modification.
    3. Ensure all changes remain medically safe and grounded in the provided context.
    """
    original_plan = dspy.InputField(desc="The previously generated treatment plan.")
    patient_context = dspy.InputField(desc="The verified clinical summary and retrieved guidelines.")
    doctor_instruction = dspy.InputField(desc="The doctor's new query, constraint, or modification request.")
    updated_plan = dspy.OutputField(desc="The revised treatment plan addressing the doctor's instruction.")

def run_copilot_chat(original_plan: str, context: str, user_msg: str) -> str:
    updater = dspy.Predict(CopilotChatUpdater)
    res = updater(
        original_plan=original_plan,
        patient_context=context,
        doctor_instruction=user_msg
    )
    return res.updated_plan

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