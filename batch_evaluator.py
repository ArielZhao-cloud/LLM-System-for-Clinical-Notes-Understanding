import os, json, dspy, time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from multi_agent_pipeline import (
    OncologyExtractor, OncologyCritic, ClinicalReasoner, 
    TreatmentPlanner, clean_and_parse_json, PrecisionOncologyTools
)

# ==========================================
# 1. Environment & Model Initialization
# ==========================================
load_dotenv()
RAW_DATASET_PATH = "oncology_raw_samples.json"
OUTPUT_FILE = "oncology_final_reports.json"
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

llm = dspy.OpenAI(
    model='deepseek-chat', 
    api_key=DS_KEY, 
    api_base='https://api.deepseek.com', 
    max_tokens=2500,
    model_type='chat'
)
dspy.settings.configure(lm=llm, cache=False, max_retries=1)

# Define experiment nodes
predictor_baseline = dspy.Predict(OncologyExtractor)
predictor_agent = dspy.Predict(OncologyExtractor) 
critic_node = dspy.Predict(OncologyCritic)
reasoner_node = dspy.Predict(ClinicalReasoner)
planner_node = dspy.Predict(TreatmentPlanner)

def calc_density(d):
    """Enhanced density calculation supporting lists and nested dictionaries."""
    if not isinstance(d, dict): return 0
    count = 0
    
    for key in ["diagnosis", "biomarkers", "treatments"]:
        val = d.get(key, [])
        if isinstance(val, list): 
            count += len(val)
        elif isinstance(val, dict): 
            count += len(val.keys())
        elif isinstance(val, str) and val.strip():
            count += 1
    return count

def safe_api_call(func, **kwargs):
    """API call wrapper with exponential backoff to handle rate limits."""
    for attempt in range(5):
        try:
            return func(**kwargs)
        except Exception as e:
            wait_time = (attempt + 1) * 6 
            print(f"\n[API Busy] Retrying in {wait_time}s... (Error: {e})")
            time.sleep(wait_time)
    return None

# ==========================================
# 2. Core Experiment Loop
# ==========================================
def run_ablation_study():
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"[ERROR] {RAW_DATASET_PATH} not found.")
        return

    with tqdm(total=1, desc="Loading data") as pbar:
        with open(RAW_DATASET_PATH, 'r', encoding='utf-8') as f:
            samples = json.load(f)[:100]
        pbar.update(1)

    results = {"Group_A": [], "Group_B": [], "Group_C": [], "Corrections": 0}
    final_reports = []

    print(f"\n[START] Running Final 4-Agent Evaluation (Saving to {OUTPUT_FILE})...")

    for i, item in enumerate(tqdm(samples, desc="Processing Samples")):
        text = item["text"][:2800] 
        report_entry = {"sample_id": i, "original_text": text}

        # Retrieve extra facts by invoking multimodal tools
        tool_data = PrecisionOncologyTools.execute_all_tools(text)

        # --- Group A: Baseline (Single pass without tools) ---
        res_a = safe_api_call(predictor_baseline, clinical_note=text, tool_results="None", previous_feedback="None")
        if res_a:
            data_a = clean_and_parse_json(res_a.extracted_json)
            results["Group_A"].append(calc_density(data_a))
        
        # --- Group B: Single Optimized (Extraction Only, with tools) ---
        res_b = safe_api_call(predictor_agent, clinical_note=text, tool_results=tool_data, previous_feedback="None")
        if res_b:
            data_b = clean_and_parse_json(res_b.extracted_json)
            results["Group_B"].append(calc_density(data_b))

            # --- Group C: Multi-Agent (Extractor + Critic + Reasoner + Planner) ---
            current_js_str = res_b.extracted_json
            
            audit = safe_api_call(critic_node, original_note=text, tool_results=tool_data, extracted_json=current_js_str)
            
            is_corrected = False
            if audit and "Fail" in str(audit.audit_result):
                results["Corrections"] += 1
                is_corrected = True
                refined = safe_api_call(predictor_agent, clinical_note=text, tool_results=tool_data, previous_feedback=audit.feedback)
                if refined:
                    current_js_str = refined.extracted_json
            
            data_c = clean_and_parse_json(current_js_str)
            results["Group_C"].append(calc_density(data_c))

            # Generate final clinical reports
            summary_res = safe_api_call(reasoner_node, extracted_data=json.dumps(data_c))
            plan_res = safe_api_call(planner_node, 
                                    clinical_summary=summary_res.clinical_summary if summary_res else "N/A", 
                                    guideline_context="NCCN 2026 Oncology Guidelines")

            # Save all outputs
            report_entry.update({
                "structured_data": data_c,
                "clinical_assessment": summary_res.clinical_summary if summary_res else "Failed to generate",
                "treatment_recommendations": plan_res.treatment_plan if plan_res else "Failed to generate",
                "was_corrected": is_corrected,
                "critic_feedback": audit.feedback if is_corrected else "N/A",
                "tool_data_used": tool_data
            })
            final_reports.append(report_entry)

            print(f"\n[Sample {i} Assessment Summary]:\n{summary_res.clinical_summary[:150]}...\n")
            time.sleep(2)

    # ==========================================
    # 3. Statistics & Persistence
    # ==========================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_reports, f, indent=2, ensure_ascii=False)

    avg_a = np.mean(results["Group_A"]) if results["Group_A"] else 0
    avg_c = np.mean(results["Group_C"]) if results["Group_C"] else 0

    print("\n" + "="*80)
    print(f"{'CONFIGURATION':<40} | {'ENTITY DENSITY'}")
    print("-" * 80)
    print(f"{'A: Baseline (Single)':<40} | {avg_a:>15.2f}")
    print(f"{'C: Multi-Agent (Final Loop)':<40} | {avg_c:>15.2f}")
    print("-" * 80)
    print(f"Total Corrections: {results['Corrections']}")
    if avg_a > 0:
        boost = ((avg_c / avg_a) - 1) * 100
        print(f"Final Density Boost (C vs A): {boost:.1f}%")
    print(f"\n[SUCCESS] Detailed reports saved to: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    if os.path.exists("cached_gpt3_turbo_request_v2.joblib"):
        os.remove("cached_gpt3_turbo_request_v2.joblib")
    run_ablation_study()