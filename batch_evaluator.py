import os, json, dspy, time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from multi_agent_pipeline import (
    OncologyExtractor, OncologyCritic, ClinicalReasoner, 
    TreatmentPlanner, clean_and_parse_json, PrecisionOncologyTools
)

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

predictor_baseline = dspy.Predict(OncologyExtractor)
predictor_agent = dspy.Predict(OncologyExtractor) 
critic_node = dspy.Predict(OncologyCritic)
reasoner_node = dspy.Predict(ClinicalReasoner)
planner_node = dspy.Predict(TreatmentPlanner)

def calc_density(d):
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
    for attempt in range(5):
        try:
            res = func(**kwargs)
            time.sleep(3) 
            return res
        except Exception as e:
            wait_time = (attempt + 1) * 6 
            print(f"\n[API Busy] Retrying in {wait_time}s... (Error: {e})")
            time.sleep(wait_time)
    return None

def generate_report(data_json, original_text):
    summary = safe_api_call(reasoner_node, original_note=original_text, extracted_data=json.dumps(data_json))
    plan = safe_api_call(planner_node, 
                         clinical_summary=summary.clinical_summary if summary else "N/A", 
                         guideline_context="NCCN 2026 Oncology Guidelines")
    return {
        "assessment": summary.clinical_summary if summary else "N/A",
        "plan": plan.treatment_plan if plan else "N/A"
    }

def run_ablation_study():
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"[ERROR] {RAW_DATASET_PATH} not found.")
        return

    with open(RAW_DATASET_PATH, 'r', encoding='utf-8') as f:
        samples = json.load(f)[:100]

    results = {"Group_A": [], "Group_B": [], "Group_C": [], "Corrections": 0}
    final_reports = []

    print(f"\n[START] Running Final 4-Agent Evaluation (Saving to {OUTPUT_FILE})...")

    for i, item in enumerate(tqdm(samples, desc="Processing Samples")):
        text = item["text"][:2800] 
        report_entry = {"sample_id": i, "original_text": text}
        tool_data = PrecisionOncologyTools.execute_all_tools(text)

        res_a = safe_api_call(predictor_baseline, clinical_note=text, tool_results="None", previous_json="None", previous_feedback="None")
        if res_a:
            data_a = clean_and_parse_json(res_a.extracted_json)
            results["Group_A"].append(calc_density(data_a))
            report_a = generate_report(data_a, text)
        else:
            data_a, report_a = {}, {"assessment": "N/A", "plan": "N/A"}
        
        res_b = safe_api_call(predictor_agent, clinical_note=text, tool_results=tool_data, previous_json="None", previous_feedback="None")
        if res_b:
            data_b = clean_and_parse_json(res_b.extracted_json)
            results["Group_B"].append(calc_density(data_b))
            report_b = generate_report(data_b, text)
        else:
            data_b, report_b = {}, {"assessment": "N/A", "plan": "N/A"}

        start_time = time.time()
        current_js_str = res_b.extracted_json if res_b else "{}"
        
        audit = safe_api_call(critic_node, original_note=text, tool_results=tool_data, extracted_json=current_js_str)
        
        is_corrected = False
        if audit and "Fail" in str(audit.audit_result):
            results["Corrections"] += 1
            is_corrected = True
            refined = safe_api_call(
                predictor_agent, 
                clinical_note=text, 
                tool_results=tool_data, 
                previous_json=current_js_str,
                previous_feedback=audit.feedback
            )
            if refined:
                current_js_str = refined.extracted_json
        
        data_c = clean_and_parse_json(current_js_str)
        results["Group_C"].append(calc_density(data_c))

        report_c = generate_report(data_c, text)

        end_time = time.time() 
        case_latency = end_time - start_time 
        approx_tokens = (len(text) + len(str(data_c)) + len(str(report_c['assessment'])) + len(str(report_c['plan']))) // 4

        report_entry.update({
            "group_a": {
                "structured_data": data_a, 
                "clinical_assessment": report_a['assessment'], 
                "treatment_recommendations": report_a['plan']
            },
            "group_b": {
                "structured_data": data_b, 
                "clinical_assessment": report_b['assessment'], 
                "treatment_recommendations": report_b['plan']
            },
            "group_c": {
                "structured_data": data_c, 
                "clinical_assessment": report_c['assessment'], 
                "treatment_recommendations": report_c['plan']
            },
            "was_corrected": is_corrected,
            "critic_feedback": audit.feedback if is_corrected else "N/A",
            "tool_data_used": tool_data,
            "latency_seconds": case_latency, 
            "approx_token_usage": approx_tokens 
        })
        final_reports.append(report_entry)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_reports, f, indent=2, ensure_ascii=False)

        time.sleep(2)

    avg_a = np.mean(results["Group_A"]) if results["Group_A"] else 0
    avg_b = np.mean(results["Group_B"]) if results["Group_B"] else 0
    avg_c = np.mean(results["Group_C"]) if results["Group_C"] else 0

    print("\n" + "="*80)
    print(f"{'ABLATION STUDY CONFIGURATION':<45} | {'ENTITY DENSITY'}")
    print("-" * 80)
    print(f"{'A: Baseline (LLM Only)':<45} | {avg_a:>15.2f}")
    print(f"{'B: + Multimodal Tools (Single Pass)':<45} | {avg_b:>15.2f}")
    print(f"{'C: + Critic & Reflection (Full 4-Agent)':<45} | {avg_c:>15.2f}")
    print("-" * 80)
    print(f"Total Corrections Triggered: {results['Corrections']} / {len(samples)}")
    
    if avg_a > 0:
        boost_ab = ((avg_b / avg_a) - 1) * 100
        boost_bc = ((avg_c / avg_b) - 1) * 100 if avg_b > 0 else 0
        boost_ac = ((avg_c / avg_a) - 1) * 100
        print(f"\nDensity Boost (A -> B):  +{boost_ab:.1f}%")
        print(f"Density Boost (B -> C):  +{boost_bc:.1f}%")
        print(f"Total Density Boost:     +{boost_ac:.1f}%")
        
    print(f"\n[SUCCESS] Detailed reports saved to: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    if os.path.exists("cached_gpt3_turbo_request_v2.joblib"):
        os.remove("cached_gpt3_turbo_request_v2.joblib")
    run_ablation_study()