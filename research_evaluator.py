import json, dspy, os, time, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

evaluator_llm = dspy.OpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    api_base='https://api.deepseek.com', 
    max_tokens=250, 
    model_type='chat'
)
dspy.settings.configure(lm=evaluator_llm, cache=False, max_retries=1)

class ExpertAuditSignature(dspy.Signature):
    """
    Oncology Expert Auditor. Evaluate the clinical report.
    CRITICAL INSTRUCTION: Output ONLY a pure integer number between 0 and 100. DO NOT output any words, symbols, explanations, or fractions.
    """
    structured_data = dspy.InputField()
    clinical_assessment = dspy.InputField()
    treatment_plan = dspy.InputField()
    
    completeness = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    correctness = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    safety = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    reasoning_clarity = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    plan_actionability = dspy.OutputField(desc="ONLY a single integer from 0 to 100")

def evaluate_group(auditor, group_data):
    try:
        res = auditor(
            structured_data=json.dumps(group_data.get('structured_data', ''))[:800],
            clinical_assessment=group_data.get('clinical_assessment', '')[:800],
            treatment_plan=group_data.get('treatment_recommendations', '')[:800]
        )
        
        def get_score(val):
            nums = re.findall(r'\b(100|[1-9]?[0-9])\b', str(val))
            return float(nums[0]) if nums else 0.0

        comp = get_score(res.completeness)
        corr = get_score(res.correctness)
        safe = get_score(res.safety)
        clarity = get_score(res.reasoning_clarity)
        action = get_score(res.plan_actionability)
        
        if comp == 0 and corr == 0 and safe == 0:
            return None
            
        return {
            "Completeness": comp, 
            "Correctness": corr, 
            "Safety": safe, 
            "Reasoning Clarity": clarity, 
            "Plan Actionability": action
        }
    except Exception as e:
        print(f"  [Error]: {e}")
        return None

def run_research_evaluation():
    file_path = "oncology_final_reports.json" 
    if not os.path.exists(file_path):
        print(f"[ERROR] {file_path} not found.")
        return

    with open(file_path, "r") as f:
        reports = json.load(f)
    
    auditor = dspy.Predict(ExpertAuditSignature)
    
    all_metrics = {"A": [], "B": [], "C": []}
    comp_scores = {"A": [], "B": [], "C": []}
    corr_scores = {"A": [], "B": [], "C": []}
    pass_counts = {"A": 0, "B": 0, "C": 0}

    print(f"\n[START] Benchmarking Ablation Groups (A, B, C) for {len(reports)} cases...")

    for i, r in enumerate(reports):
        print(f"Auditing Case {i}...", end=" ", flush=True)
        
        for group in ["A", "B", "C"]:
            group_key = f"group_{group.lower()}"
            if group_key in r:
                scores = evaluate_group(auditor, r[group_key])
                if scores:
                    all_metrics[group].append(scores)
                    comp_scores[group].append(scores["Completeness"])
                    corr_scores[group].append(scores["Correctness"])
                    if scores["Completeness"] >= 75.0 and scores["Correctness"] >= 80.0:
                        pass_counts[group] += 1
                        
        print("Done")
        time.sleep(1) 

    summaries = {}
    for group in ["A", "B", "C"]:
        if all_metrics[group]:
            df = pd.DataFrame(all_metrics[group])
            summaries[group] = df.mean().to_dict()
        else:
            summaries[group] = {
                "Completeness": 0, "Correctness": 0, "Safety": 0, 
                "Reasoning Clarity": 0, "Plan Actionability": 0
            }

    num_cases = len(reports)
    
    print("\n" + "="*80)
    print(f"{'EVALUATION METRIC':<35} | {'Group A':<10} | {'Group B':<10} | {'Group C':<10}")
    print("-" * 80)
    print("--- 1. Clinical Capability (Mean Scores) ---")
    
    metrics_list = ['Completeness', 'Correctness', 'Safety', 'Reasoning Clarity', 'Plan Actionability']
    for m in metrics_list:
        print(f"{m:<35} | {summaries['A'][m]:>9.2f}% | {summaries['B'][m]:>9.2f}% | {summaries['C'][m]:>9.2f}%")
        
    print("\n--- 2. System Reliability ---")
    pass_rate_a = (pass_counts['A']/num_cases)*100 if num_cases > 0 else 0
    pass_rate_b = (pass_counts['B']/num_cases)*100 if num_cases > 0 else 0
    pass_rate_c = (pass_counts['C']/num_cases)*100 if num_cases > 0 else 0
    
    print(f"{'Absolute Pass Rate':<35} | {pass_rate_a:>9.1f}% | {pass_rate_b:>9.1f}% | {pass_rate_c:>9.1f}%")
    
    std_comp_a = np.std(comp_scores['A']) if comp_scores['A'] else 0
    std_comp_b = np.std(comp_scores['B']) if comp_scores['B'] else 0
    std_comp_c = np.std(comp_scores['C']) if comp_scores['C'] else 0
    print(f"{'Completeness Std Dev':<35} | ±{std_comp_a:>8.2f} | ±{std_comp_b:>8.2f} | ±{std_comp_c:>8.2f}")
    
    std_corr_a = np.std(corr_scores['A']) if corr_scores['A'] else 0
    std_corr_b = np.std(corr_scores['B']) if corr_scores['B'] else 0
    std_corr_c = np.std(corr_scores['C']) if corr_scores['C'] else 0
    print(f"{'Correctness Std Dev':<35} | ±{std_corr_a:>8.2f} | ±{std_corr_b:>8.2f} | ±{std_corr_c:>8.2f}")
    print("="*80)

if __name__ == "__main__":
    run_research_evaluation()