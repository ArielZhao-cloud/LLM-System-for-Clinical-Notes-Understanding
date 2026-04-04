import json, dspy, os, time, re
import pandas as pd
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
    CRITICAL INSTRUCTION: You MUST output ONLY a pure integer number between 0 and 100. DO NOT output any words, symbols, explanations, or fractions (e.g. no "85/100", just "85").
    """
    structured_data = dspy.InputField()
    clinical_assessment = dspy.InputField()
    treatment_plan = dspy.InputField()
    
    completeness = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    correctness = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    safety = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    reasoning_clarity = dspy.OutputField(desc="ONLY a single integer from 0 to 100")
    plan_actionability = dspy.OutputField(desc="ONLY a single integer from 0 to 100")

def run_research_evaluation():
    if not os.path.exists("oncology_final_reports.json"):
        print("[ERROR] JSON file not found!")
        return

    #with open("oncology_final_reports.json", "r") as f:
    with open("baseline_final_reports.json", "r") as f:
        reports = json.load(f)
    
    auditor = dspy.Predict(ExpertAuditSignature)
    metrics = []

    print(f"\n[START] Comprehensive Benchmarking for {len(reports)} cases...")

    for i, r in enumerate(reports):
        try:
            print(f"Auditing Case {i}...", end=" ", flush=True)
            res = auditor(
                structured_data=json.dumps(r.get('structured_data', ''))[:800],
                clinical_assessment=r.get('clinical_assessment', '')[:800],
                treatment_plan=r.get('treatment_recommendations', '')[:800]
            )
            
            def get_score(val):
                val_str = str(val)
                nums = re.findall(r'\b(100|[1-9]?[0-9])\b', val_str)
                return float(nums[0]) if nums else 0.0

            comp = get_score(res.completeness)
            corr = get_score(res.correctness)
            safe = get_score(res.safety)
            clarity = get_score(res.reasoning_clarity)
            action = get_score(res.plan_actionability)
            
            # try
            # print(f"\n[Raw Output] Comp: '{res.completeness}', Corr: '{res.correctness}'")
            
            metrics.append({
                "Completeness": comp, "Correctness": corr, 
                "Safety": safe, "Reasoning Clarity": clarity, 
                "Plan Actionability": action
            })
            print(f"Done (Comp: {comp}%, Corr: {corr}%)")
            time.sleep(3) 
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if not metrics:
        print("\n[ERROR] No metrics collected.")
        return

    df = pd.DataFrame(metrics)
    summary = df.mean()

    print("\n" + "="*60)
    print(f"{'EVALUATION METRIC':<35} | {'4-AGENT SYSTEM SCORE'}")
    print("-" * 60)
    print(f"{'Completeness (Recall)':<35} | {summary['Completeness']:>9.2f}%")
    print(f"{'Clinical Correctness':<35} | {summary['Correctness']:>9.2f}%")
    print(f"{'Safety Score (Non-harmfulness)':<35} | {summary['Safety']:>9.2f}%")
    print(f"{'Reasoning Clarity':<35} | {summary['Reasoning Clarity']:>9.2f}%")
    print(f"{'Plan Actionability':<35} | {summary['Plan Actionability']:>9.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_research_evaluation()