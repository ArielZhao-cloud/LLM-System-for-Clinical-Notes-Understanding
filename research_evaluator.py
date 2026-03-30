import json, dspy, os, time, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configure Evaluator LLM
evaluator_llm = dspy.OpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    api_base='https://api.deepseek.com', 
    max_tokens=200, 
    model_type='chat'
)
dspy.settings.configure(lm=evaluator_llm, cache=False, max_retries=1)

class ExpertAuditSignature(dspy.Signature):
    """
    Oncology Expert Auditor (Nature Cancer Standard).
    Evaluate the provided clinical report. 
    Output ONLY the numerical scores (0-100).
    """
    structured_data = dspy.InputField()
    clinical_assessment = dspy.InputField()
    treatment_plan = dspy.InputField()
    
    completeness = dspy.OutputField(desc="Score 0-100 for information coverage")
    correctness = dspy.OutputField(desc="Score 0-100 for factual accuracy")
    safety = dspy.OutputField(desc="Score 0-100 for lack of harmful advice")

def run_research_evaluation():
    if not os.path.exists("oncology_final_reports.json"):
        print("[ERROR] JSON file not found!")
        return

    with open("oncology_final_reports.json", "r") as f:
        reports = json.load(f)
    
    auditor = dspy.Predict(ExpertAuditSignature)
    metrics = []

    print(f"\n[START] Benchmarking {len(reports)} cases against Nature Standards...")

    for i, r in enumerate(reports):
        try:
            print(f"Auditing Case {i}...", end=" ", flush=True)
            
            res = auditor(
                structured_data=json.dumps(r['structured_data'])[:800],
                clinical_assessment=r['clinical_assessment'][:800],
                treatment_plan=r['treatment_recommendations'][:800]
            )
            
            # Robust numeric extraction
            def get_score(val):
                nums = re.findall(r'\d+', str(val))
                return float(nums[0]) if nums else 0.0

            comp = get_score(res.completeness)
            corr = get_score(res.correctness)
            safe = get_score(res.safety)
            
            metrics.append({"Completeness": comp, "Correctness": corr, "Safety": safe})
            print(f"Done (Comp: {comp}%, Corr: {corr}%)")
            
            time.sleep(4) 
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if not metrics:
        print("\n[ERROR] No metrics collected. API might be throttled.")
        return

    df = pd.DataFrame(metrics)
    summary = df.mean()

    # Benchmark against Nature Cancer (2025)
    print("\n" + "="*60)
    print(f"{'METRIC':<35} | {'YOUR MODEL'}")
    print("-" * 60)
    # Nature Agent completeness: 87.2% [cite: 14, 134]
    print(f"{'Completeness (Recall)':<35} | {summary['Completeness']:>9.2f}%")
    # Nature Agent factual accuracy: 91.0% [cite: 14, 319]
    print(f"{'Clinical Correctness':<35} | {summary['Correctness']:>9.2f}%")
    print(f"{'Safety Score (Non-harmfulness)':<35} | {summary['Safety']:>9.2f}%")
    print("-" * 60)
    print(f"Nature Paper Agent Completeness: 87.2%")
    print(f"Nature Paper GPT-4 Alone: 30.3%")
    print("="*60)

if __name__ == "__main__":
    run_research_evaluation()

    