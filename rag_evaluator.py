import os, json, time, dspy, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from rag_pipeline import run_advanced_rag_pipeline
from research_evaluator import ExpertAuditSignature, evaluate_group

load_dotenv()
CLEAN_INPUT_FILE = "oncology_final_reports.json"
OUTPUT_FILE = "oncology_final_reports_advanced_rag.json"

evaluator_llm = dspy.OpenAI(
    model='deepseek-chat', api_key=os.getenv("DEEPSEEK_API_KEY"), 
    api_base='https://api.deepseek.com', max_tokens=250, model_type='chat'
)
dspy.settings.configure(lm=evaluator_llm, cache=False, max_retries=1)

def safe_rag_call(text):
    for attempt in range(5):
        try:
            res = run_advanced_rag_pipeline(text)
            time.sleep(3)
            return res
        except Exception as e:
            wait_time = (attempt + 1) * 6 
            print(f"\n[API Busy] Retrying in {wait_time}s... (Error: {e})")
            time.sleep(wait_time)
    return {"structured_data": {}, "clinical_assessment": "N/A", "treatment_recommendations": "N/A"}

def run_advanced_rag_experiment():
    if not os.path.exists(CLEAN_INPUT_FILE):
        print(f"[ERROR] {CLEAN_INPUT_FILE} not found. Run batch_evaluator.py first.")
        return

    # Load existing progress if available to support resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reports = json.load(f)
    else:
        with open(CLEAN_INPUT_FILE, 'r', encoding='utf-8') as f:
            reports = json.load(f)

    print("\n[START] Generating Group D (Advanced Agentic RAG) reports...")
    
    for i, item in enumerate(tqdm(reports, desc="Processing Advanced RAG Group")):
        if "group_d" in item:
            continue
            
        text = item["original_text"]
        rag_results = safe_rag_call(text)
        item["group_d"] = rag_results
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
            
    print("\n[START] Evaluating all 4 groups (A, B, C, D)...")
    auditor = dspy.Predict(ExpertAuditSignature)
    all_metrics = {"A": [], "B": [], "C": [], "D": []}
    pass_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for i, r in enumerate(reports):
        print(f"Auditing Case {i}...", end=" ", flush=True)
        for group in ["A", "B", "C", "D"]:
            group_key = f"group_{group.lower()}"
            if group_key in r:
                scores = evaluate_group(auditor, r[group_key])
                if scores:
                    all_metrics[group].append(scores)
                    if scores["Completeness"] >= 75.0 and scores["Correctness"] >= 80.0:
                        pass_counts[group] += 1
        print("Done")
        time.sleep(1)

    summaries = {}
    for group in ["A", "B", "C", "D"]:
        if all_metrics[group]:
            df = pd.DataFrame(all_metrics[group])
            summaries[group] = df.mean().to_dict()

    num_cases = len(reports)
    
    print("\n" + "="*95)
    print(f"{'EVALUATION METRIC':<25} | {'Group A':<12} | {'Group B':<12} | {'Group C':<12} | {'Group D (Adv. RAG)':<12}")
    print("-" * 95)
    metrics_list = ['Completeness', 'Correctness', 'Safety', 'Reasoning Clarity', 'Plan Actionability']
    for m in metrics_list:
        print(f"{m:<25} | {summaries['A'].get(m, 0):>11.2f}% | {summaries['B'].get(m, 0):>11.2f}% | {summaries['C'].get(m, 0):>11.2f}% | {summaries['D'].get(m, 0):>11.2f}%")
        
    print("\n" + f"{'Absolute Pass Rate':<25} | {(pass_counts['A']/num_cases)*100:>11.1f}% | {(pass_counts['B']/num_cases)*100:>11.1f}% | {(pass_counts['C']/num_cases)*100:>11.1f}% | {(pass_counts['D']/num_cases)*100:>11.1f}%")
    print("="*95)
    print(f"[SUCCESS] Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_advanced_rag_experiment()