import os
import json
import time
import dspy
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from modular_rag_pipeline import run_modular_rag_pipeline
from research_evaluator import ExpertAuditSignature, evaluate_group

load_dotenv()
BASE_INPUT_FILE = "/Users/haotingzhaooutlook.com/Desktop/T3/oncology_final_reports.json"
ADVANCED_OUTPUT_FILE = "/Users/haotingzhaooutlook.com/Desktop/T3/oncology_final_reports_adv_rag.json"

evaluator_llm = dspy.OpenAI(
    model='deepseek-chat', api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base='https://api.deepseek.com', max_tokens=250, model_type='chat'
)
dspy.settings.configure(lm=evaluator_llm, cache=False, max_retries=1)

def safe_pipeline_execution(text: str) -> dict:
    """Executes the RAG pipeline with exponential backoff for rate limits."""
    for attempt in range(5):
        try:
            res = run_modular_rag_pipeline(text)
            time.sleep(3)
            return res
        except Exception as e:
            wait_time = (attempt + 1) * 6
            print(f"\n[API Rate Limit] Retrying in {wait_time}s... (Error: {e})")
            time.sleep(wait_time)
    return {"structured_data": {}, "clinical_assessment": "N/A", "treatment_recommendations": "N/A"}

def run_advanced_evaluation():
    if not os.path.exists(BASE_INPUT_FILE):
        print(f"[ERROR] {BASE_INPUT_FILE} not found. Base ablation data is required.")
        return

    if os.path.exists(ADVANCED_OUTPUT_FILE):
        with open(ADVANCED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reports = json.load(f)
    else:
        with open(BASE_INPUT_FILE, 'r', encoding='utf-8') as f:
            reports = json.load(f)

    print("\n[START] Generating Modular RAG Reports...")

    for i, item in enumerate(tqdm(reports, desc="Processing Modular RAG")):
        if "group_d_adv" in item:
            continue

        text = item["original_text"]
        rag_results = safe_pipeline_execution(text)
        item["group_d_adv"] = rag_results

        with open(ADVANCED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

    print("\n[START] Cross-Evaluating All Ablation Groups...")
    auditor = dspy.Predict(ExpertAuditSignature)
    all_metrics = {"A": [], "B": [], "C": [], "D_ADV": []}
    pass_counts = {"A": 0, "B": 0, "C": 0, "D_ADV": 0}

    for i, r in enumerate(reports):
        print(f"Auditing Case {i}...", end=" ", flush=True)
        groups = zip(["A", "B", "C", "D_ADV"], ["group_a", "group_b", "group_c", "group_d_adv"])

        for group, json_key in groups:
            if json_key in r:
                scores = evaluate_group(auditor, r[json_key])
                if scores:
                    all_metrics[group].append(scores)
                    if scores["Completeness"] >= 75.0 and scores["Correctness"] >= 80.0:
                        pass_counts[group] += 1
        print("Done")
        time.sleep(1)

    summaries = {}
    for group in ["A", "B", "C", "D_ADV"]:
        if all_metrics[group]:
            df = pd.DataFrame(all_metrics[group])
            summaries[group] = df.mean().to_dict()

    num_cases = len(reports)

    print("\n" + "="*105)
    print(f"{'EVALUATION METRIC':<25} | {'Group A':<12} | {'Group B':<12} | {'Group C':<12} | {'Group D (Adv. RAG)':<18}")
    print("-" * 105)
    metrics_list = ['Completeness', 'Correctness', 'Safety', 'Reasoning Clarity', 'Plan Actionability']
    for m in metrics_list:
        print(f"{m:<25} | {summaries['A'].get(m, 0):>11.2f}% | {summaries['B'].get(m, 0):>11.2f}% | {summaries['C'].get(m, 0):>11.2f}% | {summaries['D_ADV'].get(m, 0):>17.2f}%")

    print("\n" + f"{'Absolute Pass Rate':<25} | {(pass_counts['A']/num_cases)*100:>11.1f}% | {(pass_counts['B']/num_cases)*100:>11.1f}% | {(pass_counts['C']/num_cases)*100:>11.1f}% | {(pass_counts['D_ADV']/num_cases)*100:>17.1f}%")
    print("="*105)
    print(f"[SUCCESS] Advanced ablation data saved to {ADVANCED_OUTPUT_FILE}")

if __name__ == "__main__":
    run_advanced_evaluation()
