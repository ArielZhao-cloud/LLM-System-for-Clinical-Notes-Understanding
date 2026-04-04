import os, json, dspy, time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# 初始化原生大模型
llm = dspy.OpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    api_base='https://api.deepseek.com', 
    max_tokens=2500,
    model_type='chat'
)
dspy.settings.configure(lm=llm, cache=False, max_retries=2)

class BaselineOncology(dspy.Signature):
    """
    Baseline Task: Extract information, assess the patient, and plan treatment based ONLY on the raw clinical note.
    (No external tools or multi-agent corrections are provided).
    """
    clinical_note = dspy.InputField(desc="Raw clinical text.")
    extracted_json = dspy.OutputField(desc="JSON format with diagnosis, biomarkers, treatments.")
    clinical_assessment = dspy.OutputField(desc="Narrative clinical assessment.")
    treatment_plan = dspy.OutputField(desc="Treatment recommendations.")

def run_pure_baseline():
    if not os.path.exists("oncology_raw_samples.json"):
        print("[ERROR] Raw data not found!")
        return

    with open("oncology_raw_samples.json", 'r', encoding='utf-8') as f:
        samples = json.load(f)[:100] # 读取相同的 100 个原始病历
    
    baseline_predictor = dspy.Predict(BaselineOncology)
    baseline_reports = []

    print(f"\n🚀 Running PURE BASELINE (Zero-shot) for {len(samples)} cases...")

    for i, item in enumerate(tqdm(samples)):
        try:
            text = item["text"][:2800]
            # 原生模型单次调用，没有任何上下文工具和纠偏反馈
            res = baseline_predictor(clinical_note=text)
            
            baseline_reports.append({
                "sample_id": i,
                "structured_data": res.extracted_json,
                "clinical_assessment": res.clinical_assessment,
                "treatment_recommendations": res.treatment_plan
            })
            time.sleep(2) # 保护 API
        except Exception as e:
            print(f"\nFailed on sample {i}: {e}")
            continue
            
    # 将基线生成的报告保存为新文件
    with open("baseline_final_reports.json", 'w', encoding='utf-8') as f:
        json.dump(baseline_reports, f, indent=2, ensure_ascii=False)
    print("\n✅ Baseline reports saved to baseline_final_reports.json")

if __name__ == "__main__":
    run_pure_baseline()