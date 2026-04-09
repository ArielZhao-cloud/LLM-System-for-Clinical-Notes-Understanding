import dspy
import os
import chromadb

# 屏蔽警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

print("1. Initializing DSPy and Vector DB...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=2000
)
dspy.settings.configure(lm=my_domestic_model)

# 重建内存知识库
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection(name="cardiology_guidelines")
except Exception:
    pass

collection = chroma_client.create_collection(name="cardiology_guidelines")
collection.add(
    documents=[
        "For acute ST-segment elevation myocardial infarction (STEMI), immediate percutaneous coronary intervention (PCI) is the standard of care. Post-PCI, it is crucial that patients are prescribed dual antiplatelet therapy (DAPT), typically Aspirin combined with Clopidogrel or Ticagrelor. Additionally, a high-intensity statin (e.g., Atorvastatin 40-80mg) must be initiated to stabilize plaques and reduce future cardiovascular events. Beta-blockers and ACE inhibitors should also be considered if no contraindications exist."
    ],
    metadatas=[{"source": "AHA/ACC STEMI Guidelines 2025"}],
    ids=["guide_01"]
)


class ClinicalDecisionMaker(dspy.Signature):
    """Read the patient's clinical note and the retrieved medical guidelines. Verify if clinical interventions match the guidelines."""
    guideline_context = dspy.InputField(desc="Authoritative medical guidelines.")
    clinical_note = dspy.InputField(desc="The patient clinical discharge note.")
    missing_recommendations = dspy.OutputField(
        desc="List ONLY the medications recommended by the guideline that are missing. If none, output 'None'.")


class UltimateClinicalAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ClinicalDecisionMaker)

    def forward(self, clinical_note):
        retrieval_results = collection.query(query_texts=[clinical_note], n_results=1)
        retrieved_context = retrieval_results['documents'][0][0]
        prediction = self.analyze(guideline_context=retrieved_context, clinical_note=clinical_note)
        return prediction


# 构建带有 ID 的测试集
test_dataset = [
    {
        "id": "Case 1 (Missing Beta-blockers)",
        "clinical_note": "STEMI patient. PCI done. Meds: Aspirin 81mg, Clopidogrel 75mg, Atorvastatin 40mg.",
        "expected_missing": "beta-blocker"
    },
    {
        "id": "Case 2 (Perfect Note)",
        "clinical_note": "STEMI patient. PCI done. Meds: Aspirin 81mg, Ticagrelor 90mg, Atorvastatin 80mg, Metoprolol 25mg (Beta-blocker), Lisinopril 10mg (ACE inhibitor).",
        "expected_missing": "none"
    },
    {
        "id": "Case 3 (Patient Refused Meds)",
        "clinical_note": "STEMI patient. PCI done. Meds: Aspirin 81mg, Clopidogrel 75mg, Metoprolol 25mg, Lisinopril 10mg. Patient refused other meds.",
        "expected_missing": "statin"
    }
]


# 自定义判分逻辑
def clinical_audit_metric(expected, predicted):
    expected = expected.lower()
    predicted = predicted.lower()
    if expected == "none":
        return "none" in predicted or "no missing" in predicted or "matches the guideline" in predicted
    return expected in predicted


print("\n2. Starting Detailed Evaluation (Debugging Mode)...\n")
agent = UltimateClinicalAgent()
correct_count = 0

# 遍历测试集，公开打印每一题的输入输出
for i, case in enumerate(test_dataset):
    print("=" * 60)
    print(f"Test: {case['id']}")
    print(f"Input Note: {case['clinical_note']}")
    print(f"Expected Keyword: '{case['expected_missing']}'")
    print("-" * 30)

    # 呼叫 Agent 进行预测
    pred = agent(clinical_note=case['clinical_note'])
    actual_output = pred.missing_recommendations
    rationale = pred.rationale

    print(f"Agent's Output: {actual_output}")

    # 判分并统计
    is_correct = clinical_audit_metric(case['expected_missing'], actual_output)
    if is_correct:
        correct_count += 1
        print("Result: CORRECT")
    else:
        print("Result: INCORRECT!")
        print(f"Exposed Rationale: {rationale}")

print("=" * 60)
print(f"Final Score: {correct_count}/3 ({correct_count / 3 * 100:.2f}%)")