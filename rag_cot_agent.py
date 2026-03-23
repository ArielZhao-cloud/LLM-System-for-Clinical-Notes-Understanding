import dspy
import os
import chromadb

# 屏蔽警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

print("1. Initializing DSPy and Zhipu LLM...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=2000
)
dspy.settings.configure(lm=my_domestic_model)

print("2. Setting up Local Medical Knowledge Base (ChromaDB)...")
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection(name="cardiology_guidelines")
except Exception:
    pass

collection = chroma_client.create_collection(name="cardiology_guidelines")

# 存入一条权威的临床指南作为事实依据
collection.add(
    documents=[
        "For acute ST-segment elevation myocardial infarction (STEMI), immediate percutaneous coronary intervention (PCI) is the standard of care. Post-PCI, it is crucial that patients are prescribed dual antiplatelet therapy (DAPT), typically Aspirin combined with Clopidogrel or Ticagrelor. Additionally, a high-intensity statin (e.g., Atorvastatin 40-80mg) must be initiated to stabilize plaques and reduce future cardiovascular events. Beta-blockers and ACE inhibitors should also be considered if no contraindications exist."
    ],
    metadatas=[{"source": "AHA/ACC STEMI Guidelines 2025"}],
    ids=["guide_01"]
)


# 3. 定义结合了 RAG 资料和 CoT 推理的签名 (Signature)
class ClinicalDecisionMaker(dspy.Signature):
    """Read the patient's clinical note and the retrieved medical guidelines. Use step-by-step reasoning to verify if the clinical interventions match the guidelines, and highlight any missing steps."""

    guideline_context = dspy.InputField(desc="Authoritative medical guidelines retrieved from the database.")
    clinical_note = dspy.InputField(desc="The messy and complex patient clinical discharge note.")

    # 强制输出三个模块，ChainOfThought 会自动在最前面加上 Rationale (推理过程)
    patient_summary = dspy.OutputField(desc="Brief summary of the patient's acute event and interventions.")
    guideline_adherence = dspy.OutputField(desc="Did the hospital treatments match the guidelines? Explain why.")
    missing_recommendations = dspy.OutputField(
        desc="Are there any medications or steps recommended by the guideline that are missing in the discharge note?")


# 4. 把大模型包装成一个同时具备 RAG 和 CoT 能力的模块
class UltimateClinicalAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 ChainOfThought 赋予多步推理能力
        self.analyze = dspy.ChainOfThought(ClinicalDecisionMaker)

    def forward(self, patient_note):
        # 第一步：RAG 检索。用患者笔记去数据库里搜最相关的指南
        print("\n   -> [Agent Action] Searching local database for relevant guidelines...")
        retrieval_results = collection.query(
            query_texts=[patient_note],
            n_results=1
        )
        retrieved_context = retrieval_results['documents'][0][0]
        source = retrieval_results['metadatas'][0][0]['source']
        print(f"   -> [Agent Action] Found relevant guideline from: {source}")

        # 第二步：CoT 推理。把找到的指南和患者笔记一起喂给大模型，让它仔细对比思考
        print("   -> [Agent Action] Analyzing patient case against guidelines using Chain-of-Thought...\n")
        prediction = self.analyze(guideline_context=retrieved_context, clinical_note=patient_note)

        return prediction


print("3. Executing the Ultimate Clinical Agent...")
# 提供一段患者出院小结（故意漏掉了 Beta-blockers 和 ACE inhibitors）
complex_note = """
Patient: John Doe, 72yo M. 
Admission Date: 2026-03-10. Discharge Date: 2026-03-15.
HPI: Pt presented to ED with severe acute chest pain radiating to the left arm and shortness of breath starting 2 hours prior. 
Hospital Course: ECG in ED showed ST elevation in leads V1-V4. Troponin was elevated at 2.5. Pt was rushed to cath lab. Coronary angiography revealed 90% occlusion of the LAD. A drug-eluting stent was successfully placed. Post-op course was uncomplicated. 
Discharge Medications: Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 40mg daily.
Follow-up: Cardiology clinic in 2 weeks.
"""

# 实例化并运行我们的 Agent
agent = UltimateClinicalAgent()
result = agent(patient_note=complex_note)

print("=" * 60)
print("FINAL CLINICAL AUDIT REPORT")
print("=" * 60)
print(f"[Patient Summary]\n{result.patient_summary}\n")
print(f"[Guideline Adherence]\n{result.guideline_adherence}\n")
print(f"[Missing Recommendations]\n{result.missing_recommendations}\n")
print("-" * 60)
print("[Agent's Internal Reasoning (Rationale)]")
print(result.rationale)
print("=" * 60)