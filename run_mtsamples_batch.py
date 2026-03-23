import os
import dspy
import chromadb
import pandas as pd
import kagglehub

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

print("1. Loading MTSamples dataset...")
path = kagglehub.dataset_download("atharvakaushik/mtsamples")
csv_file = os.path.join(path, "mtsamples.csv")

print("2. Initializing LLM and Knowledge Base...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=2000
)
dspy.settings.configure(lm=my_domestic_model)

chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection(name="cardiology_guidelines")
except Exception:
    pass
collection = chroma_client.create_collection(name="cardiology_guidelines")

# 放入两条不同的指南，测试模型的鉴别能力
collection.add(
    documents=[
        "For acute ST-segment elevation myocardial infarction (STEMI), immediate percutaneous coronary intervention (PCI) is the standard of care. Post-PCI, patients must be prescribed DAPT (Aspirin + Clopidogrel), high-intensity statin, Beta-blockers, and ACE inhibitors.",
        "For routine management of essential hypertension, first-line therapy includes Thiazide diuretics, Calcium channel blockers (e.g., Amlodipine), or ACE inhibitors. Target blood pressure is <130/80 mmHg."
    ],
    metadatas=[{"source": "STEMI Guidelines"}, {"source": "Hypertension Guidelines"}],
    ids=["guide_stemi", "guide_htn"]
)


# 核心修改：增加前置拦截器 (Triage)
class ClinicalDecisionMaker(dspy.Signature):
    """Read the patient's clinical note and the retrieved medical guideline. FIRST, determine if the patient actually has the condition described in the guideline. IF YES, audit the medications. IF NO, state that it's not applicable."""

    guideline_context = dspy.InputField(desc="Retrieved medical guideline.")
    clinical_note = dspy.InputField(desc="Patient clinical note.")

    is_guideline_applicable = dspy.OutputField(
        desc="Does the patient have the condition mentioned in the guideline? Answer Yes or No, and briefly explain why.")
    missing_recommendations = dspy.OutputField(
        desc="If applicable, list missing medications. If NOT applicable, output 'N/A - Guideline does not apply'.")


class UltimateClinicalAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ClinicalDecisionMaker)

    def forward(self, clinical_note):
        retrieval_results = collection.query(query_texts=[clinical_note], n_results=1)
        retrieved_context = retrieval_results['documents'][0][0]
        prediction = self.analyze(guideline_context=retrieved_context, clinical_note=clinical_note)
        prediction.retrieved_source = retrieval_results['metadatas'][0][0]['source']
        return prediction


print("\n3. Processing Real Clinical Notes (With Triage Logic)...\n")
df = pd.read_csv(csv_file)
cardio_notes = df[df['medical_specialty'].str.contains('Cardiovascular', na=False, case=False)]
cardio_transcriptions = cardio_notes['transcription'].dropna().tolist()

agent = UltimateClinicalAgent()

for i, note in enumerate(cardio_transcriptions[:3]):
    print("=" * 70)
    print(f"--- Processing Patient Case {i + 1} ---")
    preview_note = note[:300] + "..." if len(note) > 300 else note
    print(f"[Raw Note Preview]:\n{preview_note}\n")

    try:
        result = agent(clinical_note=note)
        print(f"[Retrieved Guideline]: {result.retrieved_source}")
        print(f"[Applicability Check]:\n{result.is_guideline_applicable}\n")
        print(f"[Audit - Missing Meds]:\n{result.missing_recommendations}")
    except Exception as e:
        print(f"Error processing Case {i + 1}: {e}")

print("=" * 70)