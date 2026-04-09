import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import dspy
import chromadb
from dotenv import load_dotenv

# 屏蔽 Chroma 烦人的遥测报错信息
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

load_dotenv()
print("1. Initializing Zhipu LLM...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
api_key = os.getenv("ZHIPU_API_KEY")
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=500
)
dspy.settings.configure(lm=my_domestic_model)

print("2. Initializing Chroma Vector DB...")
chroma_client = chromadb.Client()

# 使用 get_or_create_collection 避免重复运行时的命名冲突报错
collection = chroma_client.get_or_create_collection(name="oncology_guidelines")

collection.add(
    documents=[
        "For unresectable or metastatic melanoma with a BRAF V600E mutation not suitable for local therapy, targeted therapy with Dabrafenib in combination with Trametinib is recommended.",
        "In metastatic colorectal cancer, if the patient's tumor exhibits high microsatellite instability (MSI-H) or mismatch repair deficiency (dMMR), Pembrolizumab is recommended as a first-line immunotherapy option.",
        "For patients with early breast cancer who are hormone receptor (HR) positive and HER2 negative, at least 5 years of adjuvant endocrine therapy, such as Tamoxifen, is generally recommended after surgery."
    ],
    metadatas=[
        {"source": "OncoKB_BRAF_Guideline", "disease": "Melanoma"},
        {"source": "ESMO_CRC_Guideline", "disease": "Colorectal Cancer"},
        {"source": "ASCO_Breast_Guideline", "disease": "Breast Cancer"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

print("3. Querying the patient case...")
query_text = "The patient is diagnosed with metastatic colorectal cancer, and genetic testing shows high microsatellite instability (MSI-H). What are the recommended treatment options?"

# 从数据库中召回最匹配的 Top 1 上下文
retrieval_results = collection.query(
    query_texts=[query_text],
    n_results=1
)
retrieved_context = retrieval_results['documents'][0][0]
source_metadata = retrieval_results['metadatas'][0][0]['source']

print("4. Synthesizing final answer with LLM...")

# 定义结合了上下文 (Context) 的 DSPy 问答模板
class MedicalRAG(dspy.Signature):
    """Answer the medical question based ONLY on the provided context."""
    context = dspy.InputField(desc="Medical guidelines and literature")
    question = dspy.InputField(desc="Patient case or clinical query")
    answer = dspy.OutputField(desc="Treatment recommendation with reasoning")

# 执行预测
predictor = dspy.Predict(MedicalRAG)
response = predictor(context=retrieved_context, question=query_text)

print("\n🎉 Final Clinical Decision:")
print("=" * 50)
print(response.answer)
print("=" * 50)
print(f"Reference Source: {source_metadata}")