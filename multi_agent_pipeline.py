import os
import dspy
import json
import re
import chromadb
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END

# ==========================================
# 0. 辅助函数: JSON 清洗与解析
# ==========================================
def clean_and_parse_json(raw_output: str) -> Dict[str, Any]:
    cleaned_text = re.sub(r'```json\n?', '', raw_output)
    cleaned_text = re.sub(r'```\n?', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    try:
        parsed_json = json.loads(cleaned_text)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"[Error] JSON parsing failed: {e}")
        return {"error": "Failed to parse JSON", "raw_content": raw_output}

# ==========================================
# 1. 环境与大模型初始化
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("System initializing: Loading Zhipu GLM-4 model...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=4096
)
dspy.settings.configure(lm=my_domestic_model)

# ==========================================
# 2. 定义系统状态
# ==========================================
class ClinicalNoteState(TypedDict):
    original_text: str
    extracted_entities: str
    current_summary: str
    critic_feedback: str
    revision_count: int
    status: str
    retrieved_guidelines: str
    final_treatment_plan: str

# ==========================================
# 3. 定义智能体大脑 (DSPy Signatures & Modules)
# ==========================================
class ExtractEntities(dspy.Signature):
    """You are a professional clinical data extraction expert. Extract information strictly following the JSON schema below.

    REQUIRED JSON SCHEMA:
    {
      "patient_demographics": { "age": "integer or null", "gender": "M or F or null" },
      "chief_complaint": "Brief summary of the main reason for visit",
      "medical_history": [ { "condition": "disease name", "status": "Present or Denied", "source_quote": "Exact words from the text" } ],
      "allergies": [ { "allergen": "substance name", "source_quote": "Exact words from the text" } ],
      "symptoms": [ { "name": "symptom name", "status": "Present or Absent", "source_quote": "Exact words from the text" } ],
      "diagnoses": [ { "condition": "disease name", "certainty": "Confirmed or Suspected or Ruled_Out", "source_quote": "Exact words from the text" } ],
      "medications": [ { "name": "drug name", "status": "Current or Past", "source_quote": "Exact words from the text" } ],
      "procedures": [ { "name": "procedure or surgery name", "status": "Performed or Planned", "source_quote": "Exact words from the text" } ]
    }

    RULES:
    1. Zero Hallucination: Only extract what is explicitly stated.
    2. NO MATH: DO NOT calculate age from Date of Birth. If age is not explicitly written (e.g., '55yo'), use null.
    3. IGNORE GENERAL ROS: Do not extract massive lists of denied symptoms from the general 'Review of Systems' boilerplate. Focus only on symptoms relevant to the Chief Complaint and HPI.
    4. Traceability: `source_quote` MUST be an exact substring from the clinical_note.
    5. Output ONLY valid JSON.
    """
    clinical_note = dspy.InputField(desc="The raw clinical note.")
    previous_feedback = dspy.InputField(desc="Feedback from the critic. If 'None', ignore.")
    extracted_json = dspy.OutputField(desc="Strictly formatted JSON matching the REQUIRED JSON SCHEMA.")

class ClinicalExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractEntities)

    def forward(self, clinical_note, previous_feedback="None"):
        return self.extract(clinical_note=clinical_note, previous_feedback=previous_feedback)

class SummarizeNode(dspy.Signature):
    """You are a strict medical scribe. Your task is to generate a SOAP note based EXCLUSIVELY on the provided extracted_json.

    RULES:
    1. DO NOT output or echo the input JSON data. Output ONLY the final SOAP note text.
    2. ZERO HALLUCINATION: Do not invent vital signs, physical exam findings, or test results.
    3. NO EXTRAPOLATION IN PLAN: For the 'Plan' section, strictly state only what is explicitly documented. DO NOT suggest referrals or general medical advice.
    """
    extracted_json = dspy.InputField(desc="Structured medical entities.")
    summary = dspy.OutputField(desc="Coherent SOAP note text. No JSON output.")

class CriticizeExtraction(dspy.Signature):
    """You are a pragmatic clinical auditor. Compare the original_note with the extracted_json.

    AUDIT PROTOCOL:
    1. Fabrication Check: Are all entities actually mentioned in the text?
    2. Attribute Check: Did it extract 'Ruled_Out' as 'Confirmed'?
    3. Traceability Check: Does the `source_quote` exactly exist in the original text?

    CRITICAL INSTRUCTIONS FOR AUDITING MIMIC-IV DATA:
    - DO NOT penalize the extractor for failing to calculate age from Date of Birth.
    - DO NOT penalize the extractor for ignoring long lists of denied symptoms in the 'Review of Systems' (ROS).
    - Focus ONLY on critical hallucinations or dangerous attribute flips.

    Output 'Pass' if accurate. Output 'Fail' if dangerous errors are found.
    """
    original_note = dspy.InputField(desc="The absolute truth: original clinical note.")
    extracted_json = dspy.InputField(desc="The extraction results to be reviewed.")
    audit_result = dspy.OutputField(desc="Output 'Pass' if accurate, output 'Fail' if dangerous errors or fabrications are found.")
    feedback = dspy.OutputField(desc="If 'Fail', detail the exact error and correction instructions; if 'Pass', output 'None'.")

class ClinicalDecisionMaker(dspy.Signature):
    """You are a Senior Attending Physician. Evaluate the patient's clinical summary against the provided medical guidelines.

    RULES:
    1. Compare the patient's current treatments/medications with the guideline_context.
    2. Identify if the current treatment is compliant with the guidelines.
    3. Clearly state any MISSING recommendations or medications that should be prescribed according to the guidelines.
    """
    clinical_summary = dspy.InputField(desc="The objective SOAP note summary of the patient.")
    guideline_context = dspy.InputField(desc="Relevant medical guidelines retrieved from the database.")
    compliance_analysis = dspy.OutputField(desc="Analysis of whether current treatment matches guidelines.")
    missing_recommendations = dspy.OutputField(desc="List of missing medications or steps based ONLY on the guidelines.")

# ==========================================
# 4. 初始化组件 (DSPy 优化权重 & ChromaDB)
# ==========================================
optimized_extractor = ClinicalExtractor()
model_path = "/Users/haotingzhaooutlook.com/Desktop/T3/optimized_extractor.json"

if os.path.exists(model_path):
    try:
        optimized_extractor.load(model_path)
        print(f"[Info] Successfully loaded optimized Extractor weights.")
    except Exception as e:
        print(f"[Warning] Failed to load optimized weights: {e}. Using baseline.")
else:
    print("[Warning] Optimized weights not found. Using baseline model.")

print("Connecting to ChromaDB for RAG...")
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="medical_guidelines")
    USE_RAG = True
    print("[Info] Successfully connected to ChromaDB.")
except Exception as e:
    print(f"[Warning] ChromaDB connection failed: {e}. RAG will be bypassed.")
    USE_RAG = False

# ==========================================
# 5. 定义工作流节点 (LangGraph Nodes)
# ==========================================
def extraction_node(state: ClinicalNoteState) -> Dict[str, Any]:
    print("\n[Node 2: Extractor] Calling Optimized GLM-4 to extract entities...")
    prediction = optimized_extractor(
        clinical_note=state["original_text"],
        previous_feedback=state.get("critic_feedback", "None")
    )
    parsed_data = clean_and_parse_json(prediction.extracted_json)
    standardized_json_string = json.dumps(parsed_data, ensure_ascii=False, indent=2)
    return {"extracted_entities": standardized_json_string}

def reasoning_node(state: ClinicalNoteState) -> Dict[str, Any]:
    print("\n[Node 3: Reasoner] Generating objective SOAP summary...")
    summarizer = dspy.Predict(SummarizeNode)
    result = summarizer(extracted_json=state["extracted_entities"])
    return {"current_summary": result.summary}

def critic_node(state: ClinicalNoteState) -> Dict[str, Any]:
    print("\n[Node 4: Critic] Auditing for clinical hallucinations...")
    rev_count = state.get("revision_count", 0)

    if rev_count >= 2:
        print("  -> WARNING: Maximum retry count (2) reached. Forcing approval.")
        return {"status": "approved"}

    critic = dspy.Predict(CriticizeExtraction)
    result = critic(
        original_note=state["original_text"],
        extracted_json=state["extracted_entities"]
    )

    if "Fail" in result.audit_result:
        print(f"  -> ALERT: Audit failed! {result.feedback}")
        return {
            "critic_feedback": result.feedback,
            "revision_count": rev_count + 1,
            "status": "needs_revision"
        }
    else:
        print("  -> SUCCESS: Audit passed!")
        return {
            "status": "approved",
            "critic_feedback": "None"
        }

def decision_node(state: ClinicalNoteState) -> Dict[str, Any]:
    print("\n[Node 5: Decision Maker (RAG)] Retrieving guidelines and generating treatment plan...")

    summary = state["current_summary"]
    entities_json_str = state["extracted_entities"]
    retrieved_text = "No guidelines available or database disconnected."

    if USE_RAG:
        try:
            entities_dict = json.loads(entities_json_str)
            diagnoses_list = entities_dict.get("diagnoses", [])

            core_conditions = []
            for d in diagnoses_list:
                if d.get("certainty") in ["Confirmed", "Present", "Suspected"]:
                    core_conditions.append(d.get("condition", ""))

            if not core_conditions:
                chief_complaint = entities_dict.get("chief_complaint", "")
                search_query = chief_complaint if chief_complaint else "General medical guidelines"
            else:
                search_query = " ".join(core_conditions)

            search_query = f"Clinical guidelines and treatment management for {search_query}"
            print(f"  -> Optimized Search Query: '{search_query}'")

            results = collection.query(query_texts=[search_query], n_results=1)

            if results and results['documents'] and results['documents'][0]:
                retrieved_text = results['documents'][0][0]

                # 核心创新：提取我们在 ingest_pdf 阶段注入的元数据 (Metadata)
                metadata = results['metadatas'][0][0] if results.get('metadatas') and results['metadatas'][0] else {}
                source_doc = metadata.get('source', 'Unknown Document')
                page_num = metadata.get('page', 'Unknown Page')

                # 将权威的页码溯源信息附在指南后面，传递给前端
                retrieved_text += f"\n\n**(Source: {source_doc}, Page: {page_num})**"
                print(f"  -> Retrieved relevant guidelines successfully from Page {page_num}.")

        except Exception as e:
            print(f"  -> Retrieval error: {e}")

    # 调用 DSPy 决策专家进行指南比对
    decision_maker = dspy.ChainOfThought(ClinicalDecisionMaker)
    result = decision_maker(clinical_summary=summary, guideline_context=retrieved_text)

    # ==========================================
    # 核心排版优化：强制清洗与 Markdown 格式化
    # ==========================================
    # 1. 清理合规性分析 (去除自带的标题，并强行截断溢出的 Missing 部分)
    raw_comp = result.compliance_analysis
    raw_comp = re.sub(r"^Compliance Analysis:\s*", "", raw_comp, flags=re.IGNORECASE).strip()

    # 如果大模型在第一个字段里偷跑了 Missing Recommendations，直接切掉后面的内容
    if "Missing Recommendations:" in raw_comp:
        raw_comp = raw_comp.split("Missing Recommendations:")[0].strip()
    if "Missing recommendations:" in raw_comp:
        raw_comp = raw_comp.split("Missing recommendations:")[0].strip()

    # 2. 清理缺失建议 (去除机器口吻和重复标题)
    raw_missing = result.missing_recommendations
    raw_missing = re.sub(r"^Missing Recommendations:\s*", "", raw_missing, flags=re.IGNORECASE).strip()
    raw_missing = raw_missing.replace("produce the missing_recommendations.", "")
    raw_missing = raw_missing.replace("produce the missing_recommendations", "").strip()

    # 3. 使用纯净的 Markdown 语法进行高级排版
    final_plan = (
        "### 1. Guideline Compliance Analysis\n"
        f"{raw_comp}\n\n"
        "---\n\n"
        "### 2. Missing Recommendations & Clinical Rationale\n"
        f"{raw_missing}"
    )

    return {
        "retrieved_guidelines": retrieved_text,
        "final_treatment_plan": final_plan
    }

# ==========================================
# 6. 定义条件路由与组装流水线 (加入 HITL 断点)
# ==========================================
from langgraph.checkpoint.memory import MemorySaver # 新增：引入记忆存储

def routing_logic(state: ClinicalNoteState) -> str:
    if state["status"] == "needs_revision":
        return "revise"
    return "approved"

workflow = StateGraph(ClinicalNoteState)

workflow.add_node("Extractor", extraction_node)
workflow.add_node("Reasoner", reasoning_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("DecisionMaker", decision_node)

workflow.add_edge(START, "Extractor")
workflow.add_edge("Extractor", "Critic") # 注意：为了让人类审核最纯粹的提取结果，我们可以让 Extractor 直接到 Critic
# 修正你之前的逻辑流转：Extractor -> Critic 进行内审
workflow.add_edge("Extractor", "Critic")

# 重新梳理正确的边逻辑：
workflow = StateGraph(ClinicalNoteState)
workflow.add_node("Extractor", extraction_node)
workflow.add_node("Reasoner", reasoning_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("DecisionMaker", decision_node)

workflow.add_edge(START, "Extractor")
workflow.add_edge("Extractor", "Critic")

# 条件路由：Critic 打回给 Extractor，或者通过后走向 Reasoner
workflow.add_conditional_edges(
    "Critic",
    routing_logic,
    {
        "revise": "Extractor",
        "approved": "Reasoner" # 内部审计通过后，准备进入总结和决策阶段
    }
)
workflow.add_edge("Reasoner", "DecisionMaker")
workflow.add_edge("DecisionMaker", END)

# 初始化内存保存器
memory = MemorySaver()

# 核心创新：在 Reasoner 节点前设置中断，等待人类医生审核 JSON
clinical_system = workflow.compile(
    checkpointer=memory,
    interrupt_before=["Reasoner"]
)

# ==========================================
# 7. 执行入口 (Main)
# ==========================================
if __name__ == "__main__":
    print("\n========== Starting Multi-Agent System (Full RAG Pipeline) ==========")

    test_note = "Patient is a 65yo M presenting with acute ST-segment elevation myocardial infarction (STEMI). PCI was performed immediately. Currently prescribed Aspirin and Clopidogrel. No other medications noted."

    initial_state = {
        "original_text": test_note,
        "revision_count": 0,
        "status": "processing",
        "critic_feedback": "None"
    }

    final_state = clinical_system.invoke(initial_state)

    print("\n========== Final Output ==========")
    print("\n[Objective SOAP Summary]\n", final_state['current_summary'])
    print("\n[Guideline-Grounded Treatment Plan]\n", final_state['final_treatment_plan'])