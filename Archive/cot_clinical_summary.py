import dspy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1. Initializing LLM...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=1500
)
dspy.settings.configure(lm=my_domestic_model)


# 1. 定义多步推理的输入输出结构
class ClinicalSummary(dspy.Signature):
    """Read the complex clinical discharge summary and extract key temporal events step by step."""

    discharge_note = dspy.InputField(desc="The full clinical discharge note.")

    # 强制模型输出中间推理过程的几个关键维度
    initial_presentation = dspy.OutputField(desc="What were the patient's symptoms upon admission?")
    interventions = dspy.OutputField(desc="What major tests or treatments were administered during the stay?")
    final_diagnosis = dspy.OutputField(desc="What is the final diagnosis and discharge plan?")


# 2. 使用 ChainOfThought 模块包裹我们的签名
# 这会让模型自动生成 "Rationale" (推理过程)，然后再输出最终的结构化字段
cot_summarizer = dspy.ChainOfThought(ClinicalSummary)

# 3. 提供一段包含时间线和复杂信息的模拟出院小结
complex_note = """
Patient: John Doe, 72yo M. 
Admission Date: 2026-03-10. Discharge Date: 2026-03-15.
HPI: Pt presented to ED with severe acute chest pain radiating to the left arm and shortness of breath starting 2 hours prior. 
Hospital Course: ECG in ED showed ST elevation in leads V1-V4. Troponin was elevated at 2.5. Pt was rushed to cath lab. Coronary angiography revealed 90% occlusion of the LAD. A drug-eluting stent was successfully placed. Post-op course was uncomplicated. 
Discharge Medications: Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 40mg daily.
Follow-up: Cardiology clinic in 2 weeks.
"""

print("2. Starting Chain-of-Thought analysis...")
print(f"Reading note for patient...\n")

# 4. 执行推理
result = cot_summarizer(discharge_note=complex_note)

print("--- Structured Clinical Summary ---")
print(f"1. Initial Presentation:\n{result.initial_presentation}\n")
print(f"2. Interventions:\n{result.interventions}\n")
print(f"3. Final Diagnosis & Plan:\n{result.final_diagnosis}\n")
print("-----------------------------------")

print("\n--- How the model reasoned (Rationale) ---")
print(result.rationale)