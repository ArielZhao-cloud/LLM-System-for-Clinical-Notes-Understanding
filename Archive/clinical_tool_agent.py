import dspy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1. Initializing LLM...")
api_key = os.environ.get("GLM_API_KEY", "your_default_key_here")
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=1500
)
dspy.settings.configure(lm=my_domestic_model)

def check_drug_interaction(drug_name: str) -> str:
    """Check potential side effects, interactions, or guidelines for a specific medication."""
    database = {
        "aspirin": "Caution: May increase bleeding risk when combined with other anticoagulants.",
        "metformin": "Standard first-line treatment for Type 2 Diabetes. Monitor renal function.",
        "lisinopril": "ACE inhibitor used for hypertension. Common side effect: persistent dry cough. Monitor for hyperkalemia."
    }
    
    # 升级：从脆弱的精确匹配改为基于子串的包容性匹配，以适应临床文本(如提取出了 "Lisinopril 10mg")
    query = drug_name.lower()
    for known_drug, info in database.items():
        if known_drug in query:
            return info
    return f"No specific guidelines or side effects found in the database for this drug: {drug_name}."

# 补全 DSPy 严格要求的所有工具属性
check_drug_interaction.name = "check_drug_interaction"
check_drug_interaction.desc = "Check potential side effects, interactions, or guidelines for a specific medication."
check_drug_interaction.input_variable = "drug_name"

def normalize_medical_concept(term: str) -> str:
    """Normalize a medical concept (e.g., drug brand name to generic name, or casual symptom to formal medical term)."""
    # 模拟医学本体映射 (Entity Grounding)，生产环境可接入 RxNorm / SNOMED CT
    ontology_db = {
        "tylenol": "acetaminophen",
        "elevated bp": "hypertension",
        "high blood pressure": "hypertension"
    }
    return ontology_db.get(term.lower(), term.lower())

normalize_medical_concept.name = "normalize_medical_concept"
normalize_medical_concept.desc = "Normalize a medical term (brand name, casual symptom) to standard medical ontology before searching."
normalize_medical_concept.input_variable = "term"

class AnalyzeClinicalNote(dspy.Signature):
    """Analyze a clinical note, extract medical entities, normalize them, and assess risks using tools."""
    clinical_note = dspy.InputField(desc="The raw, messy clinical note from the patient.")
    assessment = dspy.OutputField(desc="Structured clinical assessment including extracted drugs, patient symptoms, and risk analysis based on tool results.")

print("2. Building the ReAct Agent equipped with clinical tools...")
agent = dspy.ReAct(AnalyzeClinicalNote, tools=[normalize_medical_concept, check_drug_interaction])

print("3. Analyzing patient clinical note...")
messy_note = "Pt is a 65yo M presenting with elevated BP. Currently taking Metformin 500mg BID for T2DM and recently started on Lisinopril 10mg daily. Pt complains of a persistent dry cough over the last two weeks."
print(f"Raw Note: {messy_note}\n")

result = agent(clinical_note=messy_note)

print("--- Final Clinical Assessment ---")
print(result.assessment)
print("---------------------------------")

print("\n--- Agent's Internal Thought Process ---")
# Use the language model's built-in history inspector to see the exact prompts and tool calls
my_domestic_model.inspect_history(n=1)