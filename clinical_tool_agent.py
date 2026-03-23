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

def check_drug_interaction(drug_name: str) -> str:
    """Check potential side effects, interactions, or guidelines for a specific medication."""
    database = {
        "aspirin": "Caution: May increase bleeding risk when combined with other anticoagulants.",
        "metformin": "Standard first-line treatment for Type 2 Diabetes. Monitor renal function.",
        "lisinopril": "ACE inhibitor used for hypertension. Common side effect: persistent dry cough. Monitor for hyperkalemia."
    }
    return database.get(drug_name.lower(), "No specific guidelines or side effects found in the database for this drug.")

# 补全 DSPy 严格要求的所有工具属性
check_drug_interaction.name = "check_drug_interaction"
check_drug_interaction.desc = "Check potential side effects, interactions, or guidelines for a specific medication."
check_drug_interaction.input_variable = "drug_name"

class AnalyzeClinicalNote(dspy.Signature):
    """Analyze a clinical note, extract medications, and assess risks using the provided drug database tool."""
    clinical_note = dspy.InputField(desc="The raw, messy clinical note from the patient.")
    assessment = dspy.OutputField(desc="Structured clinical assessment including extracted drugs, patient symptoms, and risk analysis based on tool results.")

print("2. Building the ReAct Agent equipped with clinical tools...")
agent = dspy.ReAct(AnalyzeClinicalNote, tools=[check_drug_interaction])

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