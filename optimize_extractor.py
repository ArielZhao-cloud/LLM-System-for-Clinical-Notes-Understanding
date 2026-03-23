import os
import json
import re
import dspy
from dspy.teleprompt import BootstrapFewShot

# ==========================================
# 1. Configuration & DSPy Setup
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Initializing DSPy Compiler with Zhipu GLM-4...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
glm4_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=4096
)
dspy.settings.configure(lm=glm4_model)

DATA_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/pseudo_ground_truth.json"
OUTPUT_MODEL_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/optimized_extractor.json"


# ==========================================
# 2. Signature & Module Definition
# (Must match the one in multi_agent_pipeline.py)
# ==========================================
class ExtractEntities(dspy.Signature):
    """You are a professional clinical data extraction expert. Extract information strictly following the JSON schema below.

    REQUIRED JSON SCHEMA:
    {
      "patient_demographics": { "age": "integer or null", "gender": "M or F or null" },
      "chief_complaint": "string",
      "medical_history": [ { "condition": "string", "status": "Present or Denied", "source_quote": "string" } ],
      "allergies": [ { "allergen": "string", "source_quote": "string" } ],
      "symptoms": [ { "name": "string", "status": "Present or Absent", "source_quote": "string" } ],
      "diagnoses": [ { "condition": "string", "certainty": "Confirmed or Suspected or Ruled_Out", "source_quote": "string" } ],
      "medications": [ { "name": "string", "status": "Current or Past", "source_quote": "string" } ],
      "procedures": [ { "name": "string", "status": "Performed or Planned", "source_quote": "string" } ]
    }

    RULES:
    1. Zero Hallucination: Only extract what is explicitly stated.
    2. NO MATH: DO NOT calculate age from Date of Birth. If age is not explicitly written, use null.
    3. IGNORE GENERAL ROS: Do not extract massive lists of denied symptoms from the general 'Review of Systems' boilerplate.
    4. Traceability: `source_quote` MUST be an exact substring from the clinical_note.
    5. Output ONLY valid JSON.
    """
    clinical_note = dspy.InputField(desc="The raw clinical note.")
    previous_feedback = dspy.InputField(desc="Feedback from the critic. If 'None', ignore.")
    extracted_json = dspy.OutputField(desc="Strictly formatted JSON matching the REQUIRED JSON SCHEMA.")


class ClinicalExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using ChainOfThought allows the model to reason before outputting JSON
        self.extract = dspy.ChainOfThought(ExtractEntities)

    def forward(self, clinical_note):
        # We optimize the first-pass extraction (no feedback yet)
        return self.extract(clinical_note=clinical_note, previous_feedback="None")


# ==========================================
# 3. Data Loading & Trainset Preparation
# ==========================================
def load_and_prepare_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    dataset = []
    for item in raw_data:
        # Create a dspy.Example with inputs and expected labels
        ex = dspy.Example(
            clinical_note=item["original_text"],
            golden_labels=item["golden_labels"]  # The perfect JSON we generated
        ).with_inputs("clinical_note")
        dataset.append(ex)

    print(f"Loaded {len(dataset)} examples.")
    return dataset


# ==========================================
# 4. The Metric Function (The strict grader)
# ==========================================
def extraction_metric(example, pred, trace=None):
    """
    Compares the predicted JSON against the golden_labels JSON.
    Calculates an F1-like score based on diagnoses, symptoms, medications, and procedures.
    """
    try:
        # Clean and parse the predicted JSON string
        raw_pred = pred.extracted_json
        cleaned = re.sub(r'```json\n?', '', raw_pred)
        cleaned = re.sub(r'```\n?', '', cleaned)
        pred_dict = json.loads(cleaned.strip())
    except Exception as e:
        return 0.0  # Strict failure if JSON is malformed

    truth_dict = example.golden_labels

    def compare_entity_lists(pred_list, truth_list, key_name):
        if not truth_list and not pred_list: return 1.0
        if not truth_list and pred_list: return 0.0  # Penalize hallucination
        if truth_list and not pred_list: return 0.0  # Penalize missing

        # Extract the target string (e.g., 'condition' name) and lowercase it
        truth_items = set(str(item.get(key_name, '')).lower().strip() for item in truth_list if item.get(key_name))
        pred_items = set(str(item.get(key_name, '')).lower().strip() for item in pred_list if item.get(key_name))

        if not truth_items: return 1.0

        intersection = truth_items.intersection(pred_items)
        if not intersection: return 0.0

        # Calculate F1 Score for this specific entity type
        precision = len(intersection) / len(pred_items)
        recall = len(intersection) / len(truth_items)
        if precision + recall == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # Score the FOUR most critical medical components
    diag_score = compare_entity_lists(pred_dict.get("diagnoses", []), truth_dict.get("diagnoses", []), "condition")
    symp_score = compare_entity_lists(pred_dict.get("symptoms", []), truth_dict.get("symptoms", []), "name")
    med_score = compare_entity_lists(pred_dict.get("medications", []), truth_dict.get("medications", []), "name")

    # 新增对 procedures (手术/操作) 的评分逻辑
    proc_score = compare_entity_lists(pred_dict.get("procedures", []), truth_dict.get("procedures", []), "name")

    # Final score is the average of the FOUR F1 scores
    return (diag_score + symp_score + med_score + proc_score) / 4.0


# ==========================================
# 5. Execute DSPy Optimization
# ==========================================
def run_optimization():
    dataset = load_and_prepare_data()

    # Split into train (for optimizing) and dev (for evaluation)
    # Since we have ~29, let's use 20 for training, 9 for dev
    trainset = dataset[:20]
    devset = dataset[20:]

    extractor_module = ClinicalExtractor()

    print("\nStarting DSPy Prompt Compilation (Bootstrapping)...")
    print("This will take a few minutes as the model evaluates and learns from the trainset.")

    # Setup the optimizer
    teleprompter = BootstrapFewShot(
        metric=extraction_metric,
        max_bootstrapped_demos=3,  # Max number of perfect examples to embed in the prompt
        max_labeled_demos=0
    )

    # Compile the model
    compiled_extractor = teleprompter.compile(student=extractor_module, trainset=trainset)

    # Save the optimized weights (Prompt + Few Shot Examples)
    compiled_extractor.save(OUTPUT_MODEL_PATH)
    print(f"\nOptimization Complete! Saved optimized model to: {OUTPUT_MODEL_PATH}")

    # Optional: Quick test of the compiled model on a dev example
    if devset:
        print("\n--- Testing Optimized Model on a Dev Example ---")
        test_note = devset[0].clinical_note[:500] + "..."
        print(f"Input: {test_note}")
        pred = compiled_extractor(clinical_note=devset[0].clinical_note)
        score = extraction_metric(devset[0], pred)
        print(f"Metric Score on Dev Example: {score:.2f} / 1.0")


if __name__ == "__main__":
    run_optimization()