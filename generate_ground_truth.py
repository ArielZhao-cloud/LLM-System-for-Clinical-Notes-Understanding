import os
import json
import re
import pandas as pd
import dspy

# ==========================================
# 1. Configuration & Initialization
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Initializing Expert Annotation System with Zhipu GLM-4...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=3000  # Give it maximum room to think and output
)
dspy.settings.configure(lm=my_domestic_model)

DATA_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/Data/raw/note/discharge.csv"
OUTPUT_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/pseudo_ground_truth.json"

SAMPLE_SIZE = 30  # We will generate 30 high-quality labels


# ==========================================
# 2. JSON Cleaning Utility (Crucial for robust generation)
# ==========================================
def clean_and_parse_json(raw_output: str) -> dict:
    cleaned_text = re.sub(r'```json\n?', '', raw_output)
    cleaned_text = re.sub(r'```\n?', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    try:
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"[Warning] Failed to parse generated JSON: {e}")
        return None


# ==========================================
# 3. Define the Expert Annotator Signature
# ==========================================
class ExpertClinicalAnnotator(dspy.Signature):
    """You are a highly experienced Senior Attending Physician acting as a Chief Data Annotator.
        Your task is to meticulously read a raw clinical discharge summary and extract the absolute ground truth.

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

        STRICT ANNOTATION RULES:
        1. EXTREME ACCURACY: Only extract what is explicitly written. Do not infer or guess.
        2. NO DATE MATH: Do not calculate age from DOB.
        3. NO BOILERPLATE: Ignore the generic negative check-boxes in the 'Review of Systems' unless critical to the Chief Complaint.
        4. Provide the exact 'source_quote' from the text for every single entity.
        5. ONLY output the valid JSON. No conversational text.
        """
    clinical_note = dspy.InputField(desc="The raw clinical discharge summary.")
    extracted_json = dspy.OutputField(desc="The perfect, gold-standard JSON matching the exact schema.")


# ==========================================
# 4. Generation Pipeline
# ==========================================
def generate_pseudo_labels():
    print(f"Loading data from: {DATA_PATH}")
    try:
        # Load data, skip the first 10 because we already processed them in testing
        # We will take the next 30 rows
        df = pd.read_csv(DATA_PATH, skiprows=range(1, 11), nrows=SAMPLE_SIZE)
        if 'text' not in df.columns:
            print("Error: 'text' column missing.")
            return
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    print(f"Successfully loaded {len(df)} records. Starting heavy expert annotation...\n")

    # We use ChainOfThought to force the model to reason before outputting JSON
    annotator = dspy.ChainOfThought(ExpertClinicalAnnotator)

    ground_truth_dataset = []

    for index, row in df.iterrows():
        real_id = index + 10  # Adjust ID based on skipped rows
        print("-" * 50)
        print(f"Annotating Record {real_id} [{index + 1}/{SAMPLE_SIZE}]...")

        raw_note = str(row['text'])[:3500]

        try:
            # The model will first output its 'rationale' (thinking process), then the JSON
            prediction = annotator(clinical_note=raw_note)

            parsed_data = clean_and_parse_json(prediction.extracted_json)

            if parsed_data:
                ground_truth_dataset.append({
                    "record_id": real_id,
                    "original_text": raw_note,
                    "golden_labels": parsed_data,
                    "expert_rationale": prediction.rationale  # Save its reasoning for our reference
                })
                print(f"  -> Success! Generated high-quality labels for Record {real_id}.")
            else:
                print(f"  -> Failed to generate valid JSON for Record {real_id}. Skipping.")

        except Exception as e:
            print(f"  -> Error processing Record {real_id}: {e}")

    # ==========================================
    # 5. Save the Dataset
    # ==========================================
    print("\n" + "=" * 50)
    print("Annotation complete. Saving Golden Dataset...")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_dataset, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(ground_truth_dataset)} highly curated records to {OUTPUT_PATH}")
    print("This dataset is now ready for DSPy evaluation and prompt optimization.")


if __name__ == "__main__":
    generate_pseudo_labels()