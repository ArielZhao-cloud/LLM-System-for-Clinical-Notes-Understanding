import json

# ==========================================
# 1. Configuration & Loading
# ==========================================
DATABASE_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/extracted_database.json"


def load_extracted_data():
    try:
        with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")
        return []


# ==========================================
# 2. Mock Ground Truth (For Demonstration)
# In reality, you would manually annotate a few records to serve as the gold standard.
# ==========================================
MOCK_GROUND_TRUTH = {
    # Assuming Record 0 is the patient with chest tightness and hypertension
    0: {
        "diagnoses": ["hypertension", "acute myocardial infarction"],
        "symptoms": ["chest tightness"],
        "medications": ["Amlodipine"]
    }
}


# ==========================================
# 3. Evaluation Logic
# ==========================================
def calculate_metrics(extracted_entities, ground_truth):
    """
    Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    """
    tp = 0
    fp = 0
    fn = 0

    # Check Diagnoses
    extracted_diagnoses = [d.get("condition", "").lower() for d in extracted_entities.get("diagnoses", [])]
    true_diagnoses = [d.lower() for d in ground_truth.get("diagnoses", [])]

    for diagnosis in extracted_diagnoses:
        if diagnosis in true_diagnoses:
            tp += 1
        else:
            fp += 1  # Hallucination / Over-extraction

    for diagnosis in true_diagnoses:
        if diagnosis not in extracted_diagnoses:
            fn += 1  # Missed extraction

    # You can expand this logic to symptoms and medications...

    return tp, fp, fn


def run_evaluation():
    data = load_extracted_data()
    if not data:
        return

    print(f"Loaded {len(data)} records for evaluation.\n")
    print("=" * 60)

    total_tp, total_fp, total_fn = 0, 0, 0

    for record in data:
        record_id = record.get("record_id")

        # Only evaluate records where we have ground truth
        if record_id in MOCK_GROUND_TRUTH:
            print(f"Evaluating Record {record_id}...")

            # Parse the extracted JSON string back into a dictionary
            extracted_json_str = record.get("extracted_json", "{}")
            if isinstance(extracted_json_str, str):
                try:
                    extracted_entities = json.loads(extracted_json_str)
                except:
                    extracted_entities = {}
            else:
                extracted_entities = extracted_json_str

            gt = MOCK_GROUND_TRUTH[record_id]

            tp, fp, fn = calculate_metrics(extracted_entities, gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            print(f"  -> TP: {tp}, FP: {fp}, FN: {fn}")

    # ==========================================
    # 4. Final Math Computation
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL SYSTEM EVALUATION METRICS")
    print("-" * 60)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f} (How accurate were the extractions?)")
    print(f"Recall:    {recall:.4f} (How much of the truth was captured?)")
    print(f"F1-Score:  {f1_score:.4f} (Overall harmonic mean)")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()