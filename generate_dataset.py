import json
import os

# ==========================================
# Pseudo Ground Truth Dataset
# Designed specifically for clinical entity extraction ablation study
# ==========================================

dataset = [
    {
        "record_id": "MIMIC-001",
        "original_text": "Patient is a 65yo M presenting with acute ST-segment elevation myocardial infarction (STEMI). PCI was performed immediately. Currently prescribed Aspirin and Clopidogrel. No other medications noted.",
        "golden_labels": {
            "diagnoses": [{"condition": "acute ST-segment elevation myocardial infarction (STEMI)", "certainty": "Confirmed"}],
            "medications": [{"name": "Aspirin", "status": "Current"}, {"name": "Clopidogrel", "status": "Current"}],
            "procedures": [{"name": "PCI", "status": "Performed"}]
        }
    },
    {
        "record_id": "MIMIC-002",
        "original_text": "72-year-old female admitted for acute exacerbation of COPD and suspected community-acquired pneumonia. Patient denies any chest pain. Started on IV Ceftriaxone and Azithromycin. Past medical history significant for Type 2 Diabetes.",
        "golden_labels": {
            "diagnoses": [
                {"condition": "acute exacerbation of COPD", "certainty": "Confirmed"},
                {"condition": "community-acquired pneumonia", "certainty": "Suspected"},
                {"condition": "Type 2 Diabetes", "certainty": "Confirmed"}
            ],
            "medications": [{"name": "Ceftriaxone", "status": "Current"}, {"name": "Azithromycin", "status": "Current"}],
            "symptoms": [{"name": "chest pain", "status": "Absent"}],
            "procedures": []
        }
    },
    {
        "record_id": "MIMIC-003",
        "original_text": "50yo M brought to ER with severe right lower quadrant abdominal pain, nausea, and vomiting. CT scan confirms acute appendicitis. Emergency appendectomy is planned for this evening. Patient is allergic to Penicillin.",
        "golden_labels": {
            "diagnoses": [{"condition": "acute appendicitis", "certainty": "Confirmed"}],
            "allergies": [{"allergen": "Penicillin"}],
            "symptoms": [
                {"name": "right lower quadrant abdominal pain", "status": "Present"},
                {"name": "nausea", "status": "Present"},
                {"name": "vomiting", "status": "Present"}
            ],
            "procedures": [{"name": "appendectomy", "status": "Planned"}]
        }
    },
    {
        "record_id": "MIMIC-004",
        "original_text": "Pt is a 88yo F presenting with sudden onset of left-sided weakness and facial droop. Stroke code activated. MRI brain shows acute ischemic stroke in the right MCA territory. Administered tPA in the ER. Hx of hypertension.",
        "golden_labels": {
            "diagnoses": [
                {"condition": "acute ischemic stroke", "certainty": "Confirmed"},
                {"condition": "hypertension", "certainty": "Confirmed"}
            ],
            "medications": [{"name": "tPA", "status": "Current"}],
            "symptoms": [
                {"name": "left-sided weakness", "status": "Present"},
                {"name": "facial droop", "status": "Present"}
            ],
            "procedures": []
        }
    },
    {
        "record_id": "MIMIC-005",
        "original_text": "45-year-old male arrives with diabetic ketoacidosis (DKA). Blood glucose is 550 mg/dL. He has a known history of Type 1 DM. Insulin drip initiated. Patient reports polydipsia and polyuria over the last 3 days.",
        "golden_labels": {
            "diagnoses": [
                {"condition": "diabetic ketoacidosis (DKA)", "certainty": "Confirmed"},
                {"condition": "Type 1 DM", "certainty": "Confirmed"}
            ],
            "medications": [{"name": "Insulin", "status": "Current"}],
            "symptoms": [
                {"name": "polydipsia", "status": "Present"},
                {"name": "polyuria", "status": "Present"}
            ],
            "procedures": []
        }
    }
]

# Duplicate to simulate a batch of 10 records for testing
expanded_dataset = dataset * 2
for i, data in enumerate(expanded_dataset):
    data["record_id"] = f"MIMIC-{i+1:03d}"

file_path = "/Users/haotingzhaooutlook.com/Desktop/T3/pseudo_ground_truth.json"

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(expanded_dataset, f, indent=4, ensure_ascii=False)

print(f"[SUCCESS] Generated testing dataset with {len(expanded_dataset)} medical records.")
print(f"[INFO] File saved to: {file_path}")