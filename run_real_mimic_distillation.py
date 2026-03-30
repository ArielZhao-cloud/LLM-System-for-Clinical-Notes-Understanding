import os
import json
import pandas as pd
import dspy
from multi_agent_pipeline import (
    ClinicalExtractor, 
    CriticizeExtraction, 
    clean_and_parse_json
)

# ==========================================
# Real MIMIC-IV Knowledge Distillation Engine
# ==========================================

RAW_CSV_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/Data/raw/note/discharge.csv"
DISTILLED_OUTPUT_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/mimic_silver_standard.json"

# Number of real records to process for the evaluation dataset
SAMPLE_SIZE = 10 

def run_real_distillation():
    print("========== Starting LLM Knowledge Distillation on REAL MIMIC-IV ==========")
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"[ERROR] CSV file not found at {RAW_CSV_PATH}")
        print("Please check if the file path and name are exactly correct.")
        return

    print(f"[INFO] Loading raw data from {RAW_CSV_PATH}...")
    
    try:
        # Read the CSV. We only need the first chunk to save memory
        df = pd.read_csv(RAW_CSV_PATH, nrows=SAMPLE_SIZE * 5)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    # Standard MIMIC-IV clinical notes column is usually 'text'
    if 'text' not in df.columns:
        print("[ERROR] Column 'text' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # Drop rows with empty text and take the exact sample size
    df = df.dropna(subset=['text'])
    sampled_df = df.head(SAMPLE_SIZE)

    print(f"[INFO] Successfully sampled {len(sampled_df)} real clinical notes.")
    
    extractor = ClinicalExtractor()
    critic = dspy.Predict(CriticizeExtraction)
    
    distilled_dataset = []

    for index, row in sampled_df.iterrows():
        # Use hadm_id as record ID if available, otherwise use index
        record_id = str(row.get('hadm_id', f"MIMIC-REAL-{index}"))
        text = str(row['text'])
        
        print(f"\nDistilling Record [{len(distilled_dataset)+1}/{SAMPLE_SIZE}]: Admission ID {record_id}")
        
        feedback = "None"
        final_json_str = ""
        
        # Teacher loop: Extract -> Criticize -> Revise (Max 3 attempts)
        for attempt in range(3):
            print(f"  -> Attempt {attempt+1} extraction...")
            try:
                pred = extractor(clinical_note=text, previous_feedback=feedback)
                final_json_str = pred.extracted_json
                
                audit = critic(original_note=text, extracted_json=final_json_str)
                if "Fail" not in audit.audit_result:
                    print("  -> Critic Audit PASSED. High-quality labels secured.")
                    break
                else:
                    print(f"  -> Critic Audit FAILED: {audit.feedback}")
                    feedback = audit.feedback
            except Exception as e:
                print(f"  -> [WARNING] LLM Generation Failed: {e}")
                break
                
        silver_labels = clean_and_parse_json(final_json_str)
        
        formatted_record = {
            "record_id": record_id,
            "original_text": text,
            "golden_labels": {
                "diagnoses": silver_labels.get("diagnoses", []),
                "medications": silver_labels.get("medications", []),
                "procedures": silver_labels.get("procedures", []),
                "allergies": silver_labels.get("allergies", []),
                "symptoms": silver_labels.get("symptoms", [])
            }
        }
        distilled_dataset.append(formatted_record)

    with open(DISTILLED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(distilled_dataset, f, indent=4, ensure_ascii=False)

    print("\n========== Distillation Complete ==========")
    print(f"Successfully generated high-quality silver standard from REAL data.")
    print(f"File saved to: {DISTILLED_OUTPUT_PATH}")

if __name__ == "__main__":
    run_real_distillation()