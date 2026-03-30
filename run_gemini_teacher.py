import os
import json
import pandas as pd
from google import genai
from google.genai import types

# ==========================================
# High-Tier Teacher Distillation Engine 
# Powered by Gemini Flash Architecture
# ==========================================

RAW_CSV_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/Data/raw/note/discharge.csv"
DISTILLED_OUTPUT_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/mimic_silver_standard.json"

# STOP AND REPLACE THIS WITH YOUR REAL API KEY
GEMINI_API_KEY = "AIzaSyBC0QYR4hSxLFZpPe8icLcW_g74SIeHAwk" 

SAMPLE_SIZE = 10 

SYSTEM_PROMPT = """You are a world-class Chief Medical Officer. Extract clinical entities from the provided clinical note.
Return ONLY a valid, parseable JSON object adhering STRICTLY to this schema, with no markdown formatting or extra text:
{
  "diagnoses": [ {"condition": "disease name", "certainty": "Confirmed or Suspected or Ruled_Out", "source_quote": "Exact words"} ],
  "medications": [ {"name": "drug name", "status": "Current or Past", "source_quote": "Exact words"} ],
  "procedures": [ {"name": "procedure or surgery", "status": "Performed or Planned", "source_quote": "Exact words"} ]
}
Zero Hallucination. Extract only what is explicitly stated."""

def run_gemini_distillation():
    print("========== Starting High-Tier Knowledge Distillation (Flash) ==========")
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"[ERROR] CSV file not found at {RAW_CSV_PATH}")
        return

    print(f"[INFO] Loading raw data from {RAW_CSV_PATH}...")
    try:
        df = pd.read_csv(RAW_CSV_PATH, nrows=SAMPLE_SIZE * 5)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    df = df.dropna(subset=['text'])
    sampled_df = df.head(SAMPLE_SIZE)
    print(f"[INFO] Successfully sampled {len(sampled_df)} real clinical notes.")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini Client: {e}")
        return
        
    distilled_dataset = []

    for index, row in sampled_df.iterrows():
        record_id = str(row.get('hadm_id', f"MIMIC-REAL-{index}"))
        text = str(row['text'])
        
        print(f"\nDistilling Record [{len(distilled_dataset)+1}/{SAMPLE_SIZE}]: Admission ID {record_id}")
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=text,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_PROMPT
                )
            )
            
            final_json_str = response.text.strip()
            silver_labels = json.loads(final_json_str)
            
            formatted_record = {
                "record_id": record_id,
                "original_text": text,
                "golden_labels": {
                    "diagnoses": silver_labels.get("diagnoses", []),
                    "medications": silver_labels.get("medications", []),
                    "procedures": silver_labels.get("procedures", [])
                }
            }
            distilled_dataset.append(formatted_record)
            print("  -> Extraction successful via gemini-2.5-flash.")
            
        except Exception as e:
            print(f"  -> [WARNING] 2.5-flash failed, attempting fallback to gemini-1.5-flash...")
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=text,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        system_instruction=SYSTEM_PROMPT
                    )
                )
                final_json_str = response.text.strip()
                silver_labels = json.loads(final_json_str)
                
                formatted_record = {
                    "record_id": record_id,
                    "original_text": text,
                    "golden_labels": {
                        "diagnoses": silver_labels.get("diagnoses", []),
                        "medications": silver_labels.get("medications", []),
                        "procedures": silver_labels.get("procedures", [])
                    }
                }
                distilled_dataset.append(formatted_record)
                print("  -> Fallback extraction successful via gemini-1.5-flash.")
            except Exception as e_fallback:
                print(f"  -> [ERROR] Complete failure on this record: {e_fallback}")

    with open(DISTILLED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(distilled_dataset, f, indent=4, ensure_ascii=False)

    print("\n========== Distillation Complete ==========")
    print(f"Successfully generated superior silver standard dataset.")
    print(f"File saved to: {DISTILLED_OUTPUT_PATH}")

if __name__ == "__main__":
    run_gemini_distillation()