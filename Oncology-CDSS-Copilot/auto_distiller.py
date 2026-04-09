import pandas as pd
import json
import re
import os
from tqdm import tqdm

MIMIC_CSV_PATH = os.path.expanduser("~/Desktop/T3/Data/raw/note/discharge.csv")
RAW_DATASET_PATH = "oncology_raw_samples.json"

def filter_raw_oncology(num_samples=50):
    print(f"Filtering raw oncology notes from: {MIMIC_CSV_PATH}")
    if not os.path.exists(MIMIC_CSV_PATH): return

    df = pd.read_csv(MIMIC_CSV_PATH, usecols=['text'], low_memory=False)
    
    onco_keywords = ['carcinoma', 'malignant', 'adenocarcinoma', 'metastatic', 'oncology', 'tumor', 'chemotherapy', 'biopsy']
    mask = df['text'].str.contains('|'.join(onco_keywords), case=False, na=False)
    filtered_df = df[mask]
    
    target_df = filtered_df.sample(min(num_samples, len(filtered_df)))
    
    dataset = []
    for i, row in target_df.iterrows():
        clean_text = re.sub(r'\[\*\*.*?\*\*\]', '', row['text'])[:5000]
        dataset.append({
            "sample_id": f"RAW_{i}",
            "text": clean_text
        })

    with open(RAW_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Saved {len(dataset)} raw oncology notes to {RAW_DATASET_PATH}")

if __name__ == "__main__":
    filter_raw_oncology(num_samples=100)