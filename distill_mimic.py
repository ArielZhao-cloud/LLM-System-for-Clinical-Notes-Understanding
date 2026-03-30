import pandas as pd
import json
import re
import os

# 配置路径
MIMIC_CSV_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/Data/raw/note/discharge.csv"
OUTPUT_JSON_PATH = "mimic_silver_standard.json"

def clean_text(text):
    """移除 MIMIC 中的脱敏标记 [** ... **] 和多余换行"""
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_oncology_samples(path, num_samples=10):
    print(f"Reading MIMIC data from: {path}...")
    # 只读取 text 列以节省内存
    df = pd.read_csv(path, usecols=['text'])
    
    # 定义肿瘤相关关键词
    oncology_keywords = ['carcinoma', 'adenocarcinoma', 'malignant', 'metastatic', 'stage iv', 'biopsy']
    
    # 筛选包含关键词的病历
    mask = df['text'].str.contains('|'.join(oncology_keywords), case=False, na=False)
    onco_df = df[mask].sample(min(num_samples * 5, len(df[mask]))) # 多取一些备选
    
    distilled_data = []
    
    for i, row in onco_df.iterrows():
        raw_text = row['text']
        
        # 尝试截取最重要的部分：现病史和出院诊断（MIMIC 典型结构）
        # 这样可以减少干扰信息，提高提取的 Precision
        sections = re.split(r'(HISTORY OF PRESENT ILLNESS:|DISCHARGE DIAGNOSIS:)', raw_text, flags=re.IGNORECASE)
        if len(sections) > 1:
            extracted_segment = "".join(sections[1:4]) # 获取标题及其内容
        else:
            extracted_segment = raw_text[:2000] # 如果格式不规范，取前2000字
            
        clean_segment = clean_text(extracted_segment)
        
        # 构建一条蒸馏数据
        # 注意：golden_labels 部分由于是原始数据，需要你根据抽出来的这10条手动校对一下
        # 下面给出一个占位符结构
        entry = {
            "record_id": f"MIMIC_DISTILLED_{len(distilled_data)+1}",
            "original_text": clean_segment,
            "golden_labels": {
                "diagnosis": {"site": "CHECK_REQUIRED", "stage": "CHECK_REQUIRED"},
                "biomarkers": [{"name": "CHECK_REQUIRED", "value": "CHECK_REQUIRED"}],
                "treatments": [{"regimen": "CHECK_REQUIRED", "status": "CHECK_REQUIRED"}]
            }
        }
        distilled_data.append(entry)
        if len(distilled_data) >= num_samples:
            break

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(distilled_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully distilled {len(distilled_data)} samples to {OUTPUT_JSON_PATH}")
    print("WARNING: Please open the JSON and manually fill 'golden_labels' based on the 'original_text' before running evaluation.")

if __name__ == "__main__":
    extract_oncology_samples(MIMIC_CSV_PATH)