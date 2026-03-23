import os
import json
import pandas as pd
import concurrent.futures
from multi_agent_pipeline import clinical_system

# ==========================================
# 1. Configuration & Paths
# ==========================================
DATA_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/Data/raw/note/discharge.csv"
OUTPUT_PATH = "/Users/haotingzhaooutlook.com/Desktop/T3/extracted_database.json"

# 为了测试并发效果，我们可以把测试数据量稍微调大一点
TEST_BATCH_SIZE = 10
# 设置并发线程数。注意：不要设置过大，否则会触发大模型 API 的并发速率限制 (Rate Limit)
MAX_WORKERS = 5


def process_single_record(index, text_content):
    """
    独立的工作函数，用于处理单条病历。
    将被丢入线程池中并发执行。
    """
    print(f"[Thread-Start] Processing Record {index}...")

    # 截断超长文本以适应上下文窗口
    raw_note = str(text_content)[:3000]

    initial_state = {
        "original_text": raw_note,
        "revision_count": 0,
        "status": "processing",
        "critic_feedback": "None"
    }

    try:
        # 调用多智能体系统
        final_state = clinical_system.invoke(initial_state)

        # 组装结果
        record_output = {
            "record_id": index,
            "original_text_snippet": raw_note[:200] + "...",
            "extracted_json": final_state.get('extracted_entities'),
            "clinical_summary": final_state.get('current_summary'),
            "revision_iterations": final_state.get('revision_count')
        }

        print(f"[Thread-Done] Record {index} completed. (Iterations: {final_state.get('revision_count')})")
        return record_output

    except Exception as e:
        print(f"[Thread-Error] Failed on Record {index}: {e}")
        return {
            "record_id": index,
            "error": str(e)
        }


def process_clinical_notes_concurrently():
    print(f"Loading data from: {DATA_PATH}")

    try:
        df = pd.read_csv(DATA_PATH, nrows=TEST_BATCH_SIZE)
        if 'text' not in df.columns:
            print("Error: Could not find 'text' column in the CSV.")
            return
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    print(f"Successfully loaded {len(df)} records.")
    print(f"Starting CONCURRENT multi-agent processing with {MAX_WORKERS} workers...\n")
    print("=" * 60)

    results_database = []

    # ==========================================
    # 2. Concurrent Batch Processing
    # ==========================================
    # 使用 ThreadPoolExecutor 管理并发线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到线程池
        # 我们将 index 和 text 传递给工作函数
        future_to_record = {
            executor.submit(process_single_record, index, row['text']): index
            for index, row in df.iterrows()
        }

        # as_completed 会在某个线程完成任务时立即返回结果
        for future in concurrent.futures.as_completed(future_to_record):
            record_id = future_to_record[future]
            try:
                result = future.result()
                results_database.append(result)
            except Exception as exc:
                print(f"Record {record_id} generated an exception: {exc}")

    # ==========================================
    # 3. Save Results to Local JSON
    # ==========================================
    print("\n" + "=" * 60)
    print("Batch processing complete. Saving to database...")

    # 按照 record_id 对结果进行排序，因为多线程完成的顺序是乱序的
    results_database = sorted(results_database, key=lambda x: x.get('record_id', 999999))

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_database, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(results_database)} processed records to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_clinical_notes_concurrently()