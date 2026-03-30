import os
import dspy
import json
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

# 必须从你的推理脚本中导入相同的 Signature 和清洗函数
# 确保 ExtractEntities 的定义在两个文件中完全一致
from multi_agent_pipeline import ExtractEntities, clean_and_parse_json

# 1. 环境初始化
load_dotenv()
api_key = os.environ.get("ZHIPU_API_KEY", "")
lm = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=4096
)
dspy.settings.configure(lm=lm)

# 2. 封装训练模块 (必须与推理时的结构镜像)
class ClinicalExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 ChainOfThought 确保推理路径也被优化
        self.extract = dspy.ChainOfThought(ExtractEntities)
        
    def forward(self, clinical_note, previous_feedback="None"):
        return self.extract(clinical_note=clinical_note, previous_feedback=previous_feedback)

# 3. 加载训练数据 (对齐 Signature 的字段)
def load_training_data(filepath: str):
    print(f"[Loading] Reading training data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trainset = []
    for item in data:
        # 核心：这里的字段名必须与 ExtractEntities 的 InputField/OutputField 完全匹配
        ex = dspy.Example(
            clinical_note=item["original_text"],
            previous_feedback="None", # 初始状态设为 None
            # 目标输出：虽然训练集可能没推理过程，但要给一个占位符或让模型自己生成
            reasoning="The clinical note describes a patient with several conditions...",
            extracted_json=json.dumps(item["golden_labels"], ensure_ascii=False)
        ).with_inputs('clinical_note', 'previous_feedback') # 明确输入字段
        trainset.append(ex)
    return trainset

# 4. 定义优化指标 (医学实体提取专用 Metric)
def medical_metric(example, pred, trace=None):
    """
    评估提取出的 JSON 与黄金标准在 Diagnoses 上的重合度。
    """
    try:
        # 使用你的清洗函数解析模型输出
        pred_dict = clean_and_parse_json(pred.extracted_json)
        truth_dict = json.loads(example.extracted_json)
        
        truth_set = set(d.get("condition", "").lower() for d in truth_dict.get("diagnoses", []))
        pred_set = set(d.get("condition", "").lower() for d in pred_dict.get("diagnoses", []))
        
        if not truth_set: return 1.0 if not pred_set else 0.0
        
        intersection = truth_set.intersection(pred_set)
        recall = len(intersection) / len(truth_set)
        
        return recall # 以召回率为主要优化目标
    except:
        return 0.0

# 5. 执行优化过程 (Compilation)
def run_optimization():
    # 数据路径请根据实际情况修改
    TRAIN_DATA_PATH = "pseudo_ground_truth.json" 
    OUTPUT_WEIGHTS_PATH = "optimized_extractor.json"

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[Error] Training data {TRAIN_DATA_PATH} not found!")
        return

    trainset = load_training_data(TRAIN_DATA_PATH)
    
    # 使用 BootstrapFewShot 算法
    # max_bootstrapped_demos: 模型自己生成的成功案例数
    # max_labeled_demos: 从训练集中直接选取的案例数
    config = BootstrapFewShot(
        metric=medical_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    print("\n[Optimize] Starting DSPy compilation process...")
    teleprompter = config.compile(ClinicalExtractorModule(), trainset=trainset)
    
    # 6. 保存权重
    teleprompter.save(OUTPUT_WEIGHTS_PATH)
    print(f"\n[Success] Optimized weights saved to: {OUTPUT_WEIGHTS_PATH}")

if __name__ == "__main__":
    run_optimization()